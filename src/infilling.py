import spacy
import benepar
from transformers import RobertaTokenizer, RobertaForMaskedLM
import tqdm
from utils import load, inverse_circular_correlation
import torch
from torch.nn.functional import cosine_similarity as cos
import random
import argparse
import os


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_holccg', type=str, help='path to trained Hol-CCG')
    parser.add_argument('--path_to_roberta', type=str, help='path to trained RoBERTa')
    parser.add_argument('--path_to_dataset', type=str, default='../dataset/', help='path to dataset')
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device('cuda:0'),
        help='device to use')

    args = parser.parse_args()
    return args


def initialize_berkeley_parser():
    """Initialize berkeley parser
    """
    benepar.download('benepar_en3_large')
    berkeley_parser = spacy.load('en_core_web_md')
    if spacy.__version__.startswith('2'):
        berkeley_parser.add_pipe(benepar.BeneparComponent("benepar_en3_large"))
    else:
        berkeley_parser.add_pipe("benepar", config={"model": "benepar_en3_large"})
    return berkeley_parser


def infilling_with_holccg(args, berkeley_parser):
    holccg = torch.load(args.path_to_holccg, map_location=args.device)
    holccg.device = args.device

    dev_tree_list = load(os.path.join(args.path_to_dataset, 'tree_list/dev_tree_list.pickle'))
    dev_tree_list.tokenizer = holccg.tokenizer
    dev_tree_list.set_info_for_training(tokenizer=holccg.tokenizer)
    with torch.no_grad():
        dev_tree_list.set_vector(holccg)

    # store vectors and contents for each node
    vector_list = []
    content_list = []
    for tree in dev_tree_list.tree_list:
        for node in tree.node_list:
            if not node.is_leaf:
                vector_list.append(node.vector)
                content_list.append(node.content)
    vector_list = torch.stack(vector_list)

    # filter out sentences whose length is longer than 30 or shorter than 10
    tree_list = []
    for tree in dev_tree_list.tree_list:
        if len(tree.sentence) > 30 or len(tree.sentence) < 10:
            continue
        tree_list.append(tree)

    # randomly sample 1 nodes from each tree whose content is longer than 2 and shorter than 6
    for tree in tree_list:
        while True:
            node = random.choice(tree.node_list)
            if node.is_leaf or node.parent_node is None:
                continue
            elif len(node.content) < 2 or len(node.content) > 6:
                continue
            else:
                break
        tree.replace_target_node = node
        parent_node = tree.replace_target_node.parent_node
        if tree.node_list[parent_node.left_child_node_id] == tree.replace_target_node:
            tree.replace_target_node.lr = 'l'
            tree.sibling_node = tree.node_list[parent_node.right_child_node_id]
            reconstruct_vector = inverse_circular_correlation(parent_node.vector,
                                                              tree.sibling_node.vector,
                                                              holccg.vector_norm,
                                                              child_is_left=False)
            tree.replace_target_node.reconstruct_vector = reconstruct_vector
        else:
            tree.replace_target_node.lr = 'r'
            tree.sibling_node = tree.node_list[parent_node.left_child_node_id]
            reconstruct_vector = inverse_circular_correlation(parent_node.vector,
                                                              tree.sibling_node.vector,
                                                              holccg.vector_norm,
                                                              child_is_left=True)
            tree.replace_target_node.reconstruct_vector = reconstruct_vector

    # find top k similar vectors and store their words or phrases
    for tree in tree_list:
        vector = tree.replace_target_node.reconstruct_vector
        content = tree.replace_target_node.content
        # search top k similar vector from vector_list
        k = 1
        vector = vector.unsqueeze(0)
        similarity = cos(vector, vector_list)
        top_k = torch.topk(similarity, k + 1)
        top_k_index = top_k.indices[1:]
        tree.replace_target_node.infilled_content = []
        for idx in top_k_index:
            tree.replace_target_node.infilled_content.append(content_list[idx])
        tree.original_sentence = ' '.join(tree.sentence)
        tree.infilled_sentences = []
        for content in tree.replace_target_node.infilled_content:
            infilled_sentence = tree.original_sentence.replace(
                ' '.join(tree.replace_target_node.content), ' '.join(content))
            tree.infilled_sentences.append(infilled_sentence)

    # parse infilled sentences with berkeley parser
    with tqdm.tqdm(total=len(tree_list)) as pbar:
        pbar.set_description('Parsing infilled sentences with Berkeley Parser')
        for tree in tree_list:
            doc = berkeley_parser(tree.original_sentence)
            sent = list(doc.sents)[0]
            tree.original_tree = sent
            tree.infilled_trees = []
            for infilled_sentence in tree.infilled_sentences:
                doc = berkeley_parser(infilled_sentence)
                sent = list(doc.sents)[0]
                tree.infilled_trees.append(sent)
            pbar.update(1)

    # extract non-terminal symbols of infilled phrases
    for tree in tree_list:
        target_phrase = ' '.join(tree.replace_target_node.content)
        tree.replace_target_node.original_symbol = None
        for constituent in tree.original_tree._.constituents:
            if constituent.text == target_phrase:
                tree.replace_target_node.original_symbol = str(constituent._.labels[0])
        tree.replace_target_node.infilled_symbols = []
        for target_phrase, infilled_tree in zip(tree.replace_target_node.infilled_content, tree.infilled_trees):
            target_phrase = ' '.join(target_phrase)
            infilled_symbol = None
            for constituent in infilled_tree._.constituents:
                if constituent.text == target_phrase:
                    infilled_symbol = str(constituent._.labels[0])
                    break
            tree.replace_target_node.infilled_symbols.append(infilled_symbol)

    # calculate match rate between original symbols and infilled symbols
    num_match = 0
    num_umatch = 0
    for tree in tree_list:
        if tree.replace_target_node.original_symbol is None:
            continue
        else:
            for infilled_symbol in tree.replace_target_node.infilled_symbols:
                if infilled_symbol == tree.replace_target_node.original_symbol:
                    num_match += 1
                else:
                    num_umatch += 1
    match_rate = num_match / (num_match + num_umatch)
    print('num match: {}'.format(num_match))
    print('num umatch: {}'.format(num_umatch))
    print('match rate: {:.2f}%'.format(match_rate * 100))

    return tree_list


def infilling_with_roberta(args, berkeley_parser, tree_list):
    state_dict = torch.load(args.path_to_roberta)
    tokenizer = RobertaTokenizer.from_pretrained('roberta-large')
    model = RobertaForMaskedLM.from_pretrained('roberta-large')
    model.load_state_dict(state_dict)
    model.to(args.device)

    # infilling with roberta
    with tqdm.tqdm(total=len(tree_list)) as pbar:
        pbar.set_description('Infilling with RoBERTa')
        for tree in tree_list:
            original_sentence = tree.original_sentence
            target_phrase = ' '.join(tree.replace_target_node.content)
            tokenized_target_phrase = tokenizer.tokenize(target_phrase)
            mask_tokens = ''.join([tokenizer.mask_token] * len(tokenized_target_phrase))
            masked_sentence = original_sentence.replace(target_phrase, mask_tokens)
            input_ids = torch.tensor([tokenizer.encode(masked_sentence, add_special_tokens=True)]).to(args.device)
            predicted_tokens = []
            for i in range(len(tokenized_target_phrase)):
                with torch.no_grad():
                    outputs = model(input_ids)
                    logits = outputs.logits
                # get the index of the masked token
                masked_index = (input_ids == tokenizer.mask_token_id).nonzero()[0, 1]
                # get the logits of the masked token
                logits = logits[0, masked_index, :]
                # get the top predicted token
                predicted_index = torch.argmax(logits).item()
                predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]
                predicted_tokens.append(predicted_token)
                # replace input_ids with the predicted token
                input_ids[0, masked_index] = predicted_index
            tree.roberta_infilled_sentence = tokenizer.decode(input_ids[0, 1:-1])
            infilled_phrase = tokenizer.convert_tokens_to_string(predicted_tokens).strip()
            tree.replace_target_node.roberta_infilled_phrase = infilled_phrase
            pbar.update(1)

    # parse infilled sentences with berkeley parser
    with tqdm.tqdm(total=len(tree_list)) as pbar:
        pbar.set_description('Parsing infilled sentences with Berkeley Parser')
        for tree in tree_list:
            doc = berkeley_parser(tree.roberta_infilled_sentence)
            sent = list(doc.sents)[0]
            tree.roberta_infilled_tree = sent
            pbar.update(1)

    # extract non-terminal symbols of infilled phrases
    for tree in tree_list:
        target_phrase = tree.replace_target_node.roberta_infilled_phrase
        infilled_symbol = None
        for constituent in tree.roberta_infilled_tree._.constituents:
            if constituent.text == target_phrase:
                if len(constituent._.labels) > 0:
                    infilled_symbol = str(constituent._.labels[0])
                    break
        tree.replace_target_node.roberta_infilled_symbol = infilled_symbol

    # calculate match rate between original symbols and infilled symbols
    num_match = 0
    num_umatch = 0
    for tree in tree_list:
        if tree.replace_target_node.original_symbol is None:
            continue
        else:
            if tree.replace_target_node.roberta_infilled_symbol == tree.replace_target_node.original_symbol:
                num_match += 1
            else:
                num_umatch += 1
    match_rate = num_match / (num_match + num_umatch)
    print('num_match: {}'.format(num_match))
    print('num_umatch: {}'.format(num_umatch))
    print('match rate: {:.2f}%'.format(match_rate * 100))


def main():
    args = arg_parse()
    berkeley_parser = initialize_berkeley_parser()
    tree_list = infilling_with_holccg(args, berkeley_parser)
    infilling_with_roberta(args, berkeley_parser, tree_list)


if __name__ == '__main__':
    main()
