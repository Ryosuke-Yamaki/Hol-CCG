import torch.nn as nn
from tqdm import tqdm
import torch
from typing import List
from holccg import HolCCG
from tree import Tree


@torch.no_grad()
def evaluate_tree_list(tree_list, holccg):
    for tree in tree_list.tree_list:
        tree.set_word_split(holccg.tokenizer)
    tree_list.set_vector(holccg)
    word_classifier = holccg.word_classifier
    phrase_classifier = holccg.phrase_classifier
    num_word = 0
    num_phrase = 0
    num_correct_word = 0
    num_correct_phrase = 0
    with tqdm(total=len(tree_list.tree_list)) as pbar:
        pbar.set_description("evaluating...")
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    output = word_classifier(node.vector)
                    predict = torch.topk(output, k=1)[1]
                    num_word += 1
                    if node.category_id in predict and node.category_id != 0:
                        num_correct_word += 1
                else:
                    output = phrase_classifier(node.vector)
                    predict = torch.topk(output, k=1)[1]
                    num_phrase += 1
                    if node.category_id in predict and node.category_id != 0:
                        num_correct_phrase += 1
            pbar.update(1)
    word_acc = num_correct_word / num_word
    phrase_acc = num_correct_phrase / num_phrase
    print('-' * 50)
    print('word acc: {}'.format(word_acc))
    print('phrase acc: {}'.format(phrase_acc))
    return word_acc, phrase_acc


@torch.no_grad()
def evaluate_batch_list(
        batch_list: list,
        holccg: HolCCG) -> dict:
    """evaluate the model by the list of batch

    Parameters
    ----------
    batch_list : list
        list of batch
    holccg : HolCCG
        Hol-CCG model

    Returns
    -------
    dict
        evaluation result
    """
    num_word = 0
    num_phrase = 0
    num_span = 0
    num_correct_word = 0
    num_correct_phrase = 0
    num_correct_span = 0
    word_loss = 0
    phrase_loss = 0
    span_loss = 0
    criteria = nn.CrossEntropyLoss()
    with tqdm(total=len(batch_list), unit="batch") as pbar:
        pbar.set_description("evaluating...")
        for batch in batch_list:
            word_output, phrase_output, span_output, word_label, phrase_label, span_label = holccg(
                batch)

            num_word += word_output.shape[0]
            num_phrase += phrase_output.shape[0]
            num_span += span_output.shape[0]

            word_loss += criteria(word_output, word_label)
            phrase_loss += criteria(phrase_output, phrase_label)
            span_loss += criteria(span_output, span_label)

            # remove unknown categories
            word_output = word_output[word_label != 0]
            phrase_output = phrase_output[phrase_label != 0]
            word_label = word_label[word_label != 0]
            phrase_label = phrase_label[phrase_label != 0]

            num_correct_word += torch.count_nonzero(torch.argmax(word_output, dim=1) == word_label)
            num_correct_phrase += torch.count_nonzero(
                torch.argmax(phrase_output, dim=1) == phrase_label)
            num_correct_span += torch.count_nonzero(torch.argmax(span_output, dim=1) == span_label)

            pbar.update(1)

    word_loss = word_loss / len(batch_list)
    phrase_loss = phrase_loss / len(batch_list)
    span_loss = span_loss / len(batch_list)
    total_loss = word_loss + phrase_loss + span_loss
    word_acc = num_correct_word / num_word
    phrase_acc = num_correct_phrase / num_phrase
    span_acc = num_correct_span / num_span

    stat = {
        "total_loss": total_loss,
        "word_acc": word_acc,
        "phrase_acc": phrase_acc,
        "span_acc": span_acc,
        "word_loss": word_loss,
        "phrase_loss": phrase_loss,
        "span_loss": span_loss}

    print("word_acc:{}\nphrase_acc:{}\nspan_acc:{}".format(
        stat["word_acc"], stat["phrase_acc"], stat["span_acc"]))

    return stat


@torch.no_grad()
def evaluate_multi(tree_list, holccg, gamma=0.1, alpha=10):
    word_classifier = holccg.word_classifier
    phrase_classifier = holccg.phrase_classifier
    num_word = 0
    num_phrase = 0
    num_correct_word = 0
    num_correct_phrase = 0
    num_predicted_word = 0
    num_predicted_phrase = 0
    with tqdm(total=len(tree_list.tree_list)) as pbar:
        pbar.set_description("evaluating...")
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    # probability distribution
                    output = torch.softmax(word_classifier(node.vector), dim=-1)
                    predict_prob, predict_idx = torch.sort(output[1:], dim=-1, descending=True)
                    # add one to index for the removing of zero index of "<UNK>"
                    predict_idx += 1
                    predict_idx = predict_idx[predict_prob >= gamma][:alpha]
                    num_word += 1
                    num_predicted_word += predict_idx.shape[0]
                    if node.category_id in predict_idx and node.category_id != 0:
                        num_correct_word += 1
                else:
                    output = torch.softmax(phrase_classifier(node.vector), dim=-1)
                    predict_prob, predict_idx = torch.sort(output[1:], dim=-1, descending=True)
                    predict_idx += 1
                    predict_idx = predict_idx[predict_prob >= gamma][:alpha]
                    num_phrase += 1
                    num_predicted_phrase += predict_idx.shape[0]
                    if node.category_id in predict_idx and node.category_id != 0:
                        num_correct_phrase += 1
            pbar.update(1)
    print('-' * 50)
    print(f'gamma={gamma}, alpha={alpha}')
    print(f'word: {num_correct_word / num_word}')
    print(f'cat per word: {num_predicted_word / num_word}')
    print(f'phrase: {num_correct_phrase / num_phrase}')
    print(f'cat per phrase: {num_predicted_phrase / num_phrase}')

    return num_correct_word / num_word, num_predicted_word / num_word


@torch.no_grad()
def evaluate_stag(tree_list: List[Tree], holccg: HolCCG) -> float:
    """Evaluate supertagging accuracy.

    Parameters
    ----------
    tree_list : List[Tree]
        list of trees
    holccg : HolCCG
        holccg model

    Returns
    -------
    float
        supertagging accuracy
    """
    word_classifier = holccg.word_classifier
    word_category_vocab = tree_list.word_category_vocab
    num_word = 0
    num_correct_word = 0
    with tqdm(total=len(tree_list.tree_list)) as pbar:
        pbar.set_description("evaluating supertag...")
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    predict_prob = torch.softmax(word_classifier(node.vector), dim=-1)
                    predict_idx = torch.argmax(predict_prob, dim=-1)
                    num_word += 1
                    if predict_idx != 0:
                        predict_cat = word_category_vocab.itos[predict_idx]
                        predict_prime_cat = predict_cat.split('-->')[0]
                        if predict_prime_cat == node.prime_category:
                            num_correct_word += 1
            pbar.update(1)
    print(f'supertagging_acc: {num_correct_word / num_word}')
    return num_correct_word / num_word
