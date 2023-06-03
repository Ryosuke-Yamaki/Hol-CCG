from utils import load
import torch
import argparse
import os


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_sentence', type=str, help='path to sentence to be supertagged')
    parser.add_argument('--path_to_model', type=str, help='path to model used for supertagging')
    parser.add_argument('--path_to_pos', type=str, default=None, help='path to pos tagged file')
    parser.add_argument('--path_to_dataset', type=str, default='../dataset/', help='path to dataset')
    parser.add_argument('--stag_threshold', type=float, default=0.1, help='threshold for supertagging')
    parser.add_argument('--print_probability', action='store_true', help='print probability of supertags')
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device('cuda:0'),
        help='device to use for supertagging')
    args = parser.parse_args()
    return args


def convert_bracket(content: str) -> str:
    """convert bracket to readable format

    Parameters
    ----------
    content : str
        content to be converted

    Returns
    -------
    str
        converted content
    """
    if content == "-LRB-":
        content = "("
    elif content == "-LCB-":
        content = "{"
    elif content == "-RRB-":
        content = ")"
    elif content == "-RCB-":
        content = "}"
    return content


def convert_slash(content: str) -> str:
    """convert slash to readable format

    Parameters
    ----------
    content : str
        content to be converted

    Returns
    -------
    str
        converted content
    """
    if r"\/" in content:
        content = content.replace(r"\/", "/")
    return content


def main():
    args = arg_parse()

    word_category_vocab = load(os.path.join(args.path_to_dataset, "grammar/word_category_vocab.pickle"))

    holccg = torch.load(args.path_to_model, map_location=args.device)
    holccg.device = args.device
    holccg.eval()

    word_classifier = holccg.word_classifier

    with open(args.path_to_sentence, "r") as f:
        sentence_list = f.readlines()
    if args.path_to_pos is None:
        # make the same shape POS list as sentence_list
        pos_list = []
        for sentence in sentence_list:
            pos_list.append(['POS'] * len(sentence.split()))
    else:
        pos_list = []
        with open(args.path_to_pos, "r") as f:
            lines = f.readlines()
            for pos in lines:
                pos_tags = []
                pos = pos.strip().split()
                pos_tags.append(pos.split('|')[1])
            pos_list.append(pos_tags)

    with torch.no_grad():
        for sentence, pos_tags in zip(sentence_list, pos_list):
            sentence = sentence.split()
            converted_sentence_for_supertagging = []
            converted_sentence_for_print = []
            for i in range(len(sentence)):
                content = sentence[i]
                converted_sentence_for_supertagging.append(convert_slash(convert_bracket(content)))
                converted_sentence_for_print.append(convert_bracket(content))
            word_split = holccg.set_word_split(converted_sentence_for_supertagging)
            word_vectors, _ = holccg.encode([" ".join(converted_sentence_for_supertagging)], [word_split])
            word_cat_prob = torch.softmax(word_classifier(word_vectors[0]), dim=-1)
            predict_cat_id = torch.argsort(word_cat_prob, descending=True)
            # remove '<unk>'
            predict_cat_id = predict_cat_id[predict_cat_id != 0].view(word_cat_prob.shape[0], -1)
            super_tags = []
            for idx in range(word_cat_prob.shape[0]):
                # add top probability category
                temp = [[word_category_vocab.itos[predict_cat_id[idx, 0]].split('-->')[0],
                        word_cat_prob[idx, predict_cat_id[idx, 0]].item()]]
                for cat_id in predict_cat_id[idx, 1:]:
                    if word_cat_prob[idx, cat_id] > args.stag_threshold:
                        temp.append([word_category_vocab.itos[cat_id].split('-->')[0],
                                    word_cat_prob[idx, cat_id].item()])
                    else:
                        break
                super_tags.append(temp)

            line = []
            for word, pos, super in zip(converted_sentence_for_print, pos_tags, super_tags):
                temp = []
                temp.append(word)
                temp.append(pos)
                for info in super:
                    temp.append(info[0])
                    if args.print_probability:
                        temp.append(str(info[1]))
                line.append('|'.join(temp))
            print(' '.join(line))


if __name__ == "__main__":
    main()
