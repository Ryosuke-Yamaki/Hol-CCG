import numpy as np
from torch._C import ParameterDict
from utils import load, dump, Condition_Setter
import torch


def extract_rule(path_to_grammar, category_vocab):
    binary_rule = []
    unary_rule = []
    binary_stat = {}
    unary_stat = {}
    unk_idx = category_vocab.unk_index + 1

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    min_freq = 1
    for rule in data:
        tokens = rule.split()
        freq = int(tokens[0])
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]] + 1
            left_cat = category_vocab[tokens[4]] + 1
            right_cat = category_vocab[tokens[5]] + 1
            if freq >= min_freq and unk_idx not in [left_cat, right_cat, parent_cat]:
                if parent_cat not in binary_stat:
                    binary_stat[parent_cat] = {"total_freq": freq,
                                               "pair_freq": {(left_cat, right_cat): freq}}
                else:
                    binary_stat[parent_cat]["total_freq"] += freq
                    binary_stat[parent_cat]["pair_freq"][(left_cat, right_cat)] = freq
                binary_rule.append([left_cat, right_cat, parent_cat])
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]] + 1
            child_cat = category_vocab[tokens[4]] + 1
            if freq >= min_freq and unk_idx not in [child_cat, parent_cat]:
                if parent_cat not in unary_stat:
                    unary_stat[parent_cat] = {"total_freq": freq, "child_freq": {child_cat: freq}}
                else:
                    unary_stat[parent_cat]["total_freq"] += freq
                    unary_stat[parent_cat]["child_freq"][child_cat] = freq
                unary_rule.append([child_cat, parent_cat])
    binary_prob = []
    unary_prob = []
    for parent_cat, stat in binary_stat.items():
        total_freq = stat["total_freq"]
        for child_pair, freq in stat["pair_freq"].items():
            binary_prob.append([parent_cat, child_pair[0], child_pair[1], freq / total_freq])
    for parent_cat, stat in unary_stat.items():
        total_freq = stat["total_freq"]
        for child_cat, freq in stat["child_freq"].items():
            unary_prob.append([parent_cat, child_cat, freq / total_freq])
    return binary_rule, unary_rule, binary_prob, unary_prob


condition = Condition_Setter(set_embedding_type=False)
PATH_TO_DIR = condition.PATH_TO_DIR

path_to_grammar = PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar"

device = torch.device('cpu')

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)
whole_category_vocab = load(condition.path_to_whole_category_vocab)

correct_list = []
for tree in test_tree_list.tree_list:
    correct_list.append(tree.correct_parse(whole_category_vocab))
dump(correct_list, PATH_TO_DIR + "Hol-CCG/data/parsing/correct_list.pkl")

binary_rule, unary_rule, binary_prob, unary_prob = extract_rule(
    condition.path_to_grammar, whole_category_vocab)

np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/binary_rule.txt", np.array(binary_rule),
           fmt='%d', header=str(len(whole_category_vocab)), comments="")
np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/unary_rule.txt", np.array(unary_rule),
           fmt='%d', header=str(len(whole_category_vocab)), comments="")
np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/binary_prob.txt", np.array(binary_prob),
           fmt='%f', header=str(len(whole_category_vocab)), comments="")
np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/unary_prob.txt", np.array(unary_prob),
           fmt='%f', header=str(len(whole_category_vocab)), comments="")


tree_net = torch.load("lstm_with_two_classifiers.pth",
                      map_location=device)
tree_net.eval()

word_classifier_weight = tree_net.word_classifier.weight
word_classifier_bias = tree_net.word_classifier.bias
phrase_classifier_weight = tree_net.phrase_classifier.weight
phrase_classifier_bias = tree_net.phrase_classifier.bias

np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/word_classifier_weight.csv",
    word_classifier_weight.detach().numpy())
np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/word_classifier_bias.csv",
    word_classifier_bias.detach().numpy())
np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/phrase_classifier_weight.csv",
    phrase_classifier_weight.detach().numpy())
np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/phrase_classifier_bias.csv",
    phrase_classifier_bias.detach().numpy())
