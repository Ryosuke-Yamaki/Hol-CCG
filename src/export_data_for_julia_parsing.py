import numpy as np
from utils import load, dump, Condition_Setter
import torch


def extract_rule(path_to_grammar, category_vocab):
    binary_rule = []
    unary_rule = []
    unk_idx = category_vocab.unk_index + 1

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    min_freq = 10
    for rule in data:
        tokens = rule.split()
        freq = int(tokens[0])
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]] + 1
            left_cat = category_vocab[tokens[4]] + 1
            right_cat = category_vocab[tokens[5]] + 1
            if freq >= min_freq and unk_idx not in [left_cat, right_cat, parent_cat]:
                binary_rule.append([left_cat, right_cat, parent_cat])
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]] + 1
            child_cat = category_vocab[tokens[4]] + 1
            if freq >= min_freq and unk_idx not in [child_cat, parent_cat]:
                unary_rule.append([child_cat, parent_cat])
    return binary_rule, unary_rule


condition = Condition_Setter()
PATH_TO_DIR = condition.PATH_TO_DIR

path_to_grammar = PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar"

device = torch.device('cpu')

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)

correct_list = []
for tree in test_tree_list.tree_list:
    correct_list.append(tree.correct_parse())
dump(correct_list, PATH_TO_DIR + "Hol-CCG/data/parsing/correct_list.pkl")

binary_rule, unary_rule = extract_rule(condition.path_to_grammar, test_tree_list.category_vocab)

np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/binary_rule.txt", np.array(binary_rule),
           fmt='%d', header=str(len(test_tree_list.category_vocab) - 1), comments="")
np.savetxt(PATH_TO_DIR + "Hol-CCG/data/parsing/unary_rule.txt", np.array(unary_rule),
           fmt='%d', header=str(len(test_tree_list.category_vocab) - 1), comments="")


tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

linear_weight = tree_net.phrase_classifier.weight
linear_bias = tree_net.phrase_classifer.bias

np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/linear_weight_{}_{}d.csv".format(
        condition.embedding_type,
        condition.embedding_dim),
    linear_weight.detach().numpy())
np.savetxt(
    PATH_TO_DIR +
    "Hol-CCG/data/parsing/linear_bias_{}_{}d.csv".format(
        condition.embedding_type,
        condition.embedding_dim),
    linear_bias.detach().numpy())
