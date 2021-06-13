from torchtext.vocab import Vocab
import pickle
from collections import Counter
from tokenize import Special
import torch
import numpy as np
import csv
import gensim.downloader as api
from models import Node, Tree
from utils import Condition_Setter
import os


def set_tree_list(PATH_TO_DATA):
    tree_list = []
    tree_id = 0
    node_list = []
    with open(PATH_TO_DATA, 'r') as f:
        node_info_list = [node_info.strip() for node_info in f.readlines()]
    node_info_list = [node_info.replace(
        '\n', '') for node_info in node_info_list]
    for node_info in node_info_list:
        if node_info != '':
            node = Node(node_info.split())
            node_list.append(node)
        elif node_list != []:
            tree_list.append(Tree(tree_id, node_list))
            node_list = []
            tree_id += 1
    return tree_list


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

PATH_TO_PRETRAINED_WEIGHT_MATRIX = PATH_TO_DIR + \
    "Hol-CCG/data/glove_{}d.csv".format(condition.embedding_dim)
path_to_word_counter = PATH_TO_DIR + "Hol-CCG/data/word_counter.pickle"

device = torch.device('cpu')
path_to_train_data = condition.path_to_train_data
path_to_dev_data = condition.path_to_dev_data
path_to_test_data = condition.path_to_test_data
print('loading tree list...')
train_tree_list = set_tree_list(path_to_train_data)
dev_tree_list = set_tree_list(path_to_dev_data)
test_tree_list = set_tree_list(path_to_test_data)

tree_list = train_tree_list + dev_tree_list + test_tree_list

word_counter = Counter()
for tree in tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            word_counter[node.content] += 1

vocab = Vocab(word_counter, specials=[])
num_word = len(vocab.itos)
weight_matrix = np.zeros((num_word, condition.embedding_dim))

print("loading vectors.....")
glove = api.load('glove-wiki-gigaword-{}'.format(condition.embedding_dim))

content_id_list = []
num_total_word = 0
num_not_in_glove = 0
for tree in tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            if vocab[node.content] not in content_id_list:
                if node.content in glove.vocab:
                    weight_matrix[vocab[node.content]] = glove[node.content]
                else:
                    num_not_in_glove += 1
                    weight_matrix[vocab[node.content]] =\
                        np.random.normal(
                        loc=0.0,
                        scale=1 /
                        np.sqrt(condition.embedding_dim),
                        size=condition.embedding_dim)
                content_id_list.append(vocab[node.content])
                num_total_word += 1

print("not in GloVe: {}/{} = {}".format(num_not_in_glove,
                                        num_total_word, num_not_in_glove / num_total_word))

with open(PATH_TO_PRETRAINED_WEIGHT_MATRIX, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(weight_matrix)

with open(path_to_word_counter, mode='wb') as f:
    pickle.dump(word_counter, f)
