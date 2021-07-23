import fasttext.util
import fasttext
from torchtext.vocab import Vocab
import pickle
from collections import Counter
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
    "Hol-CCG/data/{}_{}d.csv".format(condition.embedding_type, condition.embedding_dim)
path_to_word_counter = PATH_TO_DIR + "Hol-CCG/data/word_counter.pickle"
path_to_category_counter = PATH_TO_DIR + "Hol-CCG/data/category_counter.pickle"
path_to_train_data = condition.path_to_train_data
path_to_dev_data = condition.path_to_dev_data
path_to_test_data = condition.path_to_test_data

print('loading tree list...')
train_tree_list = set_tree_list(path_to_train_data)
dev_tree_list = set_tree_list(path_to_dev_data)
test_tree_list = set_tree_list(path_to_test_data)

word_counter = Counter()
category_counter = Counter()
for tree in train_tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            word_counter[node.content] += 1
        category_counter[node.category] += 1

for tree in dev_tree_list + test_tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            word_counter[node.content] += 1

vocab = Vocab(word_counter, specials=[])
num_word = len(vocab.itos)
weight_matrix = np.zeros((num_word, condition.embedding_dim))

print("loading vectors.....")
num_total_word = 0
num_not_in_glove = 0
idx = 0

if condition.embedding_type == 'GloVe':
    model = api.load('glove-wiki-gigaword-{}'.format(condition.embedding_dim))
    for word in vocab.itos:
        if word in model.vocab:
            weight_matrix[idx] = model[word]
        else:
            num_not_in_glove += 1
            weight_matrix[idx] = np.random.rand(condition.embedding_dim)
        idx += 1
    print("not in GloVe: {}/{} = {}".format(num_not_in_glove,
                                            len(vocab.itos), num_not_in_glove / len(vocab.itos)))
elif condition.embedding_type == 'FastText':
    model = fasttext.load_model('cc.en.300.bin')
    fasttext.util.reduce_model(model, condition.embedding_dim)
    for word in vocab.itos:
        weight_matrix[idx] = model.get_word_vector(word)
        idx += 1


with open(PATH_TO_PRETRAINED_WEIGHT_MATRIX, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(weight_matrix)

with open(path_to_word_counter, mode='wb') as f:
    pickle.dump(word_counter, f)

with open(path_to_category_counter, mode='wb') as f:
    pickle.dump(category_counter, f)
