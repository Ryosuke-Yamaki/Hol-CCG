from torchtext.vocab import Vocab
import pickle
from collections import Counter
import numpy as np
import csv
import gensim.downloader as api
from models import Node, Tree
from utils import Condition_Setter


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


condition = Condition_Setter(set_embedding_type=False)
condition.embedding_type = 'GloVe'

print('loading tree list...')
train_tree_list = set_tree_list(condition.path_to_train_data)
dev_tree_list = set_tree_list(condition.path_to_dev_data)
test_tree_list = set_tree_list(condition.path_to_test_data)

word_counter = Counter()
train_word_counter = Counter()
category_counter = Counter()
for tree in train_tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            word_counter[node.content] += 1
            train_word_counter[node.content] += 1
        category_counter[node.category] += 1

for tree in dev_tree_list + test_tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            word_counter[node.content] += 1

with open(condition.path_to_word_counter, mode='wb') as f:
    pickle.dump(word_counter, f)

with open(condition.path_to_train_word_counter, mode='wb') as f:
    pickle.dump(train_word_counter, f)

with open(condition.path_to_category_counter, mode='wb') as f:
    pickle.dump(category_counter, f)

vocab = Vocab(word_counter, specials=[])
num_word = len(vocab.itos)

for embeddin_dim in [50, 100, 300]:
    condition.embedding_dim = embeddin_dim
    condition.set_path(condition.PATH_TO_DIR)

    weight_matrix = np.zeros((num_word, condition.embedding_dim))

    print("{}d loading vectors.....".format(condition.embedding_dim))
    num_total_word = 0
    num_not_in_glove = 0
    idx = 0

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

    with open(condition.path_to_initial_weight_matrix, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(weight_matrix)
