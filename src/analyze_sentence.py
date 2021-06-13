import gensim.downloader as api
from collections import Counter
from utils import load, Condition_Setter
import os

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

PATH_TO_PRETRAINED_WEIGHT_MATRIX = PATH_TO_DIR + \
    "Hol-CCG/data/glove_{}d.csv".format(condition.embedding_dim)
path_to_word_counter = PATH_TO_DIR + "Hol-CCG/data/word_counter.pickle"

print('loading tree list...')
train_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

content_id = []
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            content_id.append(node.content_id[0])

content_id = list(set(content_id))

glove = api.load('glove-wiki-gigaword-{}'.format(condition.embedding_dim))
vocab = train_tree_list.content_vocab

for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf and node.content_id[0] not in content_id:
            if node.content in glove.vocab:
                for word in glove.most_similar(node.content):
                    if word[0] in vocab.stoi and vocab.stoi[word[0]
                                                            ] in content_id and word[1] > 0.8:
                        print(node.content, word[0], word[1])
