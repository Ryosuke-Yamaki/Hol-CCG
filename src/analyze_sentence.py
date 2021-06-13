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

counter = Counter()
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            counter[node.content] += 1
num_known_words = len(counter)
vocab = train_tree_list.content_vocab

for unk_idx in range(num_known_words, len(vocab.itos)):
    print(vocab.freqs[unk_idx])
