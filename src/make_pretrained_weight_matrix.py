import numpy as np
import csv
import gensim.downloader as api
from models import Tree_List

PATH_TO_DIR = "/home/yryosuke0519/"

PATH_TO_DATA = PATH_TO_DIR + "Hol-CCG/data/train.txt"
PATH_TO_PRETRAINED_WEIGHT_MATRIX = PATH_TO_DIR + "Hol-CCG/data/pretrained_weight_matrix.csv"

EMBEDDING_DIM = 300

tree_list = Tree_List(PATH_TO_DATA, True)
print("loading vectors.....")
glove_vectors = api.load('glove-wiki-gigaword-300')

weight_matrix = []
for word in tree_list.vocab:
    if word in glove_vectors.vocab:
        weight_matrix.append(glove_vectors[word])
    else:
        weight_matrix.append(
            np.random.normal(
                loc=0.0,
                scale=1 /
                np.sqrt(EMBEDDING_DIM),
                size=EMBEDDING_DIM))

with open(PATH_TO_PRETRAINED_WEIGHT_MATRIX, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(weight_matrix)
