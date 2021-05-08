import numpy as np
import csv
import gensim.downloader as api
from models import Tree_List

PATH_TO_DIR = "/home/yryosuke0519/"

embedding_dim = input("embedding_dim(default=100d): ")
if embedding_dim != "":
    embedding_dim = int(embedding_dim)
else:
    embedding_dim = 100
PATH_TO_DATA = PATH_TO_DIR + "Hol-CCG/data/train.txt"
PATH_TO_PRETRAINED_WEIGHT_MATRIX = PATH_TO_DIR + "Hol-CCG/data/glove_{}d.csv".format(embedding_dim)


tree_list = Tree_List(PATH_TO_DATA, True)
print("loading vectors.....")
glove_vectors = api.load('glove-wiki-gigaword-{}'.format(embedding_dim))

weight_matrix = []
for word in tree_list.content_to_id:
    if word in glove_vectors.vocab:
        vector = glove_vectors[word]
    else:
        print('{} not in GloVe!'.format(word))
        vector = weight_matrix.append(
            np.random.normal(
                loc=0.0,
                scale=1 /
                np.sqrt(embedding_dim),
                size=embedding_dim))
    weight_matrix.append(vector / np.linalg.norm(vector))

with open(PATH_TO_PRETRAINED_WEIGHT_MATRIX, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(weight_matrix)
