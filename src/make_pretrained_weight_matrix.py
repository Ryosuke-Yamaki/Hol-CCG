import csv
from gensim.models import KeyedVectors
from models import Tree_List

PATH_TO_DIR = "/home/yryosuke0519/"

PATH_TO_DATA = PATH_TO_DIR + "Hol-CCG/data/toy_data.txt"
PATH_TO_GLOVE_WEIGHT_MATRIX = PATH_TO_DIR + "Hol-CCG/data/glove.840B.300d.txt"
PATH_TO_PRETRAINED_WEIGHT_MATRIX = PATH_TO_DIR + "Hol-CCG/data/pretrained_weight_matrix.csv"

print("loading vectors.....")
glove_vectors = KeyedVectors.load_word2vec_format(PATH_TO_GLOVE_WEIGHT_MATRIX, binary=False)

tree_list = Tree_List(PATH_TO_DATA, False)

weight_matrix = []
for word in tree_list.vocab:
    weight_matrix.append(glove_vectors[word])

with open(PATH_TO_PRETRAINED_WEIGHT_MATRIX, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(weight_matrix)
