import torch
from models import Tree_List, Tree_Net
from utils import load_weight_matrix, Analyzer
import sys

FROM_RANDOM = True
REGULARIZED = True
USE_ORIGINAL_LOSS = False
EMBEDDING_DIM = 100

args = sys.argv
if len(args) > 1:
    if args[1] == 'True':
        FROM_RANDOM = True
    else:
        FROM_RANDOM = False
    if args[2] == 'True':
        REGULARIZED = True
    else:
        REGULARIZED = False
    if args[3] == 'True':
        USE_ORIGINAL_LOSS = True
    else:
        USE_ORIGINAL_LOSS = False
    EMBEDDING_DIM = int(args[4])

PATH_TO_DIR = "/home/yryosuke0519/"

PATH_TO_TRAIN_DATA = PATH_TO_DIR + "Hol-CCG/data/train.txt"
PATH_TO_TEST_DATA = PATH_TO_DIR + "Hol-CCG/data/test.txt"

path_to_initial_weight_matrix = PATH_TO_DIR + "Hol-CCG/result/data/"
path_to_model = PATH_TO_DIR + "Hol-CCG/result/model/"
path_list = [
    path_to_initial_weight_matrix,
    path_to_model]

for i in range(len(path_list)):
    if FROM_RANDOM:
        path_list[i] += "random"
    else:
        path_list[i] += "GloVe"
    if REGULARIZED:
        path_list[i] += "_reg"
    else:
        path_list[i] += "_not_reg"
    if USE_ORIGINAL_LOSS:
        path_list[i] += "_original_loss"
    path_list[i] += "_" + str(EMBEDDING_DIM) + "d"
path_to_initial_weight_matrix = path_list[0] + "_initial_weight_matrix.csv"
path_to_model = path_list[1] + "_model.pth"

train_tree_list = Tree_List(PATH_TO_TRAIN_DATA, REGULARIZED)
test_tree_list = Tree_List(PATH_TO_TEST_DATA, REGULARIZED)
test_tree_list.vocab = train_tree_list.vocab
test_tree_list.category = train_tree_list.category
test_tree_list.add_content_category_id(test_tree_list.vocab, test_tree_list.category)

vocab = test_tree_list.vocab
category = test_tree_list.category
inv_vocab = {}
inv_category = {}
for k, v in vocab.items():
    inv_vocab[v] = k
for k, v in category.items():
    inv_category[v] = k

weight_matrix = torch.tensor(load_weight_matrix(path_to_initial_weight_matrix, REGULARIZED))
tree_net = Tree_Net(test_tree_list, weight_matrix)
tree_net.load_state_dict(torch.load(path_to_model))
tree_net.eval()

analyzer = Analyzer(train_tree_list.category, tree_net)

for tree in test_tree_list.tree_list:
    analyzer.analyze(tree)
