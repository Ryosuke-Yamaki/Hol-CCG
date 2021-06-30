# from utils import load_weight_matrix, circular_correlation
from models import Tree_List, Tree_Net, Condition_Setter
import torch
# import torch.nn as nn
from parser import Linear_Classifier, Parser, CCG_Category_List
from utils import load_weight_matrix

PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"

condition = Condition_Setter(PATH_TO_DIR)

# initialize tree_list from toy_data
train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED)
print(train_tree_list.category_to_id)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
# match the vocab and category between train and test data
test_tree_list.replace_vocab_category(train_tree_list)

weight_matrix = torch.tensor(
    load_weight_matrix(
        condition.path_to_initial_weight_matrix,
        condition.REGULARIZED))
tree_net = Tree_Net(test_tree_list, weight_matrix)
tree_net.load_state_dict(torch.load(condition.path_to_model))
tree_net.eval()

# after training, parse test data and print statistics
ccg_category_list = CCG_Category_List(test_tree_list)
linear_classifier = Linear_Classifier(tree_net)
weight_matrix = tree_net.embedding.weight
parser = Parser(
    test_tree_list,
    ccg_category_list,
    weight_matrix,
    linear_classifier,
    THRESHOLD=0.3)
parser.validation()
parser.print_stat()
