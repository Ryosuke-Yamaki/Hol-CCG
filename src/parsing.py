import torch
from models import Tree_List, Tree_Net
from utils import load_weight_matrix
import sys
import numpy as np


class CCG_Category:
    def __init__(self, category):
        # for the complex categories including brackets
        if '(' in category:
            self.category = category[1:-1]
        # for the primitive categories like NP or S
        else:
            self.category = category
        # self.category = category
        self.set_composition_info(self.category)

    def set_composition_info(self, category):
        level = 0
        for idx in range(len(category)):
            char = category[idx]
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            if level == 0:
                if char == '\\':
                    self.direction_of_slash = 'L'
                    self.parent_category = category[:idx]
                    self.sibling_category = category[idx + 1:]
                    return
                elif char == '/':
                    self.direction_of_slash = 'R'
                    self.parent_category = category[:idx]
                    self.sibling_category = category[idx + 1:]
                    return
        self.direction_of_slash = None
        self.parent_category = None
        self.sibling_category = None


def compose_categories(left_category, right_category):
    if left_category.direction_of_slash == 'R' and left_category.sibling_category == right_category.category:
        return left_category.parent_category
    elif left_category.direction_of_slash == 'R' and left_category.sibling_category == '(' + right_category.category + ')':
        return left_category.parent_category
    elif right_category.direction_of_slash == 'L' and right_category.sibling_category == left_category.category:
        return right_category.parent_category
    elif right_category.direction_of_slash == 'L' and right_category.sibling_category == '(' + left_category.category + ')':
        return right_category.parent_category


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
# use same vocablary and category as train_tree_list
test_tree_list.vocab = train_tree_list.vocab
test_tree_list.category = train_tree_list.category
test_tree_list.add_content_category_id(test_tree_list.vocab, test_tree_list.category)

# make inverse vocablary and category dictionary: id -> word or category
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

tree = test_tree_list.tree_list[2]
output = tree_net(tree)

n = 5
for node_id in range(len(tree.node_list)):
    node = tree.node_list[node_id]
    # if node.is_leaf:
    node.prob_dist = output[node_id].detach().numpy().copy()
    node.top_n_category_id = np.argsort(node.prob_dist)[::-1][:n]

for node in tree.node_list:
    print(node.content)
    for category_id in node.top_n_category_id:
        print('{}:{}'.format(inv_category[category_id], node.prob_dist[category_id]))
    print('')

# for node in tree.node_list:
#     if node.is_leaf:
#         print(node.content)
#         for category_id in node.top_n_category_id:
#             print('{}:{}'.format(inv_category[category_id], node.prob_dist[category_id]))


# for tree in train_tree_list.tree_list:
#     for node in tree.node_list:
#         if not node.is_leaf:
#             node.category = None
#             print(node.category)

# tree_idx = 0
# for tree in train_tree_list.tree_list:
#     print('*' * 50)
#     print('tree_idx = {}'.format(tree_idx))
#     node_pair_list = tree.make_node_pair_list()
#     while True:
#         i = 0
#         for node_pair in node_pair_list:
#             left_node = node_pair[0]
#             right_node = node_pair[1]
#             parent_node = tree.node_list[left_node.parent_id]
#             if left_node.ready and right_node.ready:
#                 left_category = CCG_Category(left_node.category)
#                 right_category = CCG_Category(right_node.category)
#                 composed_category = compose_categories(left_category, right_category)
#                 parent_node.category = composed_category
#                 parent_node.content = left_node.content + ' ' + right_node.content
#                 parent_node.ready = True
#                 print('{} + {} ---> {}'.format(left_node.content,
#                                                right_node.content, parent_node.content))
#                 print('{} {} ---> {}'.format(left_category.category,
#                                              right_category.category, composed_category))
#                 print('')
#                 node_pair_list.remove(node_pair)
#         if node_pair_list == []:
#             break
#     tree_idx += 1
