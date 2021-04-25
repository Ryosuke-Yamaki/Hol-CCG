# import torch
# from models import Tree_List
# from torch.fft import fft, ifft
# from torch import conj, mul


# def circular_correlation(a, b, REGULARIZED):
#     a = conj(fft(a))
#     b = fft(b)
#     c = mul(a, b)
#     c = ifft(c).real
#     if REGULARIZED:
#         return c / torch.norm(c)
#     else:
#         return c


# PATH_TO_DIR = "/home/yryosuke0519/"

# PATH_TO_TRAIN_DATA = PATH_TO_DIR + "Hol-CCG/data/train.txt"
# train_tree_list = Tree_List(PATH_TO_TRAIN_DATA, True)

# tree = train_tree_list.tree_list[0]
# print(tree.sentense)
# tree.generate_info_for_training()

# embedding_dim = 3
# num_node = len(tree.node_list)

# leaf_node_info = tree.leaf_node_info
# label_list = tree.label_list
# composition_info = tree.composition_info

# weight_matrix = torch.ones((10, 3), requires_grad=True)
# node_vectors = [0] * num_node

# for info in leaf_node_info:
#     self_id = info[0]
#     content_id = info[1]
#     node_vectors[self_id] = weight_matrix[content_id]


# for info in composition_info:
#     left_node_id = info[0]
#     right_node_id = info[1]
#     parent_node_id = info[2]
#     node_vectors[parent_node_id] = circular_correlation(
#         node_vectors[left_node_id], node_vectors[right_node_id], True)
# node_vectors = torch.stack(node_vectors, dim=0)
# print(node_vectors)
# output = torch.sum(node_vectors)
# output.backward()
# print(weight_matrix.grad)

from utils import load_weight_matrix
from models import Tree_List, Tree_Net, Condition_Setter
import torch
import torch.nn as nn
PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"

condition = Condition_Setter(PATH_TO_DIR)

# initialize tree_list from toy_data
train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
# match the vocab and category between train and test data
test_tree_list.replace_vocab_category(train_tree_list.content_to_id, train_tree_list.category_to_id)

a = 1

# weight_matrix = torch.tensor(
#     load_weight_matrix(
#         condition.path_to_initial_weight_matrix,
#         condition.REGULARIZED))
# tree_net = Tree_Net(test_tree_list, weight_matrix)
# tree_net.load_state_dict(torch.load(condition.path_to_model))
# tree_net.eval()

# tree = test_tree_list.tree_list[0]
# output, node_vectors = tree_net(tree.leaf_node_info, tree.composition_info)
# criteria = nn.CrossEntropyLoss()
# loss = criteria(output, tree.label_list)
# loss.backward()
# for vector in node_vectors:
#     print(vector.grad)
#     print(vector.requires_grad)
# print(tree_net.linear.weight.grad)
# print(tree_net.embedding.weight.grad)

# import gensim.downloader

# print(list(gensim.downloader.info()['models'].keys()))
