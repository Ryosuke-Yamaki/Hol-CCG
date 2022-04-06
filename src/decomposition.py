import numpy as np
from turtle import right
from torch.fft import ifft
from torch.fft import fft
from collections import Counter
import numpy as npstandardize
from utils import circular_correlation, load, Condition_Setter, circular_convolution, inverse_circular_correlation
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity as cos
from torch.nn.functional import normalize
import random

condition = Condition_Setter(set_embedding_type=False)

if torch.cuda.is_available():
    device = torch.device('cuda:5')
else:
    device = torch.device('cpu')

model = 'roberta-large_phrase_span_2022-01-26_06:48:23.pth'
dev_tree_list = load(condition.path_to_binary_dev_tree_list)
tree = dev_tree_list.tree_list[42]
tree_net = torch.load(condition.path_to_model + model,
                      map_location=device)
tree_net.device = device
tree_net.eval()

dev_tree_list.tokenizer = tree_net.tokenizer
dev_tree_list.set_info_for_training(tokenizer=tree_net.tokenizer)

with torch.no_grad():
    dev_tree_list.set_vector(tree_net)

# t0 = [1885, 16, 'l']
# t1 = [1, 21, 'l']
# t2 = [42, 20, 'r']
# t3 = [120, 10, 'r']
# t4 = [71, 30, 'l']
# t5 = [1090, 18, 'r']
# t6 = [1148, 19, 'r']
node_list = [[1683, 34, 'r'],
             [1885, 16, 'l'],
             [1876, 19, 'l'],
             [1438, 10, 'l'],
             [78, 9, 'r'],
             [42, 24, 'l'],
             [952, 8, 'r'],
             [758, 10, 'r'],
             [1233, 15, 'l'],
             [1824, 10, 'r'],
             [512, 30, 'l'],
             [554, 32, 'l'],
             [64, 23, 'r'],
             [1766, 11, 'l'],
             [1744, 24, 'l'],
             [290, 16, 'l'],
             [744, 24, 'l'],
             [331, 11, 'r'],
             [527, 18, 'l'],
             [370, 30, 'l'],
             [840, 7, 'l'],
             [1051, 30, 'l'],
             [748, 20, 'l'],
             [1896, 8, 'l'],
             [1782, 8, 'l'],
             [246, 18, 'l'],
             [103, 15, 'l']
             ]

for info in node_list:
    tree = dev_tree_list.tree_list[info[0]]
    print('*' * 50)
    print(" ".join(tree.sentence))
    target_child = tree.node_list[info[1]]
    lr = info[2]
    if lr == 'l':
        child_is_left = False
    else:
        child_is_left = True
    parent = target_child.parent_node
    if lr == 'l':
        another_child = tree.node_list[parent.right_child_node_id]
    else:
        another_child = tree.node_list[parent.left_child_node_id]
    reconstruct_vector = inverse_circular_correlation(
        parent.vector,
        another_child.vector,
        tree_net.vector_norm,
        child_is_left=child_is_left)

    sim_list = []
    tree_node_id_list = []
    for tree in dev_tree_list.tree_list:
        for node in tree.node_list:
            sim = cos(reconstruct_vector, node.vector, dim=-1).item()
            sim_list.append(sim)
            tree_node_id_list.append([tree.self_id, node.self_id])
    sim_list = np.array(sim_list)
    idx_list = np.argsort(-1 * sim_list)[:6]
    for idx in idx_list:
        id = tree_node_id_list[idx]
        tree = dev_tree_list.tree_list[id[0]]
        node = tree.node_list[id[1]]
        print(" ".join(node.content))
        print(f'similarity:{sim_list[idx]}\n')
    a = 0
