from utils import load, load_weight_matrix, single_circular_correlation
from sklearn.decomposition import PCA
from collections import Counter
import os
import torch
import matplotlib.pyplot as plt
from utils import set_random_seed, Condition_Setter
from models import Tree_List, Tree_Net
from sklearn.manifold import TSNE
import numpy as np

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

set_random_seed(0)
print('loading tree list...')
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim))
weight_matrix = torch.tensor(weight_matrix)

NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)
tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY, condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()
weight_matrix = tree_net.embedding.weight
vocab = test_tree_list.content_vocab

the = weight_matrix[vocab["the"]]
device = weight_matrix[vocab["device"]]
was = weight_matrix[vocab["was"]]
replaced = weight_matrix[vocab["replaced"]]
the_device = single_circular_correlation(the, device)
was_replaced = single_circular_correlation(was, replaced)
the_device_was_replaced = single_circular_correlation(the_device, was_replaced)
x = range(0, condition.embedding_dim)
plt.plot(x, the_device_was_replaced.detach().numpy())
plt.show()
