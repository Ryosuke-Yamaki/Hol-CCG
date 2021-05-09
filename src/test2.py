import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt
from models import Tree_List, Tree_Net, Condition_Setter, History
from utils import load_weight_matrix, set_random_seed
from parsing import CCG_Category_List, Linear_Classifier, Parser
import time

PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"
condition = Condition_Setter(PATH_TO_DIR)

# initialize tree_list from toy_data
train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
# match the vocab and category between train and test data
test_tree_list.replace_vocab_category(train_tree_list)

EPOCHS = 1000
BATCH_SIZE = 5
THRESHOLD = 0.3
PATIENCE = 30
NUM_VOCAB = len(train_tree_list.content_to_id)

set_random_seed(0)

if condition.RANDOM:
    initial_weight_matrix = None
else:
    initial_weight_matrix = load_weight_matrix(
        condition.path_to_pretrained_weight_matrix,
        condition.REGULARIZED)

# convert from ndarray to torch.tensor
tree_net = Tree_Net(train_tree_list, condition.embedding_dim, initial_weight_matrix)
criteria = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(tree_net.parameters())

# save weight matrix as initial state
with open(condition.path_to_initial_weight_matrix, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(tree_net.embedding.weight)

start = time.time()
batch_list = train_tree_list.make_batch(BATCH_SIZE)
print(time.time() - start)

for batch in batch_list:
    vector = tree_net.forward_test(batch)
    print(vector)
