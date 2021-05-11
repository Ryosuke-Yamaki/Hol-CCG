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
criteria = nn.BCELoss()
optimizer = optim.Adam(tree_net.parameters())

# save weight matrix as initial state
with open(condition.path_to_initial_weight_matrix, 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(tree_net.embedding.weight)

batch_list = train_tree_list.make_batch(BATCH_SIZE)

for epoch in range(EPOCHS):
    batch_list = train_tree_list.make_batch(BATCH_SIZE)
    average_loss = 0.0
    for batch in batch_list:
        optimizer.zero_grad()
        output = tree_net(batch)
        label_list = batch[3]
        label_mask = batch[4]
        loss = criteria(output * label_mask, label_list)
        loss.backward()
        average_loss += loss
        optimizer.step()
    if epoch % 10 == 0:
        print(epoch, average_loss / BATCH_SIZE)
