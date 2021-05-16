from operator import itemgetter
import time
from parsing import CCG_Category_List, Linear_Classifier, Parser
from utils import load_weight_matrix, set_random_seed, single_circular_correlation
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt
from models import Tree_List, Tree_Net
from utils import Condition_Setter, History
import copy

PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"
condition = Condition_Setter(PATH_TO_DIR)


def make_batch(tree_list, tree_id):
    # make batch content id includes leaf node content id for each tree belongs to batch
    batch_leaf_content_id = []
    batch_label_list = []
    batch_composition_info = []
    for i in range(len(tree_list.tree_list)):
        if i in tree_id:
            batch_leaf_content_id.append(tree_list.leaf_node_content_id[i])
            batch_label_list.append(tree_list.label_list[i])
            batch_composition_info.append(tree_list.composition_info[i])

    content_id = batch_leaf_content_id
    label_list = batch_label_list
    composition_list = batch_composition_info

    max_num_leaf_node = max([len(i) for i in content_id])
    # set the mask for each tree in batch
    # content mask is used for two purpose,
    # 1 - embedding leaf node vector
    # 2 - decide the incex of insert position of composed vector
    true_mask = [torch.ones(len(i), dtype=torch.bool)
                 for i in content_id]
    false_mask = [
        torch.zeros(
            2 * max_num_leaf_node - 1 - len(i),
            dtype=torch.bool) for i in content_id]
    content_mask = torch.stack(
        [torch.cat((i, j)) for (i, j) in zip(true_mask, false_mask)])
    # make dummy content id to fill blank in batch
    dummy_content_id = [
        torch.zeros(
            max_num_leaf_node - len(i),
            dtype=torch.long) for i in content_id]
    batch_leaf_content_id = torch.stack([torch.cat((i, j)) for (
        i, j) in zip(content_id, dummy_content_id)])

    max_num_label = max([len(i) for i in label_list])
    num_category = label_list[0].shape[1]
    # set the mask for label of each node in tree
    true_mask = [
        torch.ones(
            (len(i), num_category),
            dtype=torch.bool) for i in label_list]
    false_mask = [
        torch.zeros(
            (max_num_label - len(i), num_category),
            dtype=torch.bool) for i in label_list]
    label_mask = torch.stack([torch.cat((i, j))
                              for (i, j) in zip(true_mask, false_mask)])
    # make dummy label to fill blank in batch
    dummy_label = [
        torch.zeros(
            2 * max_num_leaf_node - 1 - len(i),
            i.shape[1],
            dtype=torch.float) for i in label_list]
    batch_label_list = torch.stack(
        [torch.cat((i, j)) for (i, j) in zip(label_list, dummy_label)])

    # set mask for composition info in each batch
    max_num_composition = max([len(i) for i in composition_list])
    # make dummy compoisition info to fill blank in batch
    dummy_compositin_info = [
        torch.zeros(
            max_num_composition - len(i),
            i.shape[1],
            dtype=torch.long) for i in composition_list]
    batch_composition_info = torch.stack(
        [torch.cat((i, j)) for (i, j) in zip(composition_list, dummy_compositin_info)])

    # return zipped batch information, when training, extract each batch from zip itteration
    return [
        batch_leaf_content_id,
        content_mask,
        batch_composition_info,
        batch_label_list,
        label_mask]


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

# initialize tree_list from toy_data
train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED, device=device)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED, device=device)
# match the vocab and category between train and test data
test_tree_list.replace_vocab_category(train_tree_list)

EPOCHS = 100
BATCH_SIZE = 5
THRESHOLD = 0.3
PATIENCE = 100
NUM_VOCAB = len(train_tree_list.content_to_id)

set_random_seed(0)

if condition.RANDOM:
    initial_weight_matrix = None
else:
    initial_weight_matrix = load_weight_matrix(
        condition.path_to_pretrained_weight_matrix)

tree_net1 = Tree_Net(train_tree_list, condition.embedding_dim, initial_weight_matrix).to(device)
weight_matrix = tree_net1.embedding.weight.detach().numpy().copy()
tree_net2 = Tree_Net(train_tree_list, condition.embedding_dim, weight_matrix).to(device)
tree_net1.linear.weight.requires_grad = False
tree_net1.linear.bias.requires_grad = False
tree_net2.linear.weight.requires_grad = False
tree_net2.linear.bias.requires_grad = False
criteria = nn.BCELoss(reduction='sum')
optimizer1 = optim.Adam(tree_net1.parameters())
optimizer2 = optim.Adam(tree_net2.parameters())

for epoch in range(EPOCHS):
    for idx in range(0, 50, 5):
        tree_id_list = list(range(idx, idx + 5))
        batch = make_batch(train_tree_list, tree_id_list)
        optimizer1.zero_grad()
        output1 = tree_net1(batch)
        label_list = batch[3]
        label_mask = batch[4]
        loss1 = criteria(output1 * label_mask, label_list)

        optimizer2.zero_grad()
        loss2 = 0.0
        for tree_id in tree_id_list:
            tree = train_tree_list.tree_list[tree_id]
            num_leaf_node = len(tree.sentence.split(" "))
            node_vectors = [0.0] * (2 * num_leaf_node - 1)
            for node in tree.node_list:
                if node.is_leaf:
                    vector = tree_net2.embedding(torch.tensor(node.content_id))[0]
                    node_vectors[node.self_id] = vector / torch.norm(vector)
            for info in tree.composition_info:
                left_vector = node_vectors[info[0]]
                right_vector = node_vectors[info[1]]
                node_vectors[info[2]] = single_circular_correlation(left_vector, right_vector)
            node_vectors = torch.stack(node_vectors, dim=0)
            output2 = tree_net2.sigmoid(tree_net1.linear(node_vectors))
            loss2 += criteria(output2, torch.tensor(tree.label_list, dtype=torch.float))
        loss1.backward()
        optimizer1.step()
        loss2.backward()
        optimizer2.step()

        # print(loss1)
        # print(loss2)
        # print(tree_net1.embedding.weight[:100])
        # print(tree_net2.embedding.weight[:100])
        # print(
        #     torch.abs((tree_net1.embedding.weight[:100] - tree_net2.embedding.weight[:100])) > 0.5)
        # print(tree_net1.embedding.weight.grad[:100])
        # print(tree_net2.embedding.weight.grad[:100])
        # print(torch.abs(
        #     (tree_net1.embedding.weight.grad[:100] - tree_net2.embedding.weight.grad[:100])) > 0.5)
print(
    torch.abs((tree_net1.embedding.weight[:100] - tree_net2.embedding.weight[:100])) > 0.1)

a = 1
