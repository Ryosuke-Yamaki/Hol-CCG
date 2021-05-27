import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models import Tree_List, Tree_Net
from utils import load_weight_matrix, set_random_seed, make_n_hot_label, Condition_Setter, History
from tqdm import tqdm
# from parsing import CCG_Category_List, Linear_Classifier, Parser

PATH_TO_DIR = "/home/yryosuke0519/"
condition = Condition_Setter(PATH_TO_DIR)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

set_random_seed(0)
path_to_train_data = condition.path_to_train_data
path_to_dev_data = condition.path_to_dev_data
path_to_test_data = condition.path_to_test_data
print('processing data...')
train_tree_list = Tree_List(path_to_train_data, device=device)
dev_tree_list = Tree_List(
    path_to_dev_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab,
    device=device)
test_tree_list = Tree_List(
    path_to_test_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab,
    device=device)
dev_tree_list.clean_tree_list()
test_tree_list.clean_tree_list()

train_tree_list.set_info_for_training()
dev_tree_list.set_info_for_training()
test_tree_list.set_info_for_training()

EPOCHS = 200
BATCH_SIZE = 25
THRESHOLD = 0.25
PATIENCE = 3
NUM_VOCAB = len(train_tree_list.content_vocab)
NUM_CATEGORY = len(train_tree_list.category_vocab)

if condition.RANDOM:
    initial_weight_matrix = None
else:
    initial_weight_matrix = load_weight_matrix(
        condition.path_to_pretrained_weight_matrix)

tree_net = Tree_Net(train_tree_list, condition.embedding_dim,
                    initial_weight_matrix).to(device)
tree_net = torch.load(condition.path_to_model, map_location=torch.device('cpu'))
tree_net.eval()

criteria = nn.CrossEntropyLoss(reduction='sum')

test_history = History(tree_net, test_tree_list, criteria, THRESHOLD)

test_batch_list = test_tree_list.make_batch(BATCH_SIZE)

test_history.validation(test_batch_list)
test_history.print_current_stat('test')
