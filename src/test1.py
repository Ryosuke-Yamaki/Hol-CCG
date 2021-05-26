import time
from models import Tree_List, Tree_Net
from utils import set_random_seed, make_n_hot_label, Condition_Setter
import torch.nn as nn
import torch.optim as optim
import torch

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
# possible_category_dict_for_train_dev = {}
# possible_category_dict_for_test = {}
# train_tree_list.set_possible_category_id(possible_category_dict_for_train_dev)
# dev_tree_list.set_possible_category_id(possible_category_dict_for_train_dev)
# test_tree_list.set_possible_category_id(possible_category_dict_for_test)
# train_tree_list.set_info_for_training(possible_category_dict_for_train_dev)
# dev_tree_list.set_info_for_training(possible_category_dict_for_train_dev)
# test_tree_list.set_info_for_training(possible_category_dict_for_test)

test_tree_list.clean_tree_list()
dev_tree_list.clean_tree_list()

num_unk_tree = 0
num_tree = 0
cat = {}
bit = 0
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if 0 in node.content_id or node.category_id == 0:
            num_unk_tree += 1
            break
    num_tree += 1
print(num_unk_tree / num_tree)

num_unk = 0
num_node = 0
cat = {}
for tree in dev_tree_list.tree_list:
    for node in tree.node_list:
        if 0 in node.content_id or node.category_id == 0:
            num_unk += 1
        num_node += 1
print(num_unk / num_node)

tree_net = Tree_Net(train_tree_list, 100)
criteria = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(tree_net.parameters())

num_category = len(test_tree_list.category_to_id)
for i in range(100):
    start = time.time()
    batch_list = test_tree_list.make_batch(25)
    for batch in batch_list:
        optimizer.zero_grad()
        output = tree_net(batch)
        n_hot_label, mask = make_n_hot_label(batch[4], num_category)
        loss = criteria(output * mask, n_hot_label)
        loss.backward()
        optimizer.step()
        print(loss)
    print(time.time() - start)
