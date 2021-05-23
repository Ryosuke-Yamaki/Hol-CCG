import time
from models import Tree_List, Tree_Net
from utils import set_random_seed, make_n_hot_label
import torch.nn as nn
import torch.optim as optim

set_random_seed(0)
path_to_train_data = '/home/yryosuke0519/CCGbank/converted/train.txt'
path_to_dev_data = '/home/yryosuke0519/CCGbank/converted/dev.txt'
path_to_test_data = '/home/yryosuke0519/CCGbank/converted/test.txt'
print('processing data...')
train_tree_list = Tree_List(path_to_train_data)
dev_tree_list = Tree_List(
    path_to_dev_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab)
test_tree_list = Tree_List(
    path_to_test_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab)
possible_category_dict_for_train_dev = {}
possible_category_dict_for_test = {}
train_tree_list.set_possible_category_id(possible_category_dict_for_train_dev)
dev_tree_list.set_possible_category_id(possible_category_dict_for_train_dev)
test_tree_list.set_possible_category_id(possible_category_dict_for_test)
train_tree_list.set_info_for_training(possible_category_dict_for_train_dev)
dev_tree_list.set_info_for_training(possible_category_dict_for_train_dev)
test_tree_list.set_info_for_training(possible_category_dict_for_test)
print('fnish!')

tree_net = Tree_Net(train_tree_list, 100)
criteria = nn.BCELoss(reduction='sum')
optimizer = optim.Adam(tree_net.parameters())


for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if node.category not in train_tree_list.category_to_id:
            print(node.category)

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
