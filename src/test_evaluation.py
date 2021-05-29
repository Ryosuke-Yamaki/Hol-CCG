import torch
import torch.nn as nn
from models import Tree_List, Tree_Net
from utils import single_circular_correlation, load_weight_matrix, set_random_seed, Condition_Setter, History

PATH_TO_DIR = "/home/yryosuke0519/"
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

set_random_seed(0)
path_to_train_data = condition.path_to_train_data
path_to_dev_data = condition.path_to_dev_data
path_to_test_data = condition.path_to_test_data
print('processing data...')
train_tree_list = Tree_List(path_to_train_data, device=device)
train_tree_list.set_info_for_training()

test_tree_list = Tree_List(
    path_to_test_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab,
    device=device)
test_tree_list.clean_tree_list()
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
tree_net = torch.load(condition.path_to_model,
                      map_location=torch.device('cpu'))
tree_net.eval()

criteria = nn.CrossEntropyLoss()
test_history = History(tree_net, test_tree_list, criteria, THRESHOLD)

batch_list = test_tree_list.make_batch(BATCH_SIZE)
test_history.validation(batch_list)
test_history.print_current_stat('test')

# caluclate leaf accuracy and phrase accuracy
embedding = tree_net.embedding
fc = tree_net.linear
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            node.vector = embedding(torch.tensor(node.content_id))
            node.vector = node.vector / torch.norm(node.vector)

for tree in test_tree_list.tree_list:
    for composition_info in tree.composition_info:
        num_child = composition_info[0]
        parent_node = tree.node_list[composition_info[1]]
        if num_child == 1:
            child_node = tree.node_list[composition_info[2]]
            parent_node.vector = child_node.vector
        else:
            left_node = tree.node_list[composition_info[2]]
            right_node = tree.node_list[composition_info[3]]
            parent_node.vector = single_circular_correlation(
                left_node.vector, right_node.vector)

num_correct_leaf = 0
num_leaf = 0
num_correct_phrase = 0
num_phrase = 0
num_node = 0
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        output = fc(node.vector)
        predict = torch.topk(output, k=5)[1]
        if node.is_leaf:
            if node.category_id in predict:
                num_correct_leaf += 1
            num_leaf += 1
        else:
            if node.category_id in predict:
                num_correct_phrase += 1
            num_phrase += 1
        num_node += 1
print('leaf: {}'.format(num_correct_leaf / num_leaf))
print('phrase: {}'.format(num_correct_phrase / num_phrase))
print('total: {}'.format((num_correct_leaf + num_correct_phrase) / num_node))
