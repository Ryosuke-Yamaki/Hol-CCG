from utils import load, load_weight_matrix, set_random_seed, evaluate, Condition_Setter
import torch
from torch.nn import Embedding


condition = Condition_Setter()

device = torch.device('cpu')

set_random_seed(0)

print('loading tree list...')
train_tree_list = load(condition.path_to_train_tree_list)
dev_tree_list = load(condition.path_to_dev_tree_list)
test_tree_list = load(condition.path_to_test_tree_list)

new_weight_matrix = torch.tensor(load_weight_matrix(condition.path_to_weight_with_regression))

NUM_VOCAB = len(test_tree_list.content_vocab)

new_embedding = Embedding(NUM_VOCAB, condition.embedding_dim, _weight=new_weight_matrix)

tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

content_id_in_train = []
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            content_id_in_train.append(node.content_id[0])
content_id_in_train = list(set(content_id_in_train))
unk_content_id = []
for tree in dev_tree_list.tree_list + test_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf and node.content_id[0] not in content_id_in_train:
            unk_content_id.append(node.content_id[0])
unk_content_id = list(set(unk_content_id))

print('evaluating...')
print('---without regression---')
print('***for only unk nodes***')
evaluate(test_tree_list, tree_net, unk_content_id)
print('***for entire tree***')
evaluate(test_tree_list, tree_net)

tree_net.embedding = new_embedding

print('evaluating...')
print('---with regression---')
print('***for only unk nodes***')
evaluate(test_tree_list, tree_net, unk_content_id)
print('***for entire tree***')
evaluate(test_tree_list, tree_net)
