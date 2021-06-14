from utils import load, load_weight_matrix
import os
import torch
from models import Tree_Net
from utils import set_random_seed, Condition_Setter
from torch.nn import Embedding

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

set_random_seed(0)

print('loading tree list...')
train_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
dev_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/dev_tree_list.pickle")
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

new_weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim))
new_weight_matrix = torch.tensor(new_weight_matrix)


NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)

new_embedding = Embedding(NUM_VOCAB, condition.embedding_dim, _weight=new_weight_matrix)

tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
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


def include_unk(content_id, unk_content_id):
    bit = 0
    for id in content_id:
        if id in unk_content_id:
            bit = 1
            return True
    if bit == 0:
        return False


def evaluate_on_unk(tree_list, tree_net, unk_content_id):
    embedding = tree_net.embedding
    linear = tree_net.linear
    tree_list.set_vector(embedding)
    for k in [1, 5]:
        total_num_node = 0
        num_unk_word = 0
        num_unk_phrase = 0
        num_correct_word = 0
        num_correct_phrase = 0
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                total_num_node += 1
                if include_unk(node.content_id, unk_content_id):
                    output = linear(node.vector)
                    predict = torch.topk(output, k=k)[1]
                    if node.is_leaf:
                        num_unk_word += 1
                        if node.category_id in predict:
                            num_correct_word += 1
                    else:
                        num_unk_phrase += 1
                        if node.category_id in predict:
                            num_correct_phrase += 1
        print('-' * 50)
        print('overall top-{}: {}'.format(k, (num_correct_word + \
              num_correct_phrase) / (num_unk_word + num_unk_phrase)))
        print('word top-{}: {}'.format(k, num_correct_word / num_unk_word))
        print('phrase top-{}: {}'.format(k, num_correct_phrase / num_unk_phrase))


print('evaluating...')
print('***before projection learning***')
print('***for only unk nodes***')
evaluate_on_unk(test_tree_list, tree_net, unk_content_id)
print('***for entire tree***')
tree_net.evaluate(test_tree_list)

tree_net.embedding = new_embedding

print('evaluating...')
print('***after projection learning***')
print('***for only unk nodes***')
evaluate_on_unk(test_tree_list, tree_net, unk_content_id)
print('***for entire tree***')
tree_net.evaluate(test_tree_list)
