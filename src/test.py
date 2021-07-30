import matplotlib.pyplot as plt
from collections import Counter
from utils import load
import os
from torchtext.vocab import Vocab
from utils import load, load_weight_matrix
import os
import torch
from models import Tree_Net
from utils import set_random_seed, Condition_Setter
from torch.nn import Embedding


def cut_off(output, beta):
    sorted_prob = torch.sort(output, descending=True)[0]
    sorted_idx = torch.argsort(output, descending=True)
    max_prob = torch.max(output)
    for idx in range(1, len(sorted_prob)):
        if sorted_prob[idx] < max_prob * beta:
            break
    return torch.sort(sorted_idx[:idx])[0]


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

print('loading tree list...')
# train_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
# dev_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/dev_tree_list.pickle")
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab) - 1

tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

embedding = tree_net.embedding
linear = tree_net.linear
softmax = torch.nn.Softmax(dim=-1)

total_leaf = 0
correct_leaf = 0
total_phrase = 0
correct_phrase = 0
leaf_length = 0
phrase_length = 0
content_vocab = test_tree_list.content_vocab
test_tree_list.set_vector(embedding)

for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        output = softmax(linear(node.vector[0]))
        top_k = cut_off(output, 0.005)
        if node.is_leaf:
            # print(node.content)
            # print(node.category_id)
            # print(top_k)
            # a = input()
            total_leaf += 1
            leaf_length += len(top_k)
            if node.category_id in top_k:
                correct_leaf += 1
        else:
            # for id in node.content_id:
            #     print(test_tree_list.content_vocab.itos[id])
            # print(node.category_id)
            # print(top_k)
            # a = input()
            total_phrase += 1
            phrase_length += len(top_k)
            if node.category_id in top_k:
                correct_phrase += 1

print(correct_leaf / total_leaf)
print(leaf_length / total_leaf)
print(correct_phrase / total_phrase)
print(phrase_length / total_phrase)


# leaf_counter = Counter()
# phrase_counter = Counter()

# for tree in train_tree_list.tree_list:
#     for node in tree.node_list:
#         if node.is_leaf:
#             leaf_counter[node.category] += 1
# # フレーズに固有のカテゴリ
# for tree in train_tree_list.tree_list:
#     for node in tree.node_list:
#         if leaf_counter[node.category] == 0:
#             phrase_counter[node.category] += 1

# cut_off_counter = Counter()

# for k, v in leaf_counter.items():
#     if v >= 8:
#         cut_off_counter[k] += 1

# for k, v in phrase_counter.items():
#     if v >= 8:
#         cut_off_counter[k] += 1

word_counter = Counter()
category_counter = Counter()
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            word_counter[node.content] += 1
        category_counter[node.category] += 1

word_vocab = Vocab(word_counter, specials=[])
category_vocab = Vocab(category_counter, min_freq=10, specials=['<unk>'])

dict = {}
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            if node.content not in dict:
                dict[node.content] = Counter()
            dict[node.content][category_vocab[node.category]] += 1


leaf_total = 0
leaf_not_exist = 0
phrase_total = 0
phrase_not_exist = 0
for tree in dev_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            leaf_total += 1
            if cut_off_counter[node.category] == 0:
                leaf_not_exist += 1
        else:
            phrase_total += 1
            if cut_off_counter[node.category] == 0:
                phrase_not_exist += 1

print('leaf', leaf_not_exist / leaf_total)
print('phrase', phrase_not_exist / phrase_total)
print('total', (leaf_not_exist + phrase_not_exist) / (leaf_total + phrase_total))

total_tree = 0
missing_tree = 0
for tree in train_tree_list.tree_list:
    total_tree += 1
    for node in tree.node_list:
        if cut_off_counter[node.category] == 0:
            missing_tree += 1
            break
print(missing_tree / total_tree)

a = 1
# print(len(leaf_counter))
# print(len(phrase_counter))
