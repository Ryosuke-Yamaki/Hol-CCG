from collections import Counter
from utils import load
import os

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")

train_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
dev_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/dev_tree_list.pickle")
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")


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

counter = Counter()
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        counter[node.category] += 1

cut_off_counter = Counter()

for k, v in counter.items():
    if v >= 10:
        cut_off_counter[k] += 1

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
