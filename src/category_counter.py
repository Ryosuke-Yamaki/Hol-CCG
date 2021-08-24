from collections import Counter
from models import Node, Tree
from utils import Condition_Setter, dump


def set_tree_list(PATH_TO_DATA):
    tree_list = []
    tree_id = 0
    node_list = []
    with open(PATH_TO_DATA, 'r') as f:
        node_info_list = [node_info.strip() for node_info in f.readlines()]
    node_info_list = [node_info.replace(
        '\n', '') for node_info in node_info_list]
    for node_info in node_info_list:
        if node_info != '':
            node = Node(node_info.split())
            node_list.append(node)
        elif node_list != []:
            tree_list.append(Tree(tree_id, node_list))
            node_list = []
            tree_id += 1
    return tree_list


condition = Condition_Setter(set_embedding_type=False)

print('loading tree list...')
train_tree_list = set_tree_list(condition.path_to_train_data)
dev_tree_list = set_tree_list(condition.path_to_dev_data)
test_tree_list = set_tree_list(condition.path_to_test_data)

word_category_counter = Counter()
phrase_category_counter = Counter()
for tree in train_tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            word_category_counter[node.category] += 1
        else:
            phrase_category_counter[node.category] += 1

dump(word_category_counter, condition.path_to_word_category_counter)
dump(phrase_category_counter, condition.path_to_phrase_category_counter)
