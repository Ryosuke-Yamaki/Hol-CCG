from utils import dump, load
from utils import Condition_Setter
from models import Tree_List
from torchtext.vocab import Vocab

condition = Condition_Setter(set_embedding_type=False)

word_category_vocab = load(condition.path_to_word_category_vocab)
phrase_category_vocab = load(
    condition.path_to_phrase_category_vocab)

phrase_category_vocab = Vocab(phrase_category_vocab.freqs, min_freq=0, specials=['<unk>'])

print('loading tree list...')
train_tree_list = Tree_List(
    condition.path_to_train_data,
    word_category_vocab,
    phrase_category_vocab)
dev_tree_list = Tree_List(
    condition.path_to_dev_data,
    word_category_vocab,
    phrase_category_vocab)
test_tree_list = Tree_List(
    condition.path_to_test_data,
    word_category_vocab,
    phrase_category_vocab)

head_info_temp = {}
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if not node.is_leaf and node.num_child == 2:
            left_child = tree.node_list[node.left_child_node_id]
            right_child = tree.node_list[node.right_child_node_id]
            rule = (left_child.category, right_child.category, node.category)
            if rule not in head_info_temp:
                head_info_temp[rule] = [0, 0]
            head_info_temp[rule][node.head] += 1
head_info = {}
for k, v in head_info_temp.items():
    # when left head is majority
    if v[0] >= v[1]:
        head_info[k] = 0
    # when right head is majority
    else:
        head_info[k] = 1

dump(train_tree_list, condition.path_to_train_tree_list)
dump(dev_tree_list, condition.path_to_dev_tree_list)
dump(test_tree_list, condition.path_to_test_tree_list)
dump(head_info, condition.path_to_head_info)
