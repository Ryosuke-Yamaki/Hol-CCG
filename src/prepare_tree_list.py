from collections import Counter
from utils import dump, load
from utils import Condition_Setter
from torchtext.vocab import Vocab
from models import Tree_List

condition = Condition_Setter(set_embedding_type=False)

word_counter = load(condition.path_to_word_counter)
category_counter = load(condition.path_to_category_counter)
content_vocab = Vocab(word_counter, specials=[])
category_vocab = Vocab(category_counter, specials=['<unk>'], min_freq=10, specials_first=False)

print('loading tree list...')
train_tree_list = Tree_List(condition.path_to_train_data, content_vocab, category_vocab)
dev_tree_list = Tree_List(
    condition.path_to_dev_data,
    content_vocab,
    category_vocab)
test_tree_list = Tree_List(
    condition.path_to_test_data,
    content_vocab,
    category_vocab)

num_unk_sentence = 0
num_unk_node = 0
num_sentence = 0
num_node = 0
unk_idx = category_vocab.unk_index
for tree in dev_tree_list.tree_list:
    num_sentence += 1
    bit = 0
    for node in tree.node_list:
        num_node += 1
        if node.category_id == unk_idx:
            num_unk_node += 1
            bit = 1
    if bit == 1:
        num_unk_sentence += 1
print('unk_sentence: {}({}/{})'.format(num_unk_sentence / num_sentence, num_unk_sentence, num_sentence))
print('unk_node: {}({}/{})'.format(num_unk_node / num_node, num_unk_node, num_node))

train_counter = Counter()
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            train_counter[node.content] += 1

unk_dev = 0
for tree in dev_tree_list.tree_list:
    bit = 0
    for node in tree.node_list:
        if node.is_leaf:
            if train_counter[node.content] == 0:
                bit = 1
    if bit == 1:
        unk_dev += 1

unk_test = 0
for tree in test_tree_list.tree_list:
    bit = 0
    for node in tree.node_list:
        if node.is_leaf:
            if train_counter[node.content] == 0:
                bit = 1
    if bit == 1:
        unk_test += 1


dump(train_tree_list, condition.path_to_train_tree_list)
dump(dev_tree_list, condition.path_to_dev_tree_list)
dump(test_tree_list, condition.path_to_test_tree_list)
