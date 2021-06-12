from utils import dump, load
import os
from utils import Condition_Setter
from torchtext.vocab import Vocab
from models import Tree_List

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

path_to_word_counter = PATH_TO_DIR + "Hol-CCG/data/word_counter.pickle"
word_counter = load(path_to_word_counter)
content_vocab = Vocab(word_counter, specials=['<unk>'])

path_to_train_data = condition.path_to_train_data
path_to_dev_data = condition.path_to_dev_data
path_to_test_data = condition.path_to_test_data
print('loading tree list...')
train_tree_list = Tree_List(path_to_train_data, content_vocab)
dev_tree_list = Tree_List(
    path_to_dev_data,
    content_vocab,
    train_tree_list.category_vocab)
test_tree_list = Tree_List(
    path_to_test_data,
    content_vocab,
    train_tree_list.category_vocab)

dump(train_tree_list, PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
dump(dev_tree_list, PATH_TO_DIR + "Hol-CCG/data/dev_tree_list.pickle")
dump(test_tree_list, PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")
