from utils import dump, load
from utils import Condition_Setter
from models import Tree_List

condition = Condition_Setter(set_embedding_type=False)

word_category_vocab = load(condition.path_to_word_category_vocab)
phrase_category_vocab = load(
    condition.path_to_phrase_category_vocab,)

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

dump(train_tree_list, condition.path_to_train_tree_list)
dump(dev_tree_list, condition.path_to_dev_tree_list)
dump(test_tree_list, condition.path_to_test_tree_list)
