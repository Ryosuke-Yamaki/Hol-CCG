from utils import dump, load
from utils import Condition_Setter
from torchtext.vocab import Vocab
from models import Tree_List

condition = Condition_Setter(set_embedding_type=False)

category_counter = load(condition.path_to_category_counter)
category_vocab = Vocab(category_counter, specials=['<unk>'])

print('loading tree list...')
train_tree_list = Tree_List(condition.path_to_train_data, category_vocab)
dev_tree_list = Tree_List(
    condition.path_to_dev_data,
    category_vocab)
test_tree_list = Tree_List(
    condition.path_to_test_data,
    category_vocab)

dump(train_tree_list, condition.path_to_train_tree_list)
dump(dev_tree_list, condition.path_to_dev_tree_list)
dump(test_tree_list, condition.path_to_test_tree_list)
