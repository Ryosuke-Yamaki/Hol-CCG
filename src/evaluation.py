import time
import torch
from models import Tree_List, Tree_Net
from utils import set_random_seed, Condition_Setter

PATH_TO_DIR = "/home/yryosuke0519/"
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

set_random_seed(0)
print('loading tree list...')
train_tree_list = Tree_List(condition.path_to_train_data, device=device)
train_tree_list.set_info_for_training()
test_tree_list = Tree_List(
    condition.path_to_test_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab,
    device=device)
test_tree_list.clean_tree_list()
test_tree_list.set_info_for_training()

tree_net = Tree_Net(train_tree_list, condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=torch.device('cpu'))
tree_net.eval()

start = time.time()
tree_net.evaluate(train_tree_list)
print(time.time() - start)
start = time.time()
tree_net.evaluate(test_tree_list)
print(time.time())
