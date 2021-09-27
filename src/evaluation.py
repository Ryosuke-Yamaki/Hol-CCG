from utils import evaluate_tree_list, evaluate_batch_list,load, set_random_seed, Condition_Setter
import torch


condition = Condition_Setter(set_embedding_type=False)

device = torch.device('cuda:1')

set_random_seed(0)

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)
test_tree_list.embedder = 'bert'

tree_net = torch.load("roberta_0923_word_acc.pth",
                      map_location=device)
tree_net.eval()

evaluate_tree_list(test_tree_list, tree_net)
