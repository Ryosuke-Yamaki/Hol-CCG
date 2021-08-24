from utils import evaluate, load, set_random_seed, Condition_Setter
import torch


condition = Condition_Setter(set_embedding_type=False)

device = torch.device('cuda')

set_random_seed(0)

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)

tree_net = torch.load("lstm_with_two_classifiers.pth",
                      map_location=device)
tree_net.eval()

evaluate(test_tree_list, tree_net)
