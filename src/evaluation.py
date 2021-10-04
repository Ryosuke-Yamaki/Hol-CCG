from utils import evaluate_tree_list, evaluate_batch_list, evaluate_beta, load, set_random_seed, Condition_Setter
import torch


condition = Condition_Setter(set_embedding_type=False)

device = torch.device('cuda:0')

set_random_seed(0)

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)
test_tree_list.embedder = 'bert'
dev_tree_list = load(condition.path_to_test_tree_list)
dev_tree_list.embedder = 'bert'

model = "roberta-large_with_LSTM_phrase.pth"
tree_net = torch.load(condition.path_to_model + model,
                      map_location=device)
tree_net.eval()

evaluate_beta(dev_tree_list, tree_net)
# evaluate_tree_list(test_tree_list, tree_net)
