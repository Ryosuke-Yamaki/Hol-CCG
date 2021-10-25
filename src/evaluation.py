from utils import evaluate_tree_list, evaluate_batch_list, evaluate_beta, load, set_random_seed, Condition_Setter
import torch

torch.cuda.empty_cache()

condition = Condition_Setter(set_embedding_type=False)

device = torch.device('cuda:0')

set_random_seed(0)

print('loading tree list...')
# test_tree_list = load(condition.path_to_test_tree_list)
# test_tree_list.embedder = 'bert'
dev_tree_list = load(condition.path_to_dev_tree_list)
dev_tree_list.embedder = 'bert'

model = "roberta-large_phrase(1).pth"
tree_net = torch.load(condition.path_to_model + model,
                      map_location=device)
tree_net.device = device
tree_net.eval()

if dev_tree_list.embedder == 'bert':
    for tree in dev_tree_list.tree_list:
        tree.set_word_split(tree_net.tokenizer)
dev_tree_list.set_vector(tree_net)

beta_list = [0.05, 0.025, 0.01, 0.005, 0.0025, 0.001, 0.0005, 0.00025, 0.0001]
alpha_list = [4, 8, 16, 32, 64, 128]

max_word_acc = 0.0

for beta in beta_list:
    for alpha in alpha_list:
        word_acc, cat_per_word = evaluate_beta(dev_tree_list, tree_net, beta=beta, alpha=alpha)
        print('bata={}, alpha={}'.format(beta, alpha))
    if word_acc > max_word_acc:
        max_word_acc = word_acc
        max_cat_per_word = cat_per_word
        max_beta = beta
        max_alpha = alpha

print('best_param:\nbeta={},alpha={},word={},cat_per_word={}'.format(
    max_beta, max_alpha, max_word_acc, max_cat_per_word))


# evaluate_tree_list(dev_tree_list, tree_net)
