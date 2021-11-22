from utils import evaluate_tree_list, evaluate_batch_list, evaluate_beta, load, set_random_seed, Condition_Setter
import torch
import sys

args = sys.argv

# beta = [float(args[1])]
# alpha = [int(args[2])]

condition = Condition_Setter(set_embedding_type=False)

device = torch.device('cuda')

set_random_seed(0)

print('loading tree list...')
# test_tree_list = load(condition.path_to_test_tree_list)
# test_tree_list.embedder = 'transformer'
dev_tree_list_base = load(condition.path_to_dev_tree_list)
dev_tree_list_base.embedder = 'transformer'
dev_tree_list_hol = load(condition.path_to_dev_tree_list)
dev_tree_list_hol.embedder = 'transformer'

base = "roberta-large(a).pth"
hol = "roberta-large_phrase(c).pth"

tree_net_base = torch.load(condition.path_to_model + base,
                           map_location=device)
tree_net_base.device = device
tree_net_base.eval()

tree_net_hol = torch.load(condition.path_to_model + hol,
                          map_location=device)
tree_net_hol.device = device
tree_net_hol.eval()

with torch.no_grad():
    if dev_tree_list_base.embedder == 'transformer':
        for tree in dev_tree_list_base.tree_list:
            tree.set_word_split(tree_net_base.tokenizer)
    dev_tree_list_base.set_vector(tree_net_base)

    if dev_tree_list_hol.embedder == 'transformer':
        for tree in dev_tree_list_hol.tree_list:
            tree.set_word_split(tree_net_hol.tokenizer)
    dev_tree_list_hol.set_vector(tree_net_hol)

beta_list = [0.00075, 0.0005, 0.00025]
alpha_list = [5, 10, 15]

# for beta in beta_list:
#     for alpha in alpha_list:
for data in [[0.075, None], [0.05, None], [0.01, None]]:
    beta = data[0]
    alpha = data[1]
    base_word_acc, base_cat_per_word = evaluate_beta(
        dev_tree_list_base, tree_net_base, beta=beta, alpha=alpha)
    hol_word_acc, hol_cat_per_word = evaluate_beta(
        dev_tree_list_hol, tree_net_hol, beta=beta, alpha=alpha)

    if hol_word_acc > 0.994 and hol_cat_per_word < 1.7 and hol_word_acc > base_word_acc and hol_cat_per_word < base_cat_per_word:
        print('best_param:\nbeta={},alpha={},word={},cat_per_word={}'.format(
            beta, alpha, hol_word_acc, hol_cat_per_word))
#     if word_acc > max_word_acc:
#         max_word_acc = word_acc
#         max_cat_per_word = cat_per_word
#         max_beta = beta
#         max_alpha = alpha

# print('best_param:\nbeta={},alpha={},word={},cat_per_word={}'.format(
#     max_beta, max_alpha, max_word_acc, max_cat_per_word))


# evaluate_tree_list(dev_tree_list, tree_net)
