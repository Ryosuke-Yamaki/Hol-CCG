from utils import load, load_weight_matrix
import os
import torch
from models import Tree_Net
from utils import set_random_seed, Condition_Setter
from torch.nn import Embedding

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

set_random_seed(0)

print('loading tree list...')
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

new_weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim))
new_weight_matrix = torch.tensor(new_weight_matrix)


NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)

new_embedding = Embedding(NUM_VOCAB, condition.embedding_dim, _weight=new_weight_matrix)

tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

tree_net.evaluate(test_tree_list)

tree_net.embedding = new_embedding

tree_net.evaluate(test_tree_list)
