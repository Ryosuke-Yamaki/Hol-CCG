import os
import torch
from models import Tree_Net
from utils import set_random_seed, Condition_Setter
from utils import load

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

set_random_seed(0)

print('loading tree list...')
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)

tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

tree_net.evaluate(test_tree_list)
