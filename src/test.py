import torch
from models import Tree_Net
from utils import Condition_Setter
import os
from utils import load, load_weight_matrix
from parser import extract_rule, Parser
from torch.nn import Embedding

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
path_to_grammar = PATH_TO_DIR + 'CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar'
path_to_raw_sentence = PATH_TO_DIR + 'CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw'

condition = Condition_Setter(PATH_TO_DIR)
device = torch.device('cpu')
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

new_weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim))
new_weight_matrix = torch.tensor(new_weight_matrix)

NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)
tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY, condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()
new_embedding = Embedding(NUM_VOCAB, condition.embedding_dim, _weight=new_weight_matrix)
tree_net.embedding = new_embedding

f = open(path_to_raw_sentence, 'r')
test_sentence = f.readlines()
f.close()

binary_rule, unary_rule = extract_rule(path_to_grammar, test_tree_list.category_vocab)
parser = Parser(tree_net, test_tree_list.content_vocab, binary_rule, unary_rule)

f1, precision, recall = parser.validation(test_sentence, test_tree_list)
print('f1:{}, precision:{}, recall:{}'.format(f1, precision, recall))
