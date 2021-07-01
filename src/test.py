import time
import torch
from models import Tree_Net
from utils import Condition_Setter
import os
from utils import load
from parser import extract_rule, Parser


path_to_grammar = '/home/yryosuke0519/CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar'
path_to_raw_sentence = '/home/yryosuke0519/CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw'

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

print('loading tree list...')
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")


device = torch.device('cpu')
NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)
tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY, condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

f = open(path_to_raw_sentence, 'r')
test_sentence = f.readlines()
f.close()

binary_rule, unary_rule = extract_rule(path_to_grammar, test_tree_list.category_vocab)
parser = Parser(tree_net, test_tree_list.content_vocab, binary_rule, unary_rule)

for sentence, tree in zip(test_sentence, test_tree_list.tree_list):
    sentence = sentence.rstrip()
    if len(sentence.split()) < 10:
        print(sentence)
        prob, backpointer, vector = parser.parse(sentence)
        final = list(prob.items())[-1]
        pred = sorted(final[1].items(), key=lambda x: x[1], reverse=True)[0]
        print(pred[0], pred[1])
        node = tree.node_list[-1]
        print(node.category_id)
