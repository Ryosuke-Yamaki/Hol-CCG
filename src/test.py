import time
import torch
from models import Tree_Net
from utils import Condition_Setter
import os
from utils import load, load_weight_matrix
from parser import extract_rule, Parser
from torch.nn import Embedding


path_to_grammar = '/home/yryosuke0519/CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar'
path_to_raw_sentence = '/home/yryosuke0519/CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw'

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)
device = torch.device('cpu')
print('loading tree list...')
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

total = 0
count = 0
for sentence in test_sentence:
    total += 1
    if len(sentence.split()) < 55:
        count += 1
print(count / total)

correct = 0
for sentence, tree in zip(test_sentence, test_tree_list.tree_list):
    sentence = sentence.rstrip()
    total += 1
    if len(sentence.split()) < 55:
        count += 1
        # start = time.time()
        print(sentence)
        prob, backpointer, vector = parser.parse(sentence)
        final = list(prob.items())[-1]
        pred = sorted(final[1].items(), key=lambda x: x[1], reverse=True)[0]
        node = tree.node_list[-1]
        if node.category_id == pred[0]:
            correct += 1
        # print(node.category_id)
        # print(time.time() - start)
print(count / total)
print(correct / total)
