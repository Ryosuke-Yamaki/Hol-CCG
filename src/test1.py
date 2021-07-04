import time
from tools_for_easyccg import Converter, easyccg
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

test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

new_weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim))
new_weight_matrix = torch.tensor(new_weight_matrix)

NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)

embedding = Embedding(NUM_VOCAB, condition.embedding_dim, _weight=new_weight_matrix)

tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()
linear = tree_net.linear

start = time.time()
result = easyccg(PATH_TO_DIR, n=10)

converter = Converter(
    test_tree_list.content_vocab,
    test_tree_list.category_vocab,
    embedding,
    linear)

id_list = [int(i.replace('ID=', '')) for i in result[0::2]]
idx_list = list(range(1, len(result), 2))

current_id = 1
max_socre_idx_list = []
max_score = 0
for id, idx in zip(id_list, idx_list):
    if id == current_id:
        tree = converter.convert_to_tree(result[idx])
        score = tree.cal_score(linear)
        if score > max_score:
            max_score = score
            max_score_idx = idx
    # when the target sentence is swhiched
    else:
        max_socre_idx_list.append(max_score_idx)
        print(current_id)
        current_id = id
        tree = converter.convert_to_tree(result[idx])
        max_score = tree.cal_score(linear)
        max_score_idx = idx
max_socre_idx_list.append(max_score_idx)
print(time.time() - start)
