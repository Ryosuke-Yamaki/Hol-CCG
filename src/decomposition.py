from torch.fft import ifft
from torch.fft import fft
from collections import Counter
import numpy as npstandardize
from utils import circular_correlation, load, Condition_Setter, circular_convolution, inverse_circular_correlation
from sklearn.decomposition import PCA
import torch
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity as cos
from torch.nn.functional import normalize

condition = Condition_Setter(set_embedding_type=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = 'roberta-large_phrase_span_2022-01-22_13:49:06.pth'
dev_tree_list = load(condition.path_to_dev_tree_list)
# dev_tree_list.tree_list = dev_tree_list.tree_list[:100]
tree_net = torch.load(condition.path_to_model + model,
                      map_location=device)
tree_net.device = device
tree_net.eval()

dev_tree_list.tokenizer = tree_net.tokenizer
dev_tree_list.set_info_for_training(tokenizer=tree_net.tokenizer)

with torch.no_grad():
    dev_tree_list.set_vector(tree_net)

fail = 0
success = 0
total_cos = 0


counter = Counter()

for tree in dev_tree_list.tree_list:
    for node in tree.node_list:
        if not node.is_leaf and node.num_child == 2:
            parent = node
            left_child = tree.node_list[node.left_child_node_id]
            right_child = tree.node_list[node.right_child_node_id]
            p = parent.vector
            l = left_child.vector
            r = right_child.vector
            rr = inverse_circular_correlation(p, l, k=None)
            temp = cos(r, rr, dim=-1)
            # ll = inverse_circular_correlation(p, r, child_is_left=False)
            # temp = cos(l, ll, dim=-1)
            if temp < 0.9:
                print(left_child.content, right_child.content)
                p_ = fft(p)
                l_ = fft(l)
                norm_r = torch.norm(r)
                r_ = fft(r)
                counter[parent.category] += 1
                fail += 1
            else:
                success += 1
                total_cos += temp
print(total_cos / success)
print(f'{fail}/{fail+success}')
