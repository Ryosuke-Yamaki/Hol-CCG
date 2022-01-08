import numpy as npstandardize
from utils import load, Condition_Setter, circular_convolution
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

model = 'roberta-large_phrase_span_2021-12-20_13:53:40.pth'
dev_tree_list = load(condition.path_to_dev_tree_list)
dev_tree_list.tree_list = dev_tree_list.tree_list[:100]
tree_net = torch.load(condition.path_to_model + model,
                      map_location=device)
tree_net.device = device
tree_net.eval()

dev_tree_list.tokenizer = tree_net.tokenizer
dev_tree_list.set_info_for_training(tokenizer=tree_net.tokenizer)

with torch.no_grad():
    dev_tree_list.set_vector(tree_net)

total = 0
correct = 0
total_cos = 0
cos_list = []
for tree in dev_tree_list.tree_list:
    for node in tree.node_list:
        if not node.is_leaf and node.num_child == 2:
            parent = node
            left_child = tree.node_list[node.left_child_node_id]
            right_child = tree.node_list[node.right_child_node_id]
            p = parent.vector
            l = left_child.vector
            r = right_child.vector
            r_ = circular_convolution(l, p)
            if right_child.is_leaf:
                classifier = tree_net.word_ff
                vocab = dev_tree_list.word_category_vocab
            else:
                classifier = tree_net.phrase_ff
                vocab = dev_tree_list.phrase_category_vocab
            cat_score = classifier(r_)
            cat_id = torch.argmax(cat_score)
            predict_cat = vocab.itos[cat_id]
            # print('parent:{}, {}'.format(parent.content, parent.category))
            # print('left:{}, {}'.format(left_child.content, left_child.category))
            # print('right:{}, {}'.format(right_child.content, right_child.category))
            # print('predicted right category:{}'.format(predict_cat))
            # print('cos similarity:{}\n'.format(cos(r, r_, dim=-1)))
            total_cos += cos(r, r_, dim=-1)
            cos_list.append(cos(r, r_, dim=-1).cpu().numpy())
            total += 1
print(total_cos / total)
print(correct / total)
plt.hist(cos_list, bins=10)
plt.show()
a = 0
