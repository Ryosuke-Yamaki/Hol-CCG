import numpy as np
from utils import load, dump, set_random_seed, Condition_Setter
from sklearn.decomposition import PCA
from collections import Counter
import torch
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE

condition = Condition_Setter(set_embedding_type=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

model = 'roberta-large_phrase(b).pth'
embedder = 'transformer'
n_dot = 10000
dev_tree_list = load(condition.path_to_dev_tree_list)
# dev_tree_list.tree_list = dev_tree_list.tree_list[:100]
tree_net = torch.load(condition.path_to_model + model,
                      map_location=device)
tree_net.device = device
tree_net.embedder = embedder
tree_net.eval()

dev_tree_list.tokenizer = tree_net.tokenizer
dev_tree_list.embedder = embedder


dev_tree_list.set_info_for_training(tokenizer=tree_net.tokenizer)
with torch.no_grad():
    dev_tree_list.set_vector(tree_net)

vector_list = []
node_type_list = []
node_cat_list = []
for tree in dev_tree_list.tree_list:
    for node in tree.node_list:
        vector_list.append(node.vector.to(torch.device('cpu')).detach().numpy())
        node_cat_list.append(node.category)
        if node.is_leaf:
            node_type_list.append(0)
        else:
            if 'S' in node.category and '/' not in node.category and '\\' not in node.category:
                node_type_list.append(2)
            else:
                node_type_list.append(1)

vector_list = np.array(vector_list)
node_type_list = np.array(node_type_list)[:n_dot]
pca = PCA(n_components=3)
compressed_vector = pca.fit_transform(vector_list)[:n_dot]
word_vector = compressed_vector[node_type_list == 0]
phrase_vector = compressed_vector[node_type_list == 1]
sentence_vector = compressed_vector[node_type_list == 2]

fig = plt.figure()
ax = fig.add_subplot(projection='3d')
ax.scatter(word_vector[:, 0], word_vector[:, 1], word_vector[:, 2], c='r', s=1)
ax.scatter(phrase_vector[:, 0], phrase_vector[:, 1], phrase_vector[:, 2], c='b', s=1)
ax.scatter(sentence_vector[:, 0], sentence_vector[:, 1], sentence_vector[:, 2], c='g', s=1)
fig.savefig('/home/yamaki-ryosuke/Hol-CCG/result/fig/map/' + model.replace('.pth', '.pdf'))
plt.show()
