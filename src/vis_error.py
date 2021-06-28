from sklearn.decomposition import PCA
from collections import Counter
import os
import torch
import matplotlib.pyplot as plt
from utils import set_random_seed, Condition_Setter
from models import Tree_List, Tree_Net
from sklearn.manifold import TSNE
import numpy as np

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)
method = int(input('t-SNE(0) or PCA(1): '))
visualize_dim = int(input('2d(2) or 3d(3): '))

device = torch.device('cpu')

set_random_seed(0)
print('loading tree list...')
train_tree_list = Tree_List(condition.path_to_train_data, device=device)
test_tree_list = Tree_List(
    condition.path_to_test_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab,
    device=device)
test_tree_list.clean_tree_list()

tree_net = Tree_Net(train_tree_list, condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()

embedding = tree_net.embedding
test_tree_list.set_vector(embedding)

vector_list = []
counter = Counter()
linear = tree_net.linear

predict_list = []
cat_list = []
err_idx = []
idx = 0
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if counter[tuple(node.content_id)] == 0:
            counter[tuple(node.content_id)] += 1
            vector_list.append(node.vector.detach().numpy()[0])
            if node.category == 'N':
                cat_list.append('N')
            elif node.category == 'NP':
                cat_list.append('NP')
            elif 'S' in node.category and '/' not in node.category and '\\' not in node.category:
                cat_list.append('S')
            else:
                if node.is_leaf:
                    cat_list.append('Word')
                else:
                    cat_list.append('Phrase')
            output = linear(node.vector)
            predict = torch.topk(output, k=5)[1]
            if node.category_id in predict:
                predict_list.append(True)
            else:
                predict_list.append(False)
                err_idx.append(idx)
            idx += 1


if method == 0:
    method = TSNE(n_components=visualize_dim)
    path_to_map = condition.path_to_map + "_t-SNE.png"
    print("t-SNE working.....")
else:
    method = PCA(n_components=visualize_dim)
    path_to_map = condition.path_to_map + "_PCA.png"
    print("PCA working.....")

embedded = method.fit_transform(vector_list)

N = [0 for i in range(20)]
NP = [0 for i in range(20)]
S = [0 for i in range(20)]
Word = [0 for i in range(20)]
Phrase = [0 for i in range(20)]

stat = [[0, 0] for i in range(20)]
for idx in range(len(embedded[:, 2])):
    z = embedded[:, 2][idx]
    correct = predict_list[idx]
    cat = cat_list[idx]
    for i in range(20):
        if np.abs(z - (-0.75 + i * 0.1)) < 0.05:
            stat[i][0] += 1
            if cat == 'N':
                N[i] += 1
            elif cat == 'NP':
                NP[i] += 1
            elif cat == 'S':
                S[i] += 1
            elif cat == 'Word':
                Word[i] += 1
            elif cat == 'Phrase':
                Phrase[i] += 1
            if not correct:
                stat[i][1] += 1


num_sample = []
num_wrong = []
err = []
for info in stat:
    num_sample.append(info[0])
    num_wrong.append(info[1])
    err.append(info[1] / info[0])

x = np.linspace(-0.75, 1.25, len(num_sample))
plt.plot(x, num_sample, label='Total', linestyle='--')
plt.plot(x, num_wrong, label='Error', linestyle='--')
plt.plot(x, N, label='N')
plt.plot(x, NP, label='NP')
plt.plot(x, S, label='S')
plt.plot(x, Word, label='Word')
plt.plot(x, Phrase, label='Phrase')
plt.legend()
plt.show()

plt.plot(x, err, label='err ratio')
plt.legend()
plt.show()

fig = plt.figure(figsize=(10, 10))
if visualize_dim == 2:
    ax = fig.add_subplot()
    # for k, v in vis_dict.items():
    #     ax.scatter(embedded[v][:, 0], embedded[v][:, 1], s=1, label=k)
    # ax.legend(fontsize='large')
    # plt.xticks(fontsize='large')
    # plt.yticks(fontsize='large')
    # fig.savefig(path_to_map)
elif visualize_dim == 3:
    ax = fig.add_subplot(projection='3d')
    ax.scatter(embedded[err_idx][:, 0], embedded[err_idx][:, 1], embedded[err_idx][:, 2], s=1)
plt.show()
