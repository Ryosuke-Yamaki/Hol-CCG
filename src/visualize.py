from network import Tree_Net
from tree_structure import Tree_List
import matplotlib.pyplot as plt
import torch
import numpy as np
from sklearn.manifold import TSNE
from utils import load_weight_matrix

FROM_RANDOM = True
REGURALIZED = True
TRAINED = True

PATH_TO_DATA = '~/Hol-CCG/data/toy_data.txt'
PATH_TO_WEIGHT_MATRIX = '~/Hol-CCG/data/INITIAL_WEIGHT_MATRIX.csv'
MODEL_LOAD_PATH = '~/Hol-CCG/data/'
if FROM_RANDOM:
    MODEL_LOAD_PATH += 'from_random'
else:
    MODEL_LOAD_PATH += 'glove'
if REGURALIZED:
    MODEL_LOAD_PATH += '_reguralized.pth'
else:
    MODEL_LOAD_PATH += '.pth'

tree_list = Tree_List(PATH_TO_DATA, REGURALIZED)
INITIAL_WEIGHT_MATRIX = load_weight_matrix(PATH_TO_WEIGHT_MATRIX)
tree_net = Tree_Net(tree_list, INITIAL_WEIGHT_MATRIX, REGURALIZED, FROM_RANDOM)
tree_net.load_state_dict(torch.load(MODEL_LOAD_PATH))
tree_net.eval()
TRAINED_WEIGHT_MATRIX = tree_net.embedding.weight

for tree in tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            vector = INITIAL_WEIGHT_MATRIX[node.content_id]
            if REGURALIZED:
                vector = vector / torch.norm(vector)
            node.vector = vector

tensor_list = []
label_list = []
for tree in tree_list.tree_list:
    tree.climb()
    tensor_list.append(tree.make_node_vector_tensor().detach().numpy())
    label_list.append(tree.make_label_tensor().detach().numpy())
tensor_list = np.array(tensor_list)
label_list = np.array(label_list)
flatten_tensor_list = []
flatten_label_list = []
for i in range(len(tensor_list)):
    for j in range(len(tensor_list[i])):
        flatten_tensor_list.append(tensor_list[i][j])
        flatten_label_list.append(label_list[i][j])
tensor_list = np.array(flatten_tensor_list)
label_list = np.array(flatten_label_list)

torch.manual_seed(0)
tsne = TSNE(n_components=2, FROM_RANDOM_state=0, perplexity=30, n_iter=1000)
embedded = tsne.fit_transform(tensor_list)

group_list = []
group_list.append([1, 4])  # 1 名詞・名詞句
group_list.append([6])  # 2 文
group_list.append([0, 13])  # 3 冠詞・形容詞的な働きをする名詞
group_list.append([10, 15, 17, 20, 23])  # 4 名詞にかかる前置詞
group_list.append([8, 21])  # 5 動詞句にかかる前置詞
group_list.append([3, 7, 12, 22, 27])  # 6 他動詞
group_list.append([5])  # 7 自動詞
group_list.append([2])  # 8 助動詞・受動態のbe動詞
group_list.append([14])  # 9 疑問詞
group_list.append([11, 16, 24])  # 10 名詞句を修飾
group_list.append([9, 18, 25])  # 11 動詞句を修飾
group_list.append([19])  # 12 to不定詞
group_list.append([26])  # 13 接続詞

plt.figure(figsize=(15, 15))
color_list = ['black', 'gray', 'lightcoral', 'red', 'saddlebrown', 'orange', 'yellowgreen',
              'forestgreen', 'turquoise', 'deepskyblue', 'blue', 'darkviolet', 'magenta']
group_num = 0
for group in group_list:
    scatter_x_list = []
    scatter_y_list = []
    for i in range(len(label_list)):
        if label_list[i] in group:
            scatter_x_list.append(embedded[i][0])
            scatter_y_list.append(embedded[i][1])
    label = 'group ' + str(group_num + 1)
    plt.scatter(scatter_x_list, scatter_y_list, label=label, c=color_list[group_num])
    group_num += 1
plt.legend()
plt.title('regularized GloVe')
plt.savefig('/content/drive/MyDrive/Lab/HOL_CCG/Data/result/glove_reg_map.png',
            dpi=300, orientation='portrait', transparent=False, pad_inches=0.0)
plt.show()
