import torch
import sys
import matplotlib.pyplot as plt
from utils import load_weight_matrix, visualize, interactive_visualize, set_random_seed
from models import Tree_List, Tree_Net
from sklearn.manifold import TSNE
import copy

FROM_RANDOM = True
REGULARIZED = True
USE_ORIGINAL_LOSS = False
EMBEDDING_DIM = 100
visualize_tree_id = range(100)
WITH_ARROW = False

args = sys.argv
if len(args) > 1 and args[1] == 'False':
    FROM_RANDOM = False
if len(args) > 2 and args[2] == 'False':
    REGULARIZED = False
if len(args) > 3 and args[3] == 'True':
    USE_ORIGINAL_LOSS = True
if len(args) > 4:
    EMBEDDING_DIM = int(args[4])
if len(args) > 5:
    visualize_tree_id = [int(args[5])]
    WITH_ARROW = True


PATH_TO_DIR = "/home/yryosuke0519/"

PATH_TO_DATA = PATH_TO_DIR + "Hol-CCG/data/train.txt"
PATH_TO_WEIGHT_MATRIX = PATH_TO_DIR + "Hol-CCG/data/pretrained_weight_matrix.csv"

path_to_initial_weight_matrix = PATH_TO_DIR + "Hol-CCG/result/data/"
path_to_model = PATH_TO_DIR + "Hol-CCG/result/model/"
path_to_map = PATH_TO_DIR + "Hol-CCG/result/fig/"
path_list = [
    path_to_initial_weight_matrix,
    path_to_model,
    path_to_map]

for i in range(len(path_list)):
    if FROM_RANDOM:
        path_list[i] += "random"
    else:
        path_list[i] += "GloVe"
    if REGULARIZED:
        path_list[i] += "_reg"
    else:
        path_list[i] += "_not_reg"
    if USE_ORIGINAL_LOSS:
        path_list[i] += "_original_loss"
    path_list[i] += "_" + str(EMBEDDING_DIM) + "d"
path_to_initial_weight_matrix = path_list[0] + "_initial_weight_matrix.csv"
path_to_model = path_list[1] + "_model.pth"
path_to_map = path_list[2] + "_map.png"

tree_list = Tree_List(PATH_TO_DATA, REGULARIZED)

initial_weight_matrix = load_weight_matrix(path_to_initial_weight_matrix, REGULARIZED)
initial_weight_matrix = torch.from_numpy(initial_weight_matrix)
tree_net = Tree_Net(tree_list, torch.zeros_like(initial_weight_matrix))
tree_net.load_state_dict(torch.load(path_to_model))
tree_net.eval()
trained_weight_matrix = tree_net.embedding.weight

if FROM_RANDOM:
    fig_name = "random"
else:
    fig_name = "GloVe"
if REGULARIZED:
    fig_name += " reg"
else:
    fig_name += " not reg"
if USE_ORIGINAL_LOSS:
    fig_name += " original loss"
fig_name += " " + str(EMBEDDING_DIM) + "d"

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

color_list = ['black', 'gray', 'lightcoral', 'red', 'saddlebrown', 'orange', 'yellowgreen',
              'forestgreen', 'turquoise', 'deepskyblue', 'blue', 'darkviolet', 'magenta']

# reset random seed for t-SNE
set_random_seed(0)

# make the map of trained state
vector_list, content_info_dict = tree_list.prepare_inf_for_visualization(
    trained_weight_matrix)
print("t-SNE working.....")
tsne = TSNE()
embedded = tsne.fit_transform(vector_list)

visualize_tree_list = []
for tree_id in visualize_tree_id:
    visualize_tree_list.append(tree_list.tree_list[tree_id])

fig0 = plt.figure(figsize=(10, 10))
ax0 = fig0.add_subplot()
visualize(
    ax=ax0,
    embedded=embedded,
    visualize_tree_list=visualize_tree_list,
    content_info_dict=copy.deepcopy(content_info_dict),
    group_list=group_list,
    color_list=color_list,
    fig_name=fig_name +
    " trained",
    WITH_ARROW=WITH_ARROW)
if not WITH_ARROW:
    fig0.savefig(
        path_to_map,
        dpi=300,
        orientation='portrait',
        transparent=False,
        pad_inches=0.0)

fig1 = plt.figure(figsize=(10, 10))
interactive_visualize(
    fig=fig1,
    embedded=embedded,
    visualize_tree_list=visualize_tree_list,
    content_info_dict=copy.deepcopy(content_info_dict),
    group_list=group_list,
    color_list=color_list)
plt.show()
