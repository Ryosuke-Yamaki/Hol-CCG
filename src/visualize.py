import torch
import matplotlib.pyplot as plt
from utils import set_random_seed
from models import Tree_List, Tree_Net
from sklearn.manifold import TSNE
import copy
import numpy as np


def visualize(
        ax,
        embedded,
        visualize_tree_list,
        content_info_dict,
        group_list,
        color_list,
        fig_name,
        WITH_ARROW=False):

    scatter_x_list = [[] for idx in range(len(group_list) + 1)]
    scatter_y_list = [[] for idx in range(len(group_list) + 1)]

    for tree in visualize_tree_list:
        # in this case, tree_list is composed with the trees which will be visualized
        for node in tree.node_list:
            # get the idx where the content embedded
            idx = content_info_dict[node.content]['idx']
            # when node.content is assigned to only one category
            if len(content_info_dict[node.content]['category_id_list']) == 1:
                group_num = judge_group(node.category_id, group_list)
                scatter_x_list[group_num].append(embedded[idx, 0])
                scatter_y_list[group_num].append(embedded[idx, 1])
            # when node.content is assigned to more than 2 categories
            else:
                # judge wheter different categories in the same group or not
                category_id_list = content_info_dict[node.content]['category_id_list']
                SAME_GROUP, group_num = judge_in_same_group(category_id_list, group_list)

                # when node.content never plotted
                if (len(content_info_dict[node.content]['plotted_category_id_list']) == 0):
                    # when the categories in same group and not yet plotted
                    if SAME_GROUP:
                        scatter_x_list[group_num].append(embedded[idx, 0])
                        scatter_y_list[group_num].append(embedded[idx, 1])
                    # when the categories in different group and not yet plotted
                    else:
                        scatter_x_list[-1].append(embedded[idx, 0])
                        scatter_y_list[-1].append(embedded[idx, 1])
            if node.category_id not in content_info_dict[node.content]['plotted_category_id_list']:
                content_info_dict[node.content]['plotted_category_id_list'].append(node.category_id)
                content_info_dict[node.content]['plotted_category_list'].append(node.category)

    for group_num in range(len(group_list)):
        ax.scatter(
            scatter_x_list[group_num],
            scatter_y_list[group_num],
            c=color_list[group_num],
            edgecolors='black',
            label='group {}'.format(group_num))
    # plot contents belong to mutiple groups
    ax.scatter(scatter_x_list[-1],
               scatter_y_list[-1],
               c='white',
               edgecolors='black',
               label='multiple groups')
    ax.legend()
    ax.set_title(fig_name)

    if WITH_ARROW:  # plot with arrows
        annotated_content_list = []
        for tree in visualize_tree_list:
            for node in tree.node_list:
                start_idx = content_info_dict[node.content]['idx']
                start_point = [embedded[start_idx, 0], embedded[start_idx, 1]]
                if node.parent_id is not None:
                    parent_node = tree.node_list[node.parent_id]
                    end_idx = content_info_dict[parent_node.content]['idx']
                    end_point = [embedded[end_idx, 0], embedded[end_idx, 1]]
                    ax.annotate(
                        '',
                        xy=end_point,
                        xytext=start_point,
                        arrowprops=dict(
                            arrowstyle='->',
                            connectionstyle='arc3',
                            facecolor='black',
                            edgecolor='black'))

                if node.content not in annotated_content_list:
                    text = node.content
                    for category in content_info_dict[node.content]['plotted_category_list']:
                        text += ('\n' + category)
                    ax.annotate(text, xy=np.array(start_point) + np.array([0.5, 0.5]))
                    annotated_content_list.append(node.content)


def interactive_visualize(
        fig,
        embedded,
        visualize_tree_list,
        group_list,
        color_list,
        content_info_dict):
    ax = fig.add_subplot()
    scatter_x_list = []
    scatter_y_list = []
    plotted_content_list = []
    color_list_for_plot = []
    for tree in visualize_tree_list:
        # in this case, tree_list is composed of the trees which will be visualized
        for node in tree.node_list:
            # get the idx where the content embedded
            idx = content_info_dict[node.content]['idx']
            # when node.content is assigned to only one category
            if len(content_info_dict[node.content]['category_id_list']) == 1:
                category_id = content_info_dict[node.content]['category_id_list'][0]
                color_list_for_plot.append(color_list[judge_group(category_id, group_list)])
                scatter_x_list.append(embedded[idx, 0])
                scatter_y_list.append(embedded[idx, 1])
                plotted_content_list.append(node.content)
            # when node.content is assigned to more than 2 categories
            else:
                # judge wheter different categories in the same group or not
                category_id_list = content_info_dict[node.content]['category_id_list']
                SAME_GROUP, group_num = judge_in_same_group(category_id_list, group_list)

                # when node.content never plotted
                if (len(content_info_dict[node.content]['plotted_category_id_list']) == 0):
                    scatter_x_list.append(embedded[idx, 0])
                    scatter_y_list.append(embedded[idx, 1])
                    plotted_content_list.append(node.content)
                    # when the categories in same group and not yet plotted
                    if SAME_GROUP:
                        color_list_for_plot.append(color_list[group_num])
                    # when the categories in different group and not yet plotted
                    else:
                        color_list_for_plot.append('white')
            if node.category_id not in content_info_dict[node.content]['plotted_category_id_list']:
                content_info_dict[node.content]['plotted_category_id_list'].append(node.category_id)
                content_info_dict[node.content]['plotted_category_list'].append(node.category)

    sc = plt.scatter(scatter_x_list, scatter_y_list, c=color_list_for_plot, edgecolors='black')
    annot = ax.annotate("", xy=(0, 0), xytext=(20, 20), textcoords="offset points",
                        bbox=dict(boxstyle="round", fc="w"),
                        arrowprops=dict(arrowstyle="->"))
    annot.set_visible(False)

    def update_annot(ind):
        pos = sc.get_offsets()[ind["ind"][0]]
        annot.xy = pos
        content = plotted_content_list[ind["ind"][0]]
        text = content
        for category in content_info_dict[content]['plotted_category_list']:
            text += ('\n' + category)
        annot.set_text(text)
        annot.get_bbox_patch().set_facecolor('white')

    def hover(event):
        vis = annot.get_visible()
        if event.inaxes == ax:
            cont, ind = sc.contains(event)
            if cont:
                update_annot(ind)
                annot.set_visible(True)
                fig.canvas.draw_idle()
            else:
                if vis:
                    annot.set_visible(False)
                    fig.canvas.draw_idle()

    fig.canvas.mpl_connect("motion_notify_event", hover)


def judge_group(category_id, group_list):
    for group_num in range(len(group_list)):
        if category_id in group_list[group_num]:
            return group_num


def judge_in_same_group(category_id_list, group_list):
    group_num_temp_0 = judge_group(category_id_list[0], group_list)
    SAME_GROUP = True
    for category_id in category_id_list[1:]:
        group_num_temp_1 = judge_group(category_id, group_list)
        if group_num_temp_0 == group_num_temp_1:
            group_num_temp_0 = group_num_temp_1
        else:
            SAME_GROUP = False
            break
    return SAME_GROUP, group_num_temp_0


if int(input("random(0) or GloVe(1): ")) == 1:
    FROM_RANDOM = False
else:
    FROM_RANDOM = True
if int(input("reg(0) or not_reg(1): ")) == 1:
    REGULARIZED = False
else:
    REGULARIZED = True
if int(input("normal_loss(0) or original_loss(1): ")) == 1:
    USE_ORIGINAL_LOSS = True
else:
    USE_ORIGINAL_LOSS = False
embedding_dim = input("embedding_dim(default=100d): ")
if embedding_dim != "":
    embedding_dim = int(embedding_dim)
else:
    embedding_dim = 100
target_tree_id = input("target tree id(default=all): ")
if target_tree_id != "":
    target_tree_id = [int(x) for x in target_tree_id.split(",")]
else:
    target_tree_id = None
if int(input("without arrow(0) or with arrow(1): ")) == 1:
    WITH_ARROW = True
else:
    WITH_ARROW = False

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
    path_list[i] += "_" + str(embedding_dim) + "d"
path_to_initial_weight_matrix = path_list[0] + "_initial_weight_matrix.csv"
path_to_model = path_list[1] + "_model.pth"
path_to_map = path_list[2] + "_map.png"

tree_list = Tree_List(PATH_TO_DATA, REGULARIZED)
if target_tree_id is None:
    visualize_tree_list = tree_list.tree_list
else:
    visualize_tree_list = []
    for tree_id in target_tree_id:
        visualize_tree_list.append(tree_list.tree_list[tree_id])

tree_net = Tree_Net(tree_list, embedding_dim)
tree_net = torch.load(path_to_model)
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
fig_name += " " + str(embedding_dim) + "d"

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
vector_list, content_info_dict = tree_list.prepare_info_for_visualization(
    trained_weight_matrix)
print("t-SNE working.....")
tsne = TSNE()
embedded = tsne.fit_transform(vector_list)

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
