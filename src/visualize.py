from utils import load_weight_matrix
import copy
from torchtext.vocab import Vocab
from utils import load
from sklearn.decomposition import PCA
from collections import Counter
import os
import torch
import matplotlib.pyplot as plt
from utils import set_random_seed, Condition_Setter
from models import Tree_List, Tree_Net
from sklearn.manifold import TSNE
import numpy as np
from torch.nn import Embedding


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
        content_info_dict,
        fig_name):
    ax = fig.add_subplot()
    ax.set_title(fig_name)
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


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)
method = int(input('t-SNE(0) or PCA(1): '))
visualize_dim = int(input('2d(2) or 3d(3): '))

device = torch.device('cpu')

set_random_seed(0)
print('loading tree list...')
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

if condition.embedding_type == "random":
    path_to_train_word_counter = PATH_TO_DIR + "Hol-CCG/data/train_word_counter.pickle"
    train_word_counter = load(path_to_train_word_counter)
    train_content_vocab = Vocab(train_word_counter, specials=['<unk>'])
    test_tree_list.content_vocab = train_content_vocab
    for tree in test_tree_list.tree_list:
        for node in tree.node_list:
            if node.is_leaf:
                node.content_id = [train_content_vocab[node.content]]
        for info in tree.composition_info:
            num_child = info[0]
            if num_child == 1:
                parent_node = tree.node_list[info[1]]
                child_node = tree.node_list[info[2]]
                parent_node.content_id = child_node.content_id
            else:
                parent_node = tree.node_list[info[1]]
                left_child_node = tree.node_list[info[2]]
                right_child_node = tree.node_list[info[3]]
                parent_node.content_id = left_child_node.content_id + right_child_node.content_id
    tree_list = []
    for tree in test_tree_list.tree_list:
        bit = 0
        for node in tree.node_list:
            if node.is_leaf:
                if 0 in node.content_id:
                    bit = 1
                    break
        if bit == 0:
            tree_list.append(tree)
    test_tree_list.tree_list = tree_list

NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)
weight_matrix = torch.tensor(
    load_weight_matrix(
        PATH_TO_DIR +
        "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(
            condition.embedding_dim)))
embedding = Embedding(NUM_VOCAB, condition.embedding_dim, _weight=weight_matrix)

test_tree_list.set_vector(embedding)

vector_list = []
vis_dict = {}
vis_dict['N'] = []
vis_dict['NP'] = []
vis_dict['S'] = []
vis_dict['Word'] = []
vis_dict['Phrase'] = []
counter = Counter()
idx = 0

for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if counter[tuple(node.content_id)] == 0:
            counter[tuple(node.content_id)] += 1
            vector_list.append(node.vector.detach().numpy()[0])
            if node.category == 'N':
                vis_dict['N'].append(idx)
            elif node.category == 'NP':
                vis_dict['NP'].append(idx)
            elif 'S' in node.category and '/' not in node.category and '\\' not in node.category:
                vis_dict['S'].append(idx)
            else:
                if node.is_leaf:
                    vis_dict['Word'].append(idx)
                else:
                    vis_dict['Phrase'].append(idx)
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
print("experined variance ratio= ", method.explained_variance_ratio_)

fig = plt.figure(figsize=(10, 10))
if visualize_dim == 2:
    ax = fig.add_subplot()
    for k, v in vis_dict.items():
        ax.scatter(embedded[v][:, 0], embedded[v][:, 1], s=1, label=k)
    ax.legend(fontsize='large')
    plt.xticks(fontsize='large')
    plt.yticks(fontsize='large')
    fig.savefig(path_to_map)
elif visualize_dim == 3:
    ax = fig.add_subplot(projection='3d')
    for k, v in vis_dict.items():
        ax.scatter(embedded[v][:, 0], embedded[v][:, 1], embedded[v][:, 2], s=1, label=k)
    ax.legend(fontsize='large')
plt.show()
