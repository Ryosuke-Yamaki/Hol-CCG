from sklearn.decomposition import PCA
from collections import Counter
from threading import Condition
import torch
import matplotlib.pyplot as plt
from utils import set_random_seed
from models import Tree, Tree_List, Tree_Net
from sklearn.manifold import TSNE
import copy
import numpy as np
from utils import Condition_Setter, single_circular_correlation


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


device = torch.device('cpu')

PATH_TO_DIR = "/home/yryosuke0519/"

condition = Condition_Setter(PATH_TO_DIR)

print('loading data...')
train_tree_list = Tree_List(condition.path_to_train_data)
test_tree_list = Tree_List(
    condition.path_to_test_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab,
    device=device)
test_tree_list.clean_tree_list()
train_tree_list.set_info_for_training()
test_tree_list.set_info_for_training()
tree_net = Tree_Net(train_tree_list, condition.embedding_dim)
tree_net = torch.load(condition.path_to_model, map_location=torch.device('cpu'))
tree_net.eval()

print('caluclating vectors...')
embedding = tree_net.embedding
with torch.no_grad():
    for tree in test_tree_list.tree_list:
        for node in tree.node_list:
            if node.is_leaf:
                node.vector = torch.squeeze(embedding(torch.tensor(node.content_id)))
                node.vector = node.vector / torch.norm(node.vector)

    for tree in test_tree_list.tree_list:
        for composition_info in tree.composition_info:
            num_child = composition_info[0]
            parent_node = tree.node_list[composition_info[1]]
            if num_child == 1:
                child_node = tree.node_list[composition_info[2]]
                parent_node.vector = child_node.vector
            else:
                left_node = tree.node_list[composition_info[2]]
                right_node = tree.node_list[composition_info[3]]
                parent_node.vector = single_circular_correlation(
                    left_node.vector, right_node.vector)

with_idx = train_tree_list.content_vocab['with']
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if with_idx in node.content_id and len(node.content_id) <= 3:
            content = []
            for content_id in node.content_id:
                content.append(train_tree_list.content_vocab.itos[content_id])
            print(' '.join(content))
            print(node.category)

counter = Counter()
vector_list = []
# leaf_vector_idx = []
# phrase_vector_idx = []
# n_vector_idx = []
# np_vector_idx = []
# s_vector_idx = []
# leaf_mean = []
# phrase_mean = []
# n_mean = []
# np_mean = []
# s_mean = []

vis_dict = {}
vis_dict['N'] = []
vis_dict['NP'] = []
vis_dict['Word'] = []
vis_dict['Phrase'] = []
idx = 0
# test_tree_list.tree_list = test_tree_list.tree_list[:30]
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if counter[tuple(node.content_id)] == 0:
            counter[tuple(node.content_id)] += 1
            vector_list.append(node.vector.detach().numpy())
            if node.category == 'N':
                vis_dict['N'].append(idx)
                # n_vector_idx.append(idx)
                # n_mean.append(torch.mean(node.vector).item())
            elif node.category == 'NP':
                vis_dict['NP'].append(idx)
                # np_vector_idx.append(idx)
                # np_mean.append(torch.mean(node.vector).item())
            elif 'S' in node.category and '/' not in node.category and '\\' not in node.category:
                if node.category in vis_dict:
                    vis_dict[node.category].append(idx)
                else:
                    vis_dict[node.category] = [idx]
                # s_mean.append(torch.mean(node.vector).item())
            else:
                if node.is_leaf:
                    vis_dict['Word'].append(idx)
                    # leaf_vector_idx.append(idx)
                    # leaf_mean.append(torch.mean(node.vector).item())
                else:
                    vis_dict['Phrase'].append(idx)
                    # phrase_vector_idx.append(idx)
                    # phrase_mean.append(torch.mean(node.vector).item())
            idx += 1

# print(sum(n_mean) / len(n_mean))
# print(sum(np_mean) / len(np_mean))
# print(sum(s_mean) / len(s_mean))
# print(sum(leaf_mean) / len(leaf_mean))
# print(sum(phrase_mean) / len(phrase_mean))

# print(np.std(n_mean))
# print(np.std(np_mean))
# print(np.std(s_mean))
# print(np.std(leaf_mean))
# print(np.std(phrase_mean))

set_random_seed(0)

print("t-SNE working.....")
tsne = TSNE(n_components=2)
pca = PCA(n_components=2)
# embedded = tsne.fit_transform(vector_list)
embedded = pca.fit_transform(vector_list)
fig = plt.figure(figsize=(10, 10))
# ax = fig.add_subplot(projection='3d')
ax = fig.add_subplot()

for k, v in vis_dict.items():
    if k != 'Phrase':
        ax.scatter(embedded[v][:, 0], embedded[v][:, 1], s=5, label=k)
# ax.scatter(embedded[n_vector_idx][:, 0], embedded[n_vector_idx][:, 1],
#            embedded[n_vector_idx][:, 2], s=5, c='r', label='N')
# ax.scatter(embedded[np_vector_idx][:, 0], embedded[np_vector_idx][:, 1],
#            embedded[np_vector_idx][:, 2], s=5, c='g', label='NP')
# ax.scatter(embedded[s_vector_idx][:, 0], embedded[s_vector_idx][:, 1],
#            embedded[s_vector_idx][:, 2], s=5, c='b', label='S')
# ax.scatter(embedded[leaf_vector_idx][:, 0], embedded[leaf_vector_idx][:, 1],
#            embedded[leaf_vector_idx][:, 2], s=5, c='y', label='Word')
# ax.scatter(embedded[phrase_vector_idx][:, 0], embedded[phrase_vector_idx][:, 1],
#            embedded[phrase_vector_idx][:, 2], s=5, c='k', label='Phrase')

# ax = fig.add_subplot()
# ax.scatter(embedded[n_vector_idx][:, 0], embedded[n_vector_idx][:, 1],
#            s=5, label='N')
# ax.scatter(embedded[np_vector_idx][:, 0], embedded[np_vector_idx][:, 1],
#            s=5, label='NP')
# ax.scatter(embedded[s_vector_idx][:, 0], embedded[s_vector_idx][:, 1],
#            s=5, label='S')
# ax.scatter(embedded[leaf_vector_idx][:, 0], embedded[leaf_vector_idx][:, 1],
#            s=5, label='Word')
# ax.scatter(embedded[phrase_vector_idx][:, 0], embedded[phrase_vector_idx][:, 1],
#            s=5, label='Phrase')
ax.legend(fontsize='large')
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
plt.show()
fig.savefig('detailed_S_test.png')
plt.show()
