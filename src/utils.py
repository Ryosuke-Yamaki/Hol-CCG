import numpy as np
import torch
import csv
import random
import matplotlib.pyplot as plt


def load_weight_matrix(PATH_TO_WEIGHT_MATRIX, REGULARIZED):
    weight_matrix = []
    with open(PATH_TO_WEIGHT_MATRIX, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            weight_matrix.append(row)
    weight_matrix = np.array(weight_matrix).astype(np.float32)

    if REGULARIZED:
        for i in range(weight_matrix.shape[0]):
            weight_matrix[i] = weight_matrix[i] / np.linalg.norm(weight_matrix[i], ord=2)
    return weight_matrix


def generate_random_weight_matrix(NUM_VOCAB, EMBEDDING_DIM, REGULARIZED):
    weight_matrix = np.empty((NUM_VOCAB, EMBEDDING_DIM))
    for i in range(weight_matrix.shape[0]):
        weight_matrix[i] = np.random.normal(
            loc=0.0, scale=1 / np.sqrt(EMBEDDING_DIM), size=EMBEDDING_DIM)
        if REGULARIZED:
            weight_matrix[i] = weight_matrix[i] / np.linalg.norm(weight_matrix[i], ord=2)
    return weight_matrix.astype(np.float32)


def cal_norm_mean_std(tree):
    norm_list = []
    mean_list = []
    std_list = []
    for node in tree.node_list:
        norm_list.append(torch.norm(node.vector))
        mean_list.append(torch.mean(node.vector))
        std_list.append(torch.std(node.vector))
    norm = sum(norm_list) / len(norm_list)
    mean = sum(mean_list) / len(mean_list)
    std = sum(std_list) / len(std_list)
    return norm, mean, std


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


class Analyzer:
    def __init__(self, category, tree_net):
        self.inv_category = self.make_inv_category(category)
        self.tree_net = tree_net

    def make_inv_category(self, category):
        inv_category = {}
        for k, v in category.items():
            inv_category[v] = k
        return inv_category

    def analyze(self, tree):
        output = self.tree_net(tree)
        print(tree.sentense)
        print('acc: ' + str(cal_acc(output, tree.make_label_tensor())))
        print('*' * 50)
        i = 0
        for node in tree.node_list:
            content = node.content
            true_category = node.category
            pred_category = self.inv_category[int(torch.argmax(output[i]))]
            print('content: ' + content)
            print('true category: ' + true_category)
            print('pred category: ' + pred_category)
            if true_category == pred_category:
                print('True')
            else:
                print('False')
            print()
            i += 1
        print('*' * 50)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def original_loss(output, label, criteria, tree):
    loss = criteria(output, label)
    vector = tree.make_node_vector_tensor()
    norm = torch.norm(vector, dim=1)
    norm_base_line = torch.ones_like(norm)
    norm_loss = torch.sum(torch.abs(norm - norm_base_line))
    return loss + norm_loss


def make_batch(tree_list, BATCH_SIZE):
    batch_tree_list = []
    for index in range(0, len(tree_list) - BATCH_SIZE, BATCH_SIZE):
        batch_tree_list.append(tree_list[index:index + BATCH_SIZE])
    return batch_tree_list
