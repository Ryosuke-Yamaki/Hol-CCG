import numpy as np
import torch
import csv
import random
from sklearn.manifold import TSNE
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


def cal_acc(output, label):
    num_True = 0
    num_False = 0

    for i in range(output.shape[0]):
        pred = torch.argmax(output[i])
        correct = label[i]
        if pred == correct:
            num_True += 1
        else:
            num_False += 1
    acc = num_True / (num_True + num_False)
    return acc


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


def visualize_result(tree_list, weight_matrix, group_list, path_to_map, fig_name):
    # climb the derivation tree and make vectors for each nodes
    for tree in tree_list.tree_list:
        for node in tree.node_list:
            if node.is_leaf:
                vector = weight_matrix[node.content_id]
                if tree.regularized:
                    vector = vector / torch.norm(vector)
                node.vector = vector
    vector_list = []
    label_list = []
    for tree in tree_list.tree_list:
        tree.climb()
        vector_list.append(tree.make_node_vector_tensor().detach().numpy())
        label_list.append(tree.make_label_tensor().detach().numpy())
    vector_list = np.array(vector_list)
    label_list = np.array(label_list)
    flatten_vector_list = []
    flatten_label_list = []
    for i in range(len(vector_list)):
        for j in range(len(vector_list[i])):
            flatten_vector_list.append(vector_list[i][j])
            flatten_label_list.append(label_list[i][j])
    vector_list = np.array(flatten_vector_list)
    label_list = np.array(flatten_label_list)

    set_random_seed(0)

    tsne = TSNE()
    print("t-SNE working.....")
    embedded = tsne.fit_transform(vector_list)

    cmap = plt.cm.gist_rainbow

    plt.figure(figsize=(10, 10))
    color_list = ['black', 'gray', 'lightcoral', 'red', 'saddlebrown', 'orange', 'yellowgreen',
                  'forestgreen', 'turquoise', 'deepskyblue', 'blue', 'darkviolet', 'magenta']
    for group_num in range(len(group_list)):
        scatter_x_list = []
        scatter_y_list = []
        idx = 0
        for label in label_list:
            if label in group_list[group_num]:
                scatter_x_list.append(embedded[idx, 0])
                scatter_y_list.append(embedded[idx, 1])
            idx += 1

        plt.scatter(
            scatter_x_list,
            scatter_y_list,
            c=color_list[group_num],
            cmap=cmap,
            label='group {}'.format(group_num + 1)
        )

    plt.legend()
    plt.title(fig_name)
    plt.savefig(path_to_map, dpi=300, orientation='portrait', transparent=False, pad_inches=0.0)
    plt.show()


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
