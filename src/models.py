import torch
import torch.nn as nn
from utils import circular_correlation
import numpy as np
import csv
import random


class Node:
    def __init__(self, content, category, self_id, sibling_id, parent_id, LR):
        self.content = content
        self.category = category
        self.self_id = self_id
        self.sibling_id = sibling_id
        self.parent_id = parent_id
        self.LR = LR
        if content == 'None':
            self.ready = False
            self.is_leaf = False
        else:
            self.ready = True
            self.is_leaf = True


class Tree:
    def __init__(self, self_id, sentence, node_list, REGULARIZED):
        self.self_id = self_id
        self.sentence = sentence
        self.node_list = node_list
        self.regularized = REGULARIZED

    def set_node_pair_list(self):
        left_nodes = []
        right_nodes = []
        for node in self.node_list:
            if node.LR == 'L':
                left_nodes.append(node)
            elif node.LR == 'R':
                right_nodes.append(node)
        node_pair_list = []
        for left_node in left_nodes:
            for right_node in right_nodes:
                if left_node.sibling_id == right_node.self_id:
                    node_pair_list.append((left_node, right_node))
        self.node_pair_list = node_pair_list

    def climb(self):
        for info in self.composition_info:
            left_node = self.node_list[info[0]]
            right_node = self.node_list[info[1]]
            parent_node = self.node_list[info[2]]
            parent_node.content = left_node.content + ' ' + right_node.content
            parent_node.vector = circular_correlation(
                left_node.vector, right_node.vector, self.regularized)

    def set_leaf_node_vector(self, weight_matrix):
        for node in self.node_list:
            if node.is_leaf:
                vector = weight_matrix[node.content_id[0]]
                if self.regularized:
                    vector = vector / torch.norm(vector)
                node.vector = vector

    # when initialize tree_list, each info of tree is automatically set
    def set_info_for_training(self, num_category):
        leaf_node_info = []
        label_list = []
        for node in self.node_list:
            if node.is_leaf:
                leaf_node_info.append([node.self_id, node.content_id[0]])
            # label with multiple bit corresponding to possible category id
            label = [0] * num_category
            for category_id in node.possible_category_id:
                label[category_id] = 1
            label_list.append(label)

        node_pair_list = self.node_pair_list
        composition_info = []
        while True:
            for node_pair in node_pair_list:
                left_node = node_pair[0]
                right_node = node_pair[1]
                parent_node = self.node_list[left_node.parent_id]
                if left_node.ready and right_node.ready:
                    composition_info.append(
                        [left_node.self_id, right_node.self_id, left_node.parent_id])
                    parent_node.ready = True
                    node_pair_list.remove(node_pair)
            if node_pair_list == []:
                break
        leaf_node_info = torch.tensor(leaf_node_info, dtype=torch.int, requires_grad=False)
        label_list = torch.tensor(label_list, dtype=torch.float, requires_grad=False)
        composition_info = torch.tensor(composition_info, dtype=torch.int, requires_grad=False)
        self.leaf_node_info = leaf_node_info
        self.label_list = label_list
        self.composition_info = composition_info
        self.reset_node_status()

    def reset_node_status(self):
        for node in self.node_list:
            if not node.is_leaf and node.ready:
                node.ready = False

    def convert_node_list_for_eval(self):
        converted_node_list = []
        sentence = self.node_list[-1].content_id
        for node in self.node_list:
            content = node.content_id
            for idx in range(len(sentence) - len(content) + 1):
                if content[0] == sentence[idx] and content == sentence[idx:idx + len(content)]:
                    break
            scope_start = idx
            scope_end = idx + len(content)
            converted_node_list.append((scope_start, scope_end, node.category_id))
        return converted_node_list


class Tree_List:
    def __init__(self, PATH_TO_DATA, REGULARIZED):
        self.regularized = REGULARIZED
        self.initialize_tree_list(PATH_TO_DATA)
        self.set_vocab_category()
        self.set_content_category_id()
        self.set_possible_category_id()
        self.set_info_for_training()

    # initialize tree list from txt data
    def initialize_tree_list(self, PATH_TO_DATA):
        with open(PATH_TO_DATA, 'r') as f:
            data_list = [data.strip() for data in f.readlines()]
        data_list = data_list[2:]
        data_list = [data.replace('\n', '') for data in data_list]

        block = []
        block_list = []
        for data in data_list:
            if data != '':
                block.append(data)
            else:
                block_list.append(block)
                block = []
        block_list.append(block)

        tree_list = []
        tree_id = 0
        for block in block_list:
            sentence = block[0]
            # remove 'num:' at the top of sentence
            for idx in range(len(sentence)):
                if sentence[idx] == ':':
                    sentence = sentence[idx + 1:]
                    break
            node_list = []
            for node_inf in block[1:]:
                node_inf = node_inf.split()
                content = node_inf[0]
                category = node_inf[1]
                self_id = int(node_inf[2])
                sibling_id = node_inf[3]
                if sibling_id != 'None':
                    sibling_id = int(sibling_id)
                else:
                    sibling_id = None
                parent_id = node_inf[4]
                if parent_id != 'None':
                    parent_id = int(parent_id)
                else:
                    parent_id = None
                LR = node_inf[5]
                node_list.append(
                    Node(
                        content,
                        category,
                        self_id,
                        sibling_id,
                        parent_id,
                        LR))
            tree_list.append(Tree(tree_id, sentence, node_list, self.regularized))
            tree_id += 1
        self.tree_list = tree_list

    # create dictionary of contents, categories and thier id
    def set_vocab_category(self):
        self.content_to_id = {}
        self.id_to_content = {}
        self.category_to_id = {}
        self.id_to_category = {}
        i = 0
        j = 0
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.is_leaf:  # only when node is leaf add to vocablary
                    if node.content not in self.content_to_id:
                        self.content_to_id[node.content] = i
                        self.id_to_content[i] = node.content
                        i += 1
                if node.category not in self.category_to_id:
                    self.category_to_id[node.category] = j
                    self.id_to_category[j] = node.category
                    j += 1

    def set_content_category_id(self):
        for tree in self.tree_list:
            tree.set_node_pair_list()
            while True:
                node_pair_list = tree.node_pair_list
                for node_pair in node_pair_list:
                    left_node = node_pair[0]
                    right_node = node_pair[1]
                    if left_node.ready and right_node.ready:
                        parent_node = tree.node_list[left_node.parent_id]
                        if left_node.is_leaf:
                            left_node.content_id = [self.content_to_id[left_node.content]]
                        if right_node.is_leaf:
                            right_node.content_id = [self.content_to_id[right_node.content]]
                        parent_node.content_id = []
                        for content_id in left_node.content_id:
                            parent_node.content_id.append(content_id)
                        for content_id in right_node.content_id:
                            parent_node.content_id.append(content_id)
                        parent_node.ready = True
                        node_pair_list.remove(node_pair)
                if node_pair_list == []:
                    break
            for node in tree.node_list:
                node.category_id = self.category_to_id[node.category]
            tree.reset_node_status()

    def set_possible_category_id(self):
        for tree in self.tree_list:
            for node in tree.node_list:
                node.possible_category_id = [node.category_id]
                for opponent_tree in self.tree_list:
                    for opponent_node in opponent_tree.node_list:
                        if node.content_id == opponent_node.content_id\
                                and opponent_node.category_id not in node.possible_category_id:
                            node.possible_category_id.append(opponent_node.category_id)

    def prepare_info_for_visualization(self, weight_matrix):
        vector_list = []
        content_info_dict = {}
        # content_info_dict : contains dicts of each content's info in the type of dict
        # inner dict contains possible category id to correspond content and the
        # embedded idx of content

        idx = 0  # this index shows where each content included in the embedded vector list
        for tree in self.tree_list:
            tree.set_leaf_node_vector(weight_matrix)
            tree.climb()
            for node in tree.node_list:
                # first time which the node.contetnt subscribed
                if node.content not in content_info_dict:
                    # initialize dictionary for each content
                    content_info = {}
                    content_info['category_id_list'] = [node.category_id]
                    content_info['idx'] = idx
                    content_info['plotted_category_list'] = []
                    content_info['plotted_category_id_list'] = []
                    content_info_dict[node.content] = content_info
                    vector_list.append(node.vector.detach().numpy())
                    idx += 1

                # already node.content included, but the category is different
                elif node.category_id not in content_info_dict[node.content]['category_id_list']:
                    content_info_dict[node.content]['category_id_list'].append(node.category_id)

        vector_list = np.array(vector_list)

        return vector_list, content_info_dict

    def set_info_for_training(self):
        for tree in self.tree_list:
            tree.set_node_pair_list()
            tree.set_info_for_training(len(self.category_to_id))

    def make_batch(self, BATCH_SIZE):
        sampled_tree_list = random.sample(self.tree_list, len(self.tree_list))
        batch_tree_list = []
        for idx in range(0, len(sampled_tree_list) - BATCH_SIZE, BATCH_SIZE):
            batch_tree_list.append(sampled_tree_list[idx:idx + BATCH_SIZE])
        return batch_tree_list

    def replace_vocab_category(self, tree_list):
        self.content_to_id = tree_list.content_to_id
        self.category_to_id = tree_list.category_to_id
        self.id_to_content = tree_list.id_to_content
        self.id_to_category = tree_list.id_to_category
        self.set_content_category_id()
        self.set_possible_category_id()
        self.set_info_for_training()


class Tree_Net(nn.Module):
    def __init__(self, tree_list, initial_weight_matrix):
        super(Tree_Net, self).__init__()
        self.regularized = tree_list.regularized
        self.num_embedding = initial_weight_matrix.shape[0]
        self.embedding_dim = initial_weight_matrix.shape[1]
        self.num_category = len(tree_list.category_to_id)
        self.embedding = nn.Embedding(
            self.num_embedding,
            self.embedding_dim,
            _weight=initial_weight_matrix)
        self.linear = nn.Linear(self.embedding_dim, self.num_category)
        self.sigmoid = nn.Sigmoid()

    def forward(self, leaf_node_info, composition_info):
        num_leaf_node = len(leaf_node_info)
        node_vectors = [0] * (2 * num_leaf_node - 1)
        for info in leaf_node_info:
            self_id = info[0]
            content_id = info[1]
            if self.regularized:
                node_vectors[self_id] = self.embedding(
                    content_id) / torch.norm(self.embedding(content_id))
            else:
                node_vectors[self_id] = self.embedding(content_id)
        for info in composition_info:
            left_node_id = info[0]
            right_node_id = info[1]
            parent_node_id = info[2]
            node_vectors[parent_node_id] = circular_correlation(
                node_vectors[left_node_id], node_vectors[right_node_id], True)
        node_vectors = torch.stack(node_vectors, dim=0)
        output = self.sigmoid(self.linear(node_vectors))
        return output


class History:
    def __init__(self, tree_net, tree_list, criteria, THRESHOLD):
        self.tree_net = tree_net
        self.tree_list = tree_list
        self.criteria = criteria
        self.THRESHOLD = THRESHOLD
        self.loss_history = np.array([])
        self.acc_history = np.array([])

    def cal_stat(self):
        loss = 0.0
        acc = 0.0
        for tree in self.tree_list.tree_list:
            leaf_node_info = tree.leaf_node_info
            label_list = tree.label_list
            composition_info = tree.composition_info
            output = self.tree_net(leaf_node_info, composition_info)
            loss += self.criteria(output, label_list) / output.shape[0]
            acc += self.cal_acc(output, label_list, self.THRESHOLD)
        loss = loss.detach().numpy() / len(self.tree_list.tree_list)
        acc = np.array(acc) / len(self.tree_list.tree_list)
        self.loss_history = np.append(self.loss_history, loss)
        self.acc_history = np.append(self.acc_history, acc)
        self.min_loss = np.min(self.loss_history)
        self.min_loss_idx = np.argmin(self.loss_history)
        self.max_acc = np.max(self.acc_history)
        self.max_acc_idx = np.argmax(self.acc_history)

    def cal_acc(self, output, label_list, THRESHOLD):
        num_correct = 0
        for i in range(output.shape[0]):
            for j in range(output.shape[1]):
                if output[i][j] >= THRESHOLD:
                    output[i][j] = 1.0
                else:
                    output[i][j] = 0.0
            if torch.count_nonzero(output[i] == label_list[i]) == output.shape[1]:
                num_correct += 1
        return num_correct / output.shape[0]

    def print_current_stat(self, name):
        print('{}-loss: {}'.format(name, self.loss_history[-1]))
        print('{}-acc: {}'.format(name, self.acc_history[-1]))

    def save(self, path):
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.loss_history)
            writer.writerow(self.acc_history)


class Condition_Setter:
    def __init__(self, PATH_TO_DIR):
        if int(input("random(0) or GloVe(1): ")) == 1:
            self.RANDOM = False
        else:
            self.RANDOM = True
        if int(input("reg(0) or not_reg(1): ")) == 1:
            self.REGULARIZED = False
        else:
            self.REGULARIZED = True
        if int(input("normal_loss(0) or original_loss(1): ")) == 1:
            self.USE_ORIGINAL_LOSS = True
        else:
            self.USE_ORIGINAL_LOSS = False
        embedding_dim = input("embedding_dim(default=100d): ")
        if embedding_dim != "":
            self.embedding_dim = int(embedding_dim)
        else:
            self.embedding_dim = 100
        self.set_path(PATH_TO_DIR)

    def set_path(self, PATH_TO_DIR):
        self.path_to_train_data = PATH_TO_DIR + "data/train.txt"
        self.path_to_test_data = PATH_TO_DIR + "data/test.txt"
        self.path_to_pretrained_weight_matrix = PATH_TO_DIR + \
            "data/glove_{}d.csv".format(self.embedding_dim)
        path_to_initial_weight_matrix = PATH_TO_DIR + "result/data/"
        path_to_model = PATH_TO_DIR + "result/model/"
        path_to_train_data_history = PATH_TO_DIR + "result/data/"
        path_to_test_data_history = PATH_TO_DIR + "result/data/"
        path_to_history_fig = PATH_TO_DIR + "result/fig/"
        fig_name = ""
        path_list = [
            path_to_initial_weight_matrix,
            path_to_model,
            path_to_train_data_history,
            path_to_test_data_history,
            path_to_history_fig,
            fig_name]
        for i in range(len(path_list)):
            if self.RANDOM:
                path_list[i] += "random"
            else:
                path_list[i] += "GloVe"
            if self.REGULARIZED:
                path_list[i] += "_reg"
            else:
                path_list[i] += "_not_reg"
            if self.USE_ORIGINAL_LOSS:
                path_list[i] += "_original_loss"
            path_list[i] += "_" + str(self.embedding_dim) + "d"
        self.path_to_initial_weight_matrix = path_list[0] + "_initial_weight_matrix.csv"
        self.path_to_model = path_list[1] + "_model.pth"
        self.path_to_train_data_history = path_list[2] + "_train_history.csv"
        self.path_to_test_data_history = path_list[3] + "_test_history.csv"
        self.path_to_history_fig = path_list[4] + "_history.png"
        self.fig_name = path_list[5].replace('_', ' ')
