import torch
import torch.nn as nn
from torch.fft import fft, ifft
from torch import conj, mul
from utils import cal_norm_mean_std
import numpy as np


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
    def __init__(self, self_id, sentense, node_list, REGULARIZED):
        self.self_id = self_id
        self.sentense = sentense
        self.node_list = node_list
        self.regularized = REGULARIZED

    def make_node_pair_list(self):
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
        return node_pair_list

    def climb(self):
        node_pair_list = self.make_node_pair_list()
        roop_count = 0
        while True:
            for node_pair in node_pair_list:
                left_node = node_pair[0]
                right_node = node_pair[1]
                if left_node.ready and right_node.ready:
                    content = left_node.content + ' ' + right_node.content
                    self.node_list[left_node.parent_id].content = content
                    vector = self.circular_correlation(
                        left_node.vector, right_node.vector)
                    # regularize the norm of vector
                    if self.regularized:
                        vector = vector / torch.norm(vector)
                    self.node_list[left_node.parent_id].vector = vector
                    self.node_list[left_node.parent_id].ready = True
                    # print("step" + str(i) + ":")
                    # print(
                    #     left_node.content +
                    #     ':' +
                    #     left_node.category +
                    #     ' ' +
                    #     right_node.content +
                    #     ':' +
                    #     right_node.category)
                    # print('-> ' + self.node_list[left_node.parent_id].content +
                    #       ':' + self.node_list[left_node.parent_id].category)
                    # print()
                    node_pair_list.remove(node_pair)
            roop_count += 1
            if node_pair_list == []:
                break
            elif roop_count > 1000:
                print("***** Too many roop detected during tree climb! *****")
                exit()

    def set_leaf_node_vector(self, weight_matrix):
        for node in self.node_list:
            if node.is_leaf:
                vector = weight_matrix[node.content_id]
                if self.regularized:
                    vector = vector / torch.norm(vector)
                node.vector = vector

    # generate tensor as the input of tree net
    def make_node_vector_tensor(self):
        node_vector_list = []
        for node in self.node_list:
            node_vector_list.append(node.vector)
        return torch.stack(node_vector_list)

    def make_leaf_node_vector_tensor(self):
        node_vector_list = []
        for node in self.node_list:
            if node.is_leaf:
                node_vector_list.append(node.vector)
        return torch.stack(node_vector_list)

    def set_info_for_training(self):
        leaf_node_info = []
        label_list = []
        for node in self.node_list:
            if node.is_leaf:
                leaf_node_info.append([node.self_id, node.content_id])
            label_list.append(node.category_id)

        node_pair_list = self.make_node_pair_list()
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
        label_list = torch.tensor(label_list, dtype=torch.long, requires_grad=False)
        composition_info = torch.tensor(composition_info, dtype=torch.int, requires_grad=False)
        self.leaf_node_info = leaf_node_info
        self.label_list = label_list
        self.composition_info = composition_info


class Tree_List:
    def __init__(self, PATH_TO_DATA, REGULARIZED):
        self.regularized = REGULARIZED
        self.initialize_tree_list(PATH_TO_DATA)
        self.make_vocab_category()
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
            sentense = block[0]
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
            tree_list.append(Tree(tree_id, sentense, node_list, self.regularized))
            tree_id += 1
        self.tree_list = tree_list

    # create dictionary of contents, categories and thier id
    def make_vocab_category(self):
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
                    node.content_id = self.content_to_id[node.content]
                if node.category not in self.category_to_id:
                    self.category_to_id[node.category] = j
                    self.id_to_category[j] = node.category
                    j += 1
                node.category_id = self.category_to_id[node.category]

    def set_content_category_id(self):
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    node.content_id = self.content_to_id[node.content]
                node.category_id = self.category_to_id[node.category]

    def prepare_inf_for_visualization(self, weight_matrix):
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
            tree.set_info_for_training()


class Tree_Net(nn.Module):
    def __init__(self, tree_list, initial_weight_matrix):
        super(Tree_Net, self).__init__()
        self.num_embedding = initial_weight_matrix.shape[0]
        self.embedding_dim = initial_weight_matrix.shape[1]
        self.num_category = len(tree_list.category_to_id)
        self.embedding = nn.Embedding(
            self.num_embedding,
            self.embedding_dim,
            _weight=initial_weight_matrix)
        self.linear = nn.Linear(self.embedding_dim, self.num_category)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, leaf_node_info, composition_info):
        num_leaf_node = len(leaf_node_info)
        node_vectors = [0] * (2 * num_leaf_node - 1)
        for info in leaf_node_info:
            self_id = info[0]
            content_id = info[1]
            node_vectors[self_id] = self.embedding(content_id)
        for info in composition_info:
            left_node_id = info[0]
            right_node_id = info[1]
            parent_node_id = info[2]
            node_vectors[parent_node_id] = self.circular_correlation(
                node_vectors[left_node_id], node_vectors[right_node_id], True)
        node_vectors = torch.stack(node_vectors, dim=0)
        output = self.softmax(self.linear(node_vectors))
        return output

    def circular_correlation(self, a, b, REGULARIZED):
        a = conj(fft(a))
        b = fft(b)
        c = mul(a, b)
        c = ifft(c).real
        if REGULARIZED:
            return c / torch.norm(c)
        else:
            return c


class History:
    def __init__(self, tree_net, tree_list, criteria):
        self.tree_net = tree_net
        self.tree_list = tree_list
        self.criteria = criteria
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
            loss += self.criteria(output, label_list)
            acc += self.cal_acc(output, label_list)

        loss = loss.detach().numpy() / len(self.tree_list.tree_list)
        acc = acc.detach().numpy() / len(self.tree_list.tree_list)
        self.loss_history = np.append(self.loss_history, loss)
        self.acc_history = np.append(self.acc_history, acc)
        self.max_acc = np.max(self.acc_history)
        self.max_acc_idx = np.argmax(self.acc_history)

    def cal_acc(self, output, label_list):
        pred = torch.argmax(output, dim=1)
        num_correct = torch.count_nonzero(pred == label_list)
        return num_correct / output.shape[0]


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
        self.path_to_pretrained_weight_matrix = PATH_TO_DIR + "data/pretrained_weight_matrix.csv"
        path_to_initial_weight_matrix = PATH_TO_DIR + "result/data/"
        path_to_model = PATH_TO_DIR + "result/model/"
        path_to_train_data_history = PATH_TO_DIR + "result/data/"
        path_to_test_data_history = PATH_TO_DIR + "result/data/"
        path_to_history_fig = PATH_TO_DIR + "result/fig/"
        path_list = [
            path_to_initial_weight_matrix,
            path_to_model,
            path_to_train_data_history,
            path_to_test_data_history,
            path_to_history_fig]
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
