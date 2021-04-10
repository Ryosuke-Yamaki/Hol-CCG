import torch
import torch.nn as nn
from torch.fft import fft, ifft
from torch import conj, mul
from utils import cal_acc, cal_norm_mean_std
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
        i = 1
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
                    i += 1
            roop_count += 1
            if node_pair_list == []:
                break
            elif roop_count > 1000:
                print("***** Too many roop detected during tree climb! *****")
                exit()

    def circular_correlation(self, a, b):
        a = conj(fft(a))
        b = fft(b)
        c = mul(a, b)
        c = ifft(c).real
        return c

    def set_leaf_node_vector(self, weight_matrix):
        for node in self.node_list:
            if node.is_leaf:
                vector = weight_matrix[node.content_id]
                if self.regularized:
                    vector = vector / torch.norm(vector)
                node.vector = vector

    # reset node status for next coming epoch
    def reset_node_status(self):
        for node in self.node_list:
            if node.is_leaf:
                node.ready = True
            else:
                node.ready = False

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

    # generate tnesor as the correct category label
    def make_label_tensor(self):
        label_list = []
        for node in self.node_list:
            label_list.append(torch.tensor(node.category_id))
        return torch.stack(label_list)


class Tree_List:
    def __init__(self, PATH_TO_DATA, REGULARIZED):
        self.regularized = REGULARIZED
        self.tree_list = self.initialize_tree_list(PATH_TO_DATA)
        self.vocab, self.category = self.make_vocab_category(self.tree_list)
        self.add_content_category_id(self.vocab, self.category)

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
        return tree_list

    # create vocablary and category from tree list
    def make_vocab_category(self, tree_list):
        vocab = {}
        category = {}
        i = 0
        j = 0
        for tree in tree_list:
            for node in tree.node_list:
                if node.content not in vocab and node.content != 'None':
                    vocab[node.content] = i
                    i += 1
                if node.category not in category:
                    category[node.category] = j
                    j += 1
        return vocab, category

    # add content_id and category_id to each node of tree
    def add_content_category_id(self, vocab, category):
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.content != 'None':
                    node.content_id = vocab[node.content]
                node.category_id = category[node.category]

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


class Tree_Net(nn.Module):
    def __init__(self, tree_list, initial_weight_matrix):
        super(Tree_Net, self).__init__()
        self.num_embedding = initial_weight_matrix.shape[0]
        self.embedding_dim = initial_weight_matrix.shape[1]
        self.num_category = len(tree_list.category)
        self.embedding = nn.Embedding(
            self.num_embedding,
            self.embedding_dim,
            _weight=initial_weight_matrix)
        self.linear = nn.Linear(self.embedding_dim, self.num_category)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, tree):
        for node in tree.node_list:
            if node.is_leaf:
                vector = self.embedding(torch.tensor(node.content_id))
                if tree.regularized:
                    vector = vector / torch.norm(vector)
                node.vector = vector
        tree.climb()
        x = tree.make_node_vector_tensor()
        x = self.linear(x)
        x = self.softmax(x)
        return x

    def cal_stat(self, tree_list, criteria):
        total_loss = 0.0
        total_acc = 0.0
        total_norm = 0.0
        total_mean = 0.0
        total_std = 0.0
        for tree in tree_list.tree_list:
            label = tree.make_label_tensor()
            output = self.forward(tree)
            loss = criteria(output, label)
            acc = cal_acc(output, label)
            norm, mean, std = cal_norm_mean_std(tree)
            total_loss += loss
            total_acc += acc
            total_norm += norm
            total_mean += mean
            total_std += std
            tree.reset_node_status()

        len_tree_list = len(tree_list.tree_list)
        stat_list = []
        stat_list.append(float(total_loss / len_tree_list))
        stat_list.append(float(total_acc / len_tree_list))
        stat_list.append(float(total_norm / len_tree_list))
        stat_list.append(float(total_mean / len_tree_list))
        stat_list.append(float(total_std / len_tree_list))
        return stat_list
