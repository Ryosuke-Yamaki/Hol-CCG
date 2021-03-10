import torch.nn as nn
import torch
from torch.fft import fft, ifft
from torch import conj, mul


class Node:
    def __init__(self, content, category, self_id, sibling_id, parent_id, LR):
        self.content = content
        self.category = category
        self.self_id = self_id
        self.sibling_id = sibling_id
        self.parent_id = parent_id
        self.LR = LR
        if content == 'None':
            self.ready = bool(False)
        else:
            self.ready = bool(True)


class Tree:
    def __init__(self, sentense, node_list, reguralized):
        self.sentense = sentense
        self.node_list = node_list
        self.set_leaf_node()
        self.reguralized = reguralized

    def make_node_pairs(self):
        left_nodes = []
        right_nodes = []
        for node in self.node_list:
            if node.LR == 'L':
                left_nodes.append(node)
            elif node.LR == 'R':
                right_nodes.append(node)
        node_pairs = []
        for left_node in left_nodes:
            for right_node in right_nodes:
                if left_node.sibling_id == right_node.self_id:
                    node_pairs.append((left_node, right_node))
        return node_pairs

    def climb(self):
        node_pairs = self.make_node_pairs()
        i = 1
        while True:
            for node_pair in node_pairs:
                left_node = node_pair[0]
                right_node = node_pair[1]
                if left_node.ready and right_node.ready:
                    content = left_node.content + ' ' + right_node.content
                    self.node_list[left_node.parent_id].content = content
                    vector = self.circular_correlation(
                        left_node.vector, right_node.vector)
                    # ベクトルの大きさを1に正規化
                    if self.reguralized:
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
                    node_pairs.remove(node_pair)
                    i += 1
            if node_pairs == []:
                break

    def set_leaf_node(self):
        for node in self.node_list:
            if node.ready:
                node.is_leaf = bool(True)
            else:
                node.is_leaf = bool(False)

    def circular_correlation(self, a, b):
        a = conj(fft(a))
        b = fft(b)
        c = mul(a, b)
        c = ifft(c).real
        return c

    def reset_node_status(self):
        for node in self.node_list:
            if node.is_leaf:
                node.ready = True
            else:
                node.ready = False

    # NNに入力するためのテンソルを作成
    def make_node_vector_tensor(self):
        node_vector_list = []
        for node in self.node_list:
            node_vector_list.append(node.vector)
        return torch.stack(node_vector_list)

    # 正解ラベルのテンソルを作成
    def make_label_tensor(self):
        label_list = []
        for node in self.node_list:
            label_list.append(torch.tensor(node.category_id))
        return torch.stack(label_list)


class Tree_List:
    def __init__(self, path_to_data, reguralized):
        self.reguralized = reguralized
        self.tree_list = self.make_tree_list_from_data(path_to_data)
        self.vocab, self.category = self.make_vocab_category(self.tree_list)
        self.add_content_category_id(self.tree_list, self.vocab, self.category)

    # テキストファイルから導出木のリストを出力する関数
    def make_tree_list_from_data(self, path_to_data):
        with open(path_to_data, 'r') as f:
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
                parent_id = node_inf[4]
                if parent_id != 'None':
                    parent_id = int(parent_id)
                LR = node_inf[5]
                node_list.append(
                    Node(
                        content,
                        category,
                        self_id,
                        sibling_id,
                        parent_id,
                        LR))
            tree_list.append(Tree(sentense, node_list, self.reguralized))
        return tree_list

    # create vocablary and category from trees
    def make_vocab_category(self, tree_list):
        vocab = {}
        category = {}
        i = 0
        j = 0
        for tree in tree_list:
            for node in tree.node_list:
                if node.content not in vocab:
                    vocab[node.content] = i
                    i += 1
                if node.category not in category:
                    category[node.category] = j
                    j += 1
        return vocab, category

    # add content_id to each node of tree
    def add_content_category_id(self, tree_list, vocab, category):
        for tree in tree_list:
            for node in tree.node_list:
                node.content_id = vocab[node.content]
                node.category_id = category[node.category]


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
            if node.is_leaf:  # リーフノードのベクトルを設定
                vector = self.embedding(torch.tensor(node.content_id))
                if tree.reguralized:
                    vector = vector / torch.norm(vector)
                node.vector = vector
        tree.climb()
        x = tree.make_node_vector_tensor()
        x = self.linear(x)
        x = self.softmax(x)
        return x
