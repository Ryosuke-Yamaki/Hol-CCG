import numpy as np
import torch
import torch.nn as nn
from operator import itemgetter
from utils import circular_correlation, generate_random_weight_matrix


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
                left_node.vector, right_node.vector)

    def set_leaf_node_vector(self, weight_matrix):
        for node in self.node_list:
            if node.is_leaf:
                vector = weight_matrix[node.content_id[0]]
                if self.regularized:
                    vector = vector / torch.norm(vector)
                node.vector = vector

    # when initialize tree_list, each info of tree is automatically set
    def set_info_for_training(self, num_category, device=torch.device('cpu')):
        leaf_node_content_id = []
        label_list = []
        for node in self.node_list:
            if node.is_leaf:
                leaf_node_content_id.append(node.content_id[0])
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
                        [left_node.self_id, right_node.self_id])
                    parent_node.ready = True
                    node_pair_list.remove(node_pair)
            if node_pair_list == []:
                break
        self.reset_node_status()
        return [torch.tensor(leaf_node_content_id, dtype=torch.long, device=device),
                torch.tensor(label_list, dtype=torch.float, device=device),
                torch.tensor(composition_info, dtype=torch.long, device=device)]

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
            converted_node_list.append(
                (scope_start, scope_end, node.category_id))
        return converted_node_list


class Tree_List:
    def __init__(self, PATH_TO_DATA, REGULARIZED, device=torch.device('cpu')):
        self.regularized = REGULARIZED
        self.device = device
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
            tree_list.append(
                Tree(tree_id, sentence, node_list, self.regularized))
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
                            left_node.content_id = [
                                self.content_to_id[left_node.content]]
                        if right_node.is_leaf:
                            right_node.content_id = [
                                self.content_to_id[right_node.content]]
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
                            node.possible_category_id.append(
                                opponent_node.category_id)

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
                    content_info_dict[node.content]['category_id_list'].append(
                        node.category_id)

        vector_list = np.array(vector_list)

        return vector_list, content_info_dict

    def set_info_for_training(self):
        self.leaf_node_content_id = []
        self.label_list = []
        self.composition_info = []
        for tree in self.tree_list:
            tree.set_node_pair_list()
            info = tree.set_info_for_training(len(self.category_to_id), device=self.device)
            self.leaf_node_content_id.append(info[0])
            self.label_list.append(info[1])
            self.composition_info.append(info[2])

    def make_batch(self, BATCH_SIZE):
        num_tree = len(self.tree_list)
        # shuffle the tree_id in tree_list
        shuffled_tree_id = torch.randperm(num_tree, device=self.device)

        # make batch content id includes leaf node content id for each tree belongs to batch
        batch_leaf_content_id = []
        batch_label_list = []
        batch_composition_info = []
        for idx in range(0, num_tree - BATCH_SIZE, BATCH_SIZE):
            batch_tree_id_list = shuffled_tree_id[idx:idx + BATCH_SIZE]
            batch_leaf_content_id.append(list(itemgetter(
                *batch_tree_id_list)(self.leaf_node_content_id)))
            batch_label_list.append(list(itemgetter(
                *batch_tree_id_list)(self.label_list)))
            batch_composition_info.append(list(itemgetter(
                *batch_tree_id_list)(self.composition_info)))
        # the part cannot devided by BATCH_SIZE
        batch_leaf_content_id.append(list(itemgetter(
            *shuffled_tree_id[idx + BATCH_SIZE:])(self.leaf_node_content_id)))
        batch_label_list.append(list(itemgetter(
            *shuffled_tree_id[idx + BATCH_SIZE:])(self.label_list)))
        batch_composition_info.append(list(itemgetter(
            *shuffled_tree_id[idx + BATCH_SIZE:])(self.composition_info)))

        content_mask = []
        label_mask = []
        for idx in range(len(batch_leaf_content_id)):
            content_id = batch_leaf_content_id[idx]
            label_list = batch_label_list[idx]
            composition_list = batch_composition_info[idx]

            max_num_leaf_node = max([len(i) for i in content_id])
            # set the mask for each tree in batch
            # content mask is used for two purpose,
            # 1 - embedding leaf node vector
            # 2 - decide the incex of insert position of composed vector
            true_mask = [torch.ones(len(i), dtype=torch.bool, device=self.device)
                         for i in content_id]
            false_mask = [
                torch.zeros(
                    2 * max_num_leaf_node - 1 - len(i),
                    dtype=torch.bool,
                    device=self.device) for i in content_id]
            content_mask.append(torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(true_mask, false_mask)]))
            # make dummy content id to fill blank in batch
            dummy_content_id = [
                torch.zeros(
                    max_num_leaf_node - len(i),
                    dtype=torch.long,
                    device=self.device) for i in content_id]
            batch_leaf_content_id[idx] = torch.stack([torch.cat((i, j)) for (
                i, j) in zip(content_id, dummy_content_id)])

            max_num_label = max([len(i) for i in label_list])
            num_category = label_list[0].shape[1]
            # set the mask for label of each node in tree
            true_mask = [
                torch.ones(
                    (len(i), num_category),
                    dtype=torch.bool,
                    device=self.device) for i in label_list]
            false_mask = [
                torch.zeros(
                    (max_num_label - len(i), num_category),
                    dtype=torch.bool,
                    device=self.device) for i in label_list]
            label_mask.append(torch.stack([torch.cat((i, j))
                                           for (i, j) in zip(true_mask, false_mask)]))
            # make dummy label to fill blank in batch
            dummy_label = [
                torch.zeros(
                    2 * max_num_leaf_node - 1 - len(i),
                    i.shape[1],
                    dtype=torch.float,
                    device=self.device) for i in label_list]
            batch_label_list[idx] = torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(label_list, dummy_label)])

            # set mask for composition info in each batch
            max_num_composition = max([len(i) for i in composition_list])
            # make dummy compoisition info to fill blank in batch
            dummy_compositin_info = [
                torch.zeros(
                    max_num_composition - len(i),
                    i.shape[1],
                    dtype=torch.long,
                    device=self.device) for i in composition_list]
            batch_composition_info[idx] = torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(composition_list, dummy_compositin_info)])

        # return zipped batch information, when training extract each batch from zip itteration
        return zip(
            batch_leaf_content_id,
            content_mask,
            batch_composition_info,
            batch_label_list,
            label_mask)

    # the function use for the transfer of vocab & category from training to test tree
    def replace_vocab_category(self, tree_list):
        self.content_to_id = tree_list.content_to_id
        self.category_to_id = tree_list.category_to_id
        self.id_to_content = tree_list.id_to_content
        self.id_to_category = tree_list.id_to_category
        self.set_content_category_id()
        self.set_possible_category_id()
        self.set_info_for_training()


class Tree_Net(nn.Module):
    def __init__(self, tree_list, embedding_dim, initial_weight_matrix=None):
        super(Tree_Net, self).__init__()
        self.num_embedding = len(tree_list.content_to_id)
        self.num_category = len(tree_list.category_to_id)
        self.embedding_dim = embedding_dim
        if initial_weight_matrix is None:
            initial_weight_matrix = generate_random_weight_matrix(
                self.num_embedding, self.embedding_dim)
        initial_weight_matrix = torch.from_numpy(initial_weight_matrix).clone()
        self.embedding = nn.Embedding(
            self.num_embedding,
            self.embedding_dim,
            _weight=initial_weight_matrix)
        self.linear = nn.Linear(self.embedding_dim, self.num_category)
        self.sigmoid = nn.Sigmoid()

    # input batch as tuple of training info
    def forward(self, batch):
        # the content_id of leaf nodes
        leaf_content_id = batch[0]
        content_mask = batch[1]
        # the composition info of each tree
        composition_info = batch[2]
        vector, content_mask = self.embed_leaf_nodes(leaf_content_id, content_mask)
        vector = self.compose(vector, composition_info, content_mask)
        output = self.sigmoid(self.linear(vector))
        return output

    def embed_leaf_nodes(self, leaf_content_id, content_mask):
        vector = torch.zeros(
            (leaf_content_id.shape[0],
             2 * leaf_content_id.shape[1] - 1,
             self.embedding_dim), device=content_mask.device)
        # leaf_node_vector including padding tokens
        leaf_node_vector = self.embedding(leaf_content_id)
        # extract leaf node vector not padding tokens, using content_mask
        vector[content_mask.nonzero(as_tuple=True)
               ] = leaf_node_vector[content_mask.nonzero(as_tuple=True)]

        # record the number of leaf node of each tree in batch
        num_true = torch.count_nonzero(content_mask, dim=1)
        # change True mask of leaf node to False
        content_mask[content_mask.nonzero(as_tuple=True)] = False
        # add True mask for up coming composition
        content_mask[(torch.arange(content_mask.shape[0]), num_true)] = True
        return vector, content_mask

    def compose(self, vector, composition_info, content_mask):
        # itteration of composition
        for idx in range(composition_info.shape[1]):
            left_idx = composition_info[:, idx, 0]
            right_idx = composition_info[:, idx, 1]
            left_vector = vector[(torch.arange(len(left_idx)), left_idx)]
            right_vector = vector[(torch.arange(len(right_idx)), right_idx)]
            composed_vector = circular_correlation(left_vector, right_vector)
            vector[content_mask.nonzero(as_tuple=True)] = composed_vector
            content_mask = torch.roll(content_mask, shifts=1, dims=1)
        return vector
