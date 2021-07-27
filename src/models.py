import re
import numpy as np
import torch
import torch.nn as nn
from torchtext.vocab import Vocab
from operator import itemgetter
from utils import circular_correlation, generate_random_weight_matrix, single_circular_correlation
from collections import Counter


class Node:
    def __init__(self, node_info):
        if node_info[0] == 'True':
            self.is_leaf = True
        else:
            self.is_leaf = False
        self.self_id = int(node_info[1])
        if self.is_leaf:
            content = node_info[2].lower()
            content = re.sub(r'\d+', '0', content)
            content = re.sub(r'\d,\d', '0', content)
            self.content = content
            self.category = node_info[3]
            self.ready = True
        else:
            self.category = node_info[2]
            self.num_child = int(node_info[3])
            self.ready = False
            if self.num_child == 1:
                self.child_node_id = int(node_info[4])
            else:
                self.left_child_node_id = int(node_info[4])
                self.right_child_node_id = int(node_info[5])


class Tree:
    def __init__(self, self_id, node_list):
        self.self_id = self_id
        self.node_list = node_list

    def set_node_composition_info(self):
        self.composition_info = []
        while True:
            num_ready_node = 0
            for node in self.node_list:
                if node.ready:
                    num_ready_node += 1
                elif not node.is_leaf and not node.ready:
                    if node.num_child == 1:
                        child_node = self.node_list[node.child_node_id]
                        if child_node.ready:
                            node.content_id = child_node.content_id
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, child_node.self_id, 0])
                    else:  # when node has two children
                        left_child_node = self.node_list[node.left_child_node_id]
                        right_child_node = self.node_list[node.right_child_node_id]
                        if left_child_node.ready and right_child_node.ready:
                            node.content_id = left_child_node.content_id + right_child_node.content_id
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, left_child_node.self_id, right_child_node.self_id])
            if num_ready_node == len(self.node_list):
                break

    def climb(self):
        for info in self.composition_info:
            left_node = self.node_list[info[0]]
            right_node = self.node_list[info[1]]
            parent_node = self.node_list[info[2]]
            parent_node.content = left_node.content + ' ' + right_node.content
            parent_node.vector = single_circular_correlation(
                left_node.vector, right_node.vector)

    def set_leaf_node_vector(self, weight_matrix):
        for node in self.node_list:
            if node.is_leaf:
                vector = weight_matrix[node.content_id[0]]
                if self.regularized:
                    vector = vector / torch.norm(vector)
                node.vector = vector

    def reset_node_status(self):
        for node in self.node_list:
            if not node.is_leaf and node.ready:
                node.ready = False

    def make_correct_node_list(self):
        correct_node_list = []
        top_node = self.node_list[-1]
        top_node.start_idx = 0
        top_node.end_idx = len(top_node.content_id)
        if not top_node.is_leaf:
            correct_node_list.append((0, len(top_node.content_id), top_node.category_id))
        for info in reversed(self.composition_info):
            num_child = info[0]
            if num_child == 1:
                parent_node = self.node_list[info[1]]
                child_node = self.node_list[info[2]]
                child_node.start_idx = parent_node.start_idx
                child_node.end_idx = parent_node.end_idx
                if not child_node.is_leaf:
                    correct_node_list.append(
                        (child_node.start_idx, child_node.end_idx, child_node.category_id))
            else:
                parent_node = self.node_list[info[1]]
                left_child_node = self.node_list[info[2]]
                right_child_node = self.node_list[info[3]]
                left_child_node.start_idx = parent_node.start_idx
                left_child_node.end_idx = parent_node.start_idx + len(left_child_node.content_id)
                right_child_node.start_idx = left_child_node.end_idx
                right_child_node.end_idx = parent_node.end_idx
                if not left_child_node.is_leaf:
                    correct_node_list.append(
                        (left_child_node.start_idx,
                         left_child_node.end_idx,
                         left_child_node.category_id))
                if not right_child_node.is_leaf:
                    correct_node_list.append(
                        (right_child_node.start_idx,
                         right_child_node.end_idx,
                         right_child_node.category_id))
        return correct_node_list


class Tree_List:
    def __init__(
            self,
            PATH_TO_DATA,
            content_vocab,
            category_vocab=None,
            device=torch.device('cpu')):
        self.content_vocab = content_vocab
        self.category_vocab = category_vocab
        self.device = device
        self.set_tree_list(PATH_TO_DATA)
        if category_vocab is None:
            self.make_category_vocab()
        self.set_content_category_id(self.content_vocab, self.category_vocab)

    def set_tree_list(self, PATH_TO_DATA):
        self.tree_list = []
        tree_id = 0
        node_list = []
        with open(PATH_TO_DATA, 'r') as f:
            node_info_list = [node_info.strip() for node_info in f.readlines()]
        node_info_list = [node_info.replace(
            '\n', '') for node_info in node_info_list]
        for node_info in node_info_list:
            if node_info != '':
                node = Node(node_info.split())
                node_list.append(node)
            elif node_list != []:
                self.tree_list.append(Tree(tree_id, node_list))
                node_list = []
                tree_id += 1

    def make_category_vocab(self):
        category_counter = Counter()
        for tree in self.tree_list:
            for node in tree.node_list:
                category_counter[node.category] += 1
        self.category_vocab = Vocab(category_counter, specials=['<unk>'])

    def set_content_category_id(self, content_vocab, category_vocab):
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    node.content_id = [content_vocab[node.content]]
                node.category_id = category_vocab[node.category]
            tree.set_node_composition_info()

    def set_possible_category_id(self, possible_category_dict):
        for tree in self.tree_list:
            for node in tree.node_list:
                key = tuple(node.content_id)
                if key not in possible_category_dict:
                    possible_category_dict[key] = [node.category_id]
                elif node.category_id not in possible_category_dict[key]:
                    possible_category_dict[key].append(node.category_id)

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
        self.num_node = []
        self.leaf_node_content_id = []
        self.label_list = []
        self.composition_info = []
        for tree in self.tree_list:
            self.num_node.append(len(tree.node_list))
            leaf_node_content_id = []
            label_list = []
            for node in tree.node_list:
                if node.is_leaf:
                    # save the index of leaf node and its content
                    leaf_node_content_id.append(
                        [node.self_id, node.content_id[0]])
                    # label with multiple bit corresponding to possible category id
                label_list.append([node.category_id])
            self.leaf_node_content_id.append(
                torch.tensor(
                    leaf_node_content_id,
                    dtype=torch.long,
                    device=self.device))
            self.label_list.append(label_list)
            self.composition_info.append(
                torch.tensor(
                    tree.composition_info,
                    dtype=torch.long,
                    device=self.device))

    def make_batch(self, BATCH_SIZE=None):
        # make batch content id includes leaf node content id for each tree belongs to batch
        batch_num_node = []
        batch_leaf_content_id = []
        batch_label_list = []
        batch_composition_info = []
        num_tree = len(self.tree_list)

        if BATCH_SIZE is None:
            batch_tree_id_list = list(range(num_tree))
            batch_num_node.append(
                list(itemgetter(*batch_tree_id_list)(self.num_node)))
            batch_leaf_content_id.append(list(itemgetter(
                *batch_tree_id_list)(self.leaf_node_content_id)))
            batch_label_list.append(list(itemgetter(
                *batch_tree_id_list)(self.label_list)))
            batch_composition_info.append(list(itemgetter(
                *batch_tree_id_list)(self.composition_info)))
        else:
            # shuffle the tree_id in tree_list
            shuffled_tree_id = torch.randperm(num_tree, device=self.device)
            for idx in range(0, num_tree - BATCH_SIZE, BATCH_SIZE):
                batch_tree_id_list = shuffled_tree_id[idx:idx + BATCH_SIZE]
                batch_num_node.append(
                    list(itemgetter(*batch_tree_id_list)(self.num_node)))
                batch_leaf_content_id.append(list(itemgetter(
                    *batch_tree_id_list)(self.leaf_node_content_id)))
                batch_label_list.append(list(itemgetter(
                    *batch_tree_id_list)(self.label_list)))
                batch_composition_info.append(list(itemgetter(
                    *batch_tree_id_list)(self.composition_info)))
            # the part cannot devided by BATCH_SIZE
            batch_num_node.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.num_node)))
            batch_leaf_content_id.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.leaf_node_content_id)))
            batch_label_list.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.label_list)))
            batch_composition_info.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.composition_info)))

        content_mask = []
        for idx in range(len(batch_num_node)):
            content_id = batch_leaf_content_id[idx]
            composition_list = batch_composition_info[idx]

            max_num_leaf_node = max([len(i) for i in content_id])
            # set the mask for each tree in batch
            # content_mask used for embedding leaf node vector
            true_mask = [torch.ones(len(i), dtype=torch.bool, device=self.device)
                         for i in content_id]
            false_mask = [
                torch.zeros(
                    max_num_leaf_node - len(i),
                    dtype=torch.bool,
                    device=self.device) for i in content_id]
            content_mask.append(torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(true_mask, false_mask)]))
            # make dummy content id to fill blank in batch
            dummy_content_id = [
                torch.zeros(
                    (max_num_leaf_node - len(i), 2),
                    dtype=torch.long,
                    device=self.device) for i in content_id]
            batch_leaf_content_id[idx] = torch.stack([torch.cat((i, j)) for (
                i, j) in zip(content_id, dummy_content_id)])

            # set mask for composition info in each batch
            max_num_composition = max([len(i) for i in composition_list])
            # make dummy compoisition info to fill blank in batch
            dummy_compositin_info = [
                torch.ones(
                    max_num_composition - len(i),
                    4,
                    dtype=torch.long,
                    device=self.device) * -1 for i in composition_list]
            batch_composition_info[idx] = torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(composition_list, dummy_compositin_info)])

        # return zipped batch information, when training, extract each batch from zip itteration
        return list(zip(
            batch_num_node,
            batch_leaf_content_id,
            content_mask,
            batch_composition_info,
            batch_label_list))

    def set_vector(self, embedding):
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    node.vector = embedding(torch.tensor(node.content_id))
                    node.vector = node.vector / torch.norm(node.vector)

        for tree in self.tree_list:
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

    def clean_tree_list(self, exist_content_id, exist_category_id):
        cleaned_tree_list = self.tree_list
        for tree in self.tree_list:
            check_bit = 0
            for node in tree.node_list:
                if node.is_leaf:
                    if node.content_id[0] not in exist_content_id or node.category_id not in exist_category_id:
                        # when content or category of node not in train_tree
                        check_bit = 1
                else:
                    if node.category_id not in exist_category_id:
                        # when category of node not in train_tree
                        check_bit = 1
                if check_bit == 1:
                    cleaned_tree_list.remove(tree)
                    break
        self.tree_list = cleaned_tree_list


class Tree_Net(nn.Module):
    def __init__(self, NUM_VOCAB, NUM_CATEGORY, embedding_dim, initial_weight_matrix=None):
        super(Tree_Net, self).__init__()
        self.num_embedding = NUM_VOCAB
        self.num_category = NUM_CATEGORY
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

    # input batch as tuple of training info
    def forward(self, batch):
        # the content_id of leaf nodes
        num_node = batch[0]
        leaf_content_id = batch[1]
        content_mask = batch[2]
        # the composition info of each tree
        composition_info = batch[3]
        vector = self.embed_leaf_nodes(num_node, leaf_content_id, content_mask)
        vector = self.compose(vector, composition_info)
        output = self.linear(vector)
        return output

    def embed_leaf_nodes(self, num_node, leaf_content_id, content_mask):
        vector = torch.zeros(
            (leaf_content_id.shape[0],
             torch.tensor(max(num_node)),
             self.embedding_dim), device=content_mask.device)
        # leaf_node_vector including padding tokens
        leaf_node_index = leaf_content_id[:, :, 0]
        leaf_node_vector = self.embedding(leaf_content_id[:, :, 1])
        # extract leaf node vector not padding tokens, using content_mask
        vector[(content_mask.nonzero(as_tuple=True)[0], leaf_node_index[content_mask.nonzero(
            as_tuple=True)])] = leaf_node_vector[content_mask.nonzero(as_tuple=True)]
        # calculate norm for normalization
        norm = vector.norm(dim=2, keepdim=True) + 1e-6
        return vector / norm

    def compose(self, vector, composition_info):
        # itteration of composition
        for idx in range(composition_info.shape[1]):
            # the positional index where the composition info of one child is located in batch
            one_child_compositino_idx = torch.squeeze(
                torch.nonzero(composition_info[:, idx, 0] == 1))
            one_child_composition_info = composition_info[composition_info[:, idx, 0] == 1][:, idx]
            one_child_parent_idx = one_child_composition_info[:, 1]
            # the child node index of one child composition
            child_idx = one_child_composition_info[:, 2]
            child_vector = vector[(one_child_compositino_idx, child_idx)]
            vector[(one_child_compositino_idx, one_child_parent_idx)] = child_vector
            two_child_composition_idx = torch.squeeze(
                torch.nonzero(composition_info[:, idx, 0] == 2))
            two_child_composition_info = composition_info[composition_info[:, idx, 0] == 2][:, idx]
            if len(two_child_composition_info) != 0:
                two_child_parent_idx = two_child_composition_info[:, 1]
                # left child node index of two child composition
                left_child_idx = two_child_composition_info[:, 2]
                right_child_idx = two_child_composition_info[:, 3]
                left_child_vector = vector[(two_child_composition_idx, left_child_idx)]
                right_child_vector = vector[(two_child_composition_idx, right_child_idx)]
                composed_vector = circular_correlation(left_child_vector, right_child_vector)
                vector[(two_child_composition_idx, two_child_parent_idx)] = composed_vector
        return vector

    def evaluate(self, tree_list,unk_idx):
        embedding = self.embedding
        linear = self.linear
        tree_list.set_vector(embedding)

        # calculate top-1 and top-5 accuracy
        for k in [1, 5]:
            num_correct_word = 0
            num_word = 0
            num_correct_phrase = 0
            num_phrase = 0
            for tree in tree_list.tree_list:
                for node in tree.node_list:
                    if node.category_id != unk_idx:
                        output = linear(node.vector)
                        predict = torch.topk(output, k=k)[1]
                        if node.is_leaf:
                            if node.category_id in predict:
                                num_correct_word += 1
                            num_word += 1
                        else:
                            if node.category_id in predict:
                                num_correct_phrase += 1
                            num_phrase += 1
            print('-' * 50)
            print('overall top-{}: {}'.format(k, (num_correct_word +
                                                  num_correct_phrase) / (num_word + num_phrase)))
            print('word top-{}: {}'.format(k, num_correct_word / num_word))
            print('phrase top-{}: {}'.format(k, num_correct_phrase / num_phrase))
