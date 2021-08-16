import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from operator import itemgetter
from utils import circular_correlation, single_circular_correlation
from allennlp.modules.elmo import batch_to_ids


class Node:
    def __init__(self, node_info):
        if node_info[0] == 'True':
            self.is_leaf = True
        else:
            self.is_leaf = False
        self.self_id = int(node_info[1])
        if self.is_leaf:
            content = node_info[2]
            self.content = [self.convert_content(content)]
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

    def convert_content(self, content):
        if content == "-LRB-" or content == "-LCB-":
            content = "("
        elif content == "-RRB-" or content == "-RCB-":
            content = ")"
        elif r"\/" in content:
            content = content.replace(r"\/", "/")
        return content


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
                            node.content = child_node.content
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, child_node.self_id, 0])
                    else:  # when node has two children
                        left_child_node = self.node_list[node.left_child_node_id]
                        right_child_node = self.node_list[node.right_child_node_id]
                        if left_child_node.ready and right_child_node.ready:
                            node.content = left_child_node.content + right_child_node.content
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, left_child_node.self_id, right_child_node.self_id])
            if num_ready_node == len(self.node_list):
                break
        self.sentence = self.node_list[-1].content

    def set_original_position_of_leaf_node(self):
        self.original_pos = []
        node = self.node_list[-1]
        if node.is_leaf:
            node.original_pos = 0
            self.original_pos.append([node.self_id, node.original_pos])
        else:
            node.start_idx = 0
            node.end_idx = len(node.content)
        for info in reversed(self.composition_info):
            num_child = info[0]
            if num_child == 1:
                parent_node = self.node_list[info[1]]
                child_node = self.node_list[info[2]]
                child_node.start_idx = parent_node.start_idx
                child_node.end_idx = parent_node.end_idx
                if child_node.is_leaf:
                    child_node.original_pos = child_node.start_idx
                    self.original_pos.append([child_node.self_id, child_node.original_pos])

            else:
                parent_node = self.node_list[info[1]]
                left_child_node = self.node_list[info[2]]
                right_child_node = self.node_list[info[3]]
                left_child_node.start_idx = parent_node.start_idx
                left_child_node.end_idx = parent_node.start_idx + len(left_child_node.content)
                right_child_node.start_idx = left_child_node.end_idx
                right_child_node.end_idx = parent_node.end_idx
                if left_child_node.is_leaf:
                    left_child_node.original_pos = left_child_node.start_idx
                    self.original_pos.append(
                        [left_child_node.self_id, left_child_node.original_pos])
                if right_child_node.is_leaf:
                    right_child_node.original_pos = right_child_node.start_idx
                    self.original_pos.append(
                        [right_child_node.self_id, right_child_node.original_pos])

    def correct_parse(self):
        correct_node_list = []
        top_node = self.node_list[-1]
        top_node.start_idx = 0
        top_node.end_idx = len(top_node.content)
        if not top_node.is_leaf:
            correct_node_list.append((1, len(top_node.content) + 1, top_node.category_id + 1))
        for info in reversed(self.composition_info):
            num_child = info[0]
            if num_child == 1:
                parent_node = self.node_list[info[1]]
                child_node = self.node_list[info[2]]
                child_node.start_idx = parent_node.start_idx
                child_node.end_idx = parent_node.end_idx
                if not child_node.is_leaf:
                    correct_node_list.append(
                        (child_node.start_idx + 1,
                         child_node.end_idx + 1,
                         child_node.category_id + 1))
            else:
                parent_node = self.node_list[info[1]]
                left_child_node = self.node_list[info[2]]
                right_child_node = self.node_list[info[3]]
                left_child_node.start_idx = parent_node.start_idx
                left_child_node.end_idx = parent_node.start_idx + len(left_child_node.content)
                right_child_node.start_idx = left_child_node.end_idx
                right_child_node.end_idx = parent_node.end_idx
                if not left_child_node.is_leaf:
                    correct_node_list.append(
                        (left_child_node.start_idx + 1,
                         left_child_node.end_idx + 1,
                         left_child_node.category_id + 1))
                if not right_child_node.is_leaf:
                    correct_node_list.append(
                        (right_child_node.start_idx + 1,
                         right_child_node.end_idx + 1,
                         right_child_node.category_id + 1))
        return correct_node_list


class Tree_List:
    def __init__(
            self,
            PATH_TO_DATA,
            category_vocab=None,
            device=torch.device('cpu')):
        self.category_vocab = category_vocab
        self.device = device
        self.set_tree_list(PATH_TO_DATA)
        self.set_category_id(self.category_vocab)

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

    def set_category_id(self, category_vocab):
        for tree in self.tree_list:
            for node in tree.node_list:
                node.category_id = category_vocab[node.category]
            tree.set_node_composition_info()
            tree.set_original_position_of_leaf_node()

    def set_info_for_training(self):
        self.num_node = []
        self.sentence_list = []
        self.label_list = []
        self.original_pos = []
        self.composition_info = []
        for tree in self.tree_list:
            self.num_node.append(len(tree.node_list))
            self.sentence_list.append(tree.sentence)
            label_list = []
            for node in tree.node_list:
                label_list.append([node.category_id])
            self.label_list.append(label_list)
            self.original_pos.append(
                torch.tensor(
                    tree.original_pos,
                    dtype=torch.long,
                    device=self.device))
            self.composition_info.append(
                torch.tensor(
                    tree.composition_info,
                    dtype=torch.long,
                    device=self.device))

    def make_batch(self, BATCH_SIZE=None):
        # make batch content id includes leaf node content id for each tree belongs to batch
        batch_num_node = []
        batch_sentence_list = []
        batch_label_list = []
        batch_original_pos = []
        batch_composition_info = []
        num_tree = len(self.tree_list)

        if BATCH_SIZE is None:
            batch_tree_id_list = list(range(num_tree))
            batch_num_node.append(
                list(itemgetter(*batch_tree_id_list)(self.num_node)))
            batch_sentence_list.append(list(itemgetter(*batch_tree_id_list)(self.sentence_list)))
            batch_label_list.append(list(itemgetter(
                *batch_tree_id_list)(self.label_list)))
            batch_original_pos.append(list(itemgetter(*batch_tree_id_list)(self.original_pos)))
            batch_composition_info.append(list(itemgetter(
                *batch_tree_id_list)(self.composition_info)))
        else:
            # shuffle the tree_id in tree_list
            shuffled_tree_id = torch.randperm(num_tree, device=self.device)
            for idx in range(0, num_tree - BATCH_SIZE, BATCH_SIZE):
                batch_tree_id_list = shuffled_tree_id[idx:idx + BATCH_SIZE]
                batch_num_node.append(
                    list(itemgetter(*batch_tree_id_list)(self.num_node)))
                batch_sentence_list.append(
                    list(
                        itemgetter(
                            *
                            batch_tree_id_list)(
                            self.sentence_list)))
                batch_label_list.append(list(itemgetter(
                    *batch_tree_id_list)(self.label_list)))
                batch_original_pos.append(list(itemgetter(*batch_tree_id_list)(self.original_pos)))
                batch_composition_info.append(list(itemgetter(
                    *batch_tree_id_list)(self.composition_info)))
            # the part cannot devided by BATCH_SIZE
            batch_num_node.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.num_node)))
            batch_sentence_list.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(self.sentence_list)))
            batch_label_list.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.label_list)))
            batch_original_pos.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(self.original_pos)))
            batch_composition_info.append(list(itemgetter(
                *shuffled_tree_id[idx + BATCH_SIZE:])(self.composition_info)))

        for idx in range(len(batch_num_node)):
            composition_list = batch_composition_info[idx]
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
            batch_sentence_list,
            batch_original_pos,
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


class Tree_Net(nn.Module):
    def __init__(self, NUM_CATEGORY, elmo, embedding_dim=1024):
        super(Tree_Net, self).__init__()
        self.num_category = NUM_CATEGORY
        self.embedding_dim = embedding_dim
        self.elmo = elmo
        self.hidden_dim = 512
        self.lstm = nn.LSTM(
            input_size=self.embedding_dim,
            hidden_size=self.hidden_dim,
            num_layers=1,
            batch_first=True,
            bidirectional=True)
        self.W1 = torch.randn(self.hidden_dim, self.hidden_dim, requires_grad=True)
        self.W2 = torch.randn(self.hidden_dim, self.hidden_dim, requires_grad=True)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(self.hidden_dim, self.num_category)

    # input batch as tuple of training info
    def forward(self, batch):
        num_node = batch[0]
        sentence = batch[1]
        original_pos = batch[2]
        composition_info = batch[3]
        packed_sequence = self.elmo_embedding(sentence)
        combined_rep, len_unpacked = self.combine_foward_backward(packed_sequence)
        vector = self.set_leaf_node_vector(num_node, combined_rep, len_unpacked, original_pos)
        vector = self.compose(vector, composition_info)
        output = self.linear(vector)
        return output

    def elmo_embedding(self, sentence):
        input = batch_to_ids(sentence)
        output = self.elmo(input)
        rep = output['elmo_representations'][0]
        mask = output['mask']
        packed_sequence = pack_padded_sequence(rep, torch.count_nonzero(
            mask, dim=1), batch_first=True, enforce_sorted=False)
        return packed_sequence

    # combine the output of bidirectional LSTM, the output of foward and backward LSTM
    def combine_foward_backward(self, packed_sequence):
        seq_unapcked, len_unpacked = pad_packed_sequence(packed_sequence, batch_first=True)
        combined_rep = self.relu(torch.matmul(
            seq_unapcked[:, :, :self.hidden_dim], self.W1) + torch.matmul(seq_unapcked[:, :, self.hidden_dim:], self.W2))
        return combined_rep, len_unpacked

    def set_leaf_node_vector(self, num_node, combined_rep, len_unpacked, original_pos):
        vector = torch.zeros(
            (len(num_node),
             torch.tensor(max(num_node)),
             self.hidden_dim), device=combined_rep.device)
        for idx in range(len(num_node)):
            batch_id = torch.tensor([idx for i in range(len_unpacked[idx])])
            # target_id is node.self_id
            target_id = torch.squeeze(original_pos[idx][:, 0])
            # source_id is node.original_pos
            source_id = torch.squeeze(original_pos[idx][:, 1])
            vector[(batch_id, target_id)] = combined_rep[(batch_id, source_id)]
        return vector / (vector.norm(dim=2, keepdim=True) + 1e-6)

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
