import random
from torch.nn.init import kaiming_uniform_
from tqdm import tqdm
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from operator import itemgetter
from utils import circular_correlation, single_circular_correlation, standardize


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
                self.head = int(node_info[5])
            else:
                self.left_child_node_id = int(node_info[4])
                self.right_child_node_id = int(node_info[5])
                self.head = int(node_info[6])

    def convert_content(self, content):
        if content == "-LRB-":
            content = "("
        elif content == "-LCB-":
            content = "{"
        elif content == "-RRB-":
            content = ")"
        elif content == "-RCB-":
            content = "}"
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
        self.spans = []
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
                self.spans.append([parent_node.start_idx, parent_node.end_idx])
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

    def correct_parse(self, whole_category_vocab):
        correct_node_list = []
        top_node = self.node_list[-1]
        top_node.start_idx = 0
        top_node.end_idx = len(top_node.content)
        if not top_node.is_leaf:
            # for unk category
            if whole_category_vocab[top_node.category] == 0:
                correct_node_list.append((1, len(top_node.content) + 1, -1))
            else:
                correct_node_list.append(
                    (1, len(top_node.content) + 1, whole_category_vocab[top_node.category] + 1))
        for info in reversed(self.composition_info):
            num_child = info[0]
            if num_child == 1:
                parent_node = self.node_list[info[1]]
                child_node = self.node_list[info[2]]
                child_node.start_idx = parent_node.start_idx
                child_node.end_idx = parent_node.end_idx
                if not child_node.is_leaf:
                    # for unk category
                    if whole_category_vocab[child_node.category] == 0:
                        correct_node_list.append(
                            (child_node.start_idx + 1,
                             child_node.end_idx + 1,
                             -1))
                    else:
                        correct_node_list.append(
                            (child_node.start_idx + 1,
                             child_node.end_idx + 1,
                             whole_category_vocab[child_node.category] + 1))
            else:
                parent_node = self.node_list[info[1]]
                left_child_node = self.node_list[info[2]]
                right_child_node = self.node_list[info[3]]
                left_child_node.start_idx = parent_node.start_idx
                left_child_node.end_idx = parent_node.start_idx + len(left_child_node.content)
                right_child_node.start_idx = left_child_node.end_idx
                right_child_node.end_idx = parent_node.end_idx
                if not left_child_node.is_leaf:
                    # for unk category
                    if whole_category_vocab[left_child_node.category] == 0:
                        correct_node_list.append(
                            (left_child_node.start_idx + 1,
                             left_child_node.end_idx + 1,
                             -1))
                    else:
                        correct_node_list.append(
                            (left_child_node.start_idx + 1,
                             left_child_node.end_idx + 1,
                             whole_category_vocab[left_child_node.category] + 1))
                if not right_child_node.is_leaf:
                    # for unk category
                    if whole_category_vocab[right_child_node.category] == 0:
                        correct_node_list.append(
                            (right_child_node.start_idx + 1,
                             right_child_node.end_idx + 1,
                             -1))
                    else:
                        correct_node_list.append(
                            (right_child_node.start_idx + 1,
                             right_child_node.end_idx + 1,
                             whole_category_vocab[right_child_node.category] + 1))
        return correct_node_list

    # generate the random binary tree in order to obtain negative training sample for span scoring
    def generate_random_tree(self):
        random_composition_info = []
        random_original_pos = []
        # list of span's id which do not exist in gold tree
        negative_node_id = []

        node_id = 0
        node = [0, len(self.sentence), node_id]
        node_list = [node]

        if len(self.sentence) > 1:
            if node[:2] not in self.spans:
                negative_node_id.append(node_id)
            wait_list = [node]
        else:
            wait_list = []
            random_original_pos = [[0, 0]]

        node_id += 1

        while True:
            if wait_list == []:
                break
            # information about parent node which is split into two child nodes
            parent_node = wait_list.pop(0)
            start_idx = parent_node[0]
            end_idx = parent_node[1]
            parent_id = parent_node[2]

            # decide split point
            split_idx = random.randint(start_idx + 1, end_idx - 1)

            # define left child node
            left_node = [start_idx, split_idx, node_id]
            node_list.append(left_node)
            # when left node is not leaf node
            if left_node[1] - left_node[0] > 1:
                if left_node[:2] not in self.spans:
                    negative_node_id.append(node_id)
                wait_list.append(left_node)
            # when left node is leaf node
            else:
                random_original_pos.append([node_id, split_idx - 1])
            node_id += 1

            # define right child node
            right_node = [split_idx, end_idx, node_id]
            node_list.append(right_node)
            # when right node is not leaf node
            if right_node[1] - right_node[0] > 1:
                if right_node[:2] not in self.spans:
                    negative_node_id.append(node_id)
                wait_list.append(right_node)
            else:
                random_original_pos.append([node_id, end_idx - 1])
            node_id += 1

            random_composition_info.append([2, parent_id, node_id - 2, node_id - 1])
        random_composition_info.reverse()
        return len(node_list), random_composition_info, random_original_pos, negative_node_id

    def set_word_split(self, tokenizer):
        sentence = " ".join(self.sentence)
        tokens = tokenizer.tokenize(sentence)
        tokenized_pos = 0
        word_split = []
        for original_pos in range(len(self.sentence)):
            word = self.sentence[original_pos]
            length = 1
            while True:
                temp = tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length])
                temp = temp.replace(" ", "")
                temp = temp.replace("\"", "``")
                if word == temp or word.lower() == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        self.word_split = word_split
        return word_split


class Tree_List:
    def __init__(
            self,
            PATH_TO_DATA,
            word_category_vocab,
            phrase_category_vocab,
            device=torch.device('cpu')):
        self.word_category_vocab = word_category_vocab
        self.phrase_category_vocab = phrase_category_vocab
        self.device = device
        self.set_tree_list(PATH_TO_DATA)
        self.set_category_id(self.word_category_vocab, self.phrase_category_vocab)

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

    def set_category_id(self, word_category_vocab, phrase_category_vocab):
        for tree in self.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    node.category_id = word_category_vocab[node.category]
                else:
                    node.category_id = phrase_category_vocab[node.category]
            tree.set_node_composition_info()
            tree.set_original_position_of_leaf_node()

    def set_info_for_training(self, tokenizer=None):
        self.num_node = []
        self.sentence_list = []
        self.label_list = []
        self.original_pos = []
        self.composition_info = []
        self.word_split = []
        for tree in self.tree_list:
            self.num_node.append(len(tree.node_list))
            self.sentence_list.append(" ".join(tree.sentence))
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
            self.word_split.append(tree.set_word_split(tokenizer))
        self.sorted_tree_id = np.argsort(self.num_node)

    def make_shuffled_tree_id(self):
        shuffled_tree_id = []
        splitted = np.array_split(self.sorted_tree_id, 50)
        for id_list in splitted:
            np.random.shuffle(id_list)
            shuffled_tree_id.append(id_list)
        return np.concatenate(shuffled_tree_id)

    def make_batch(self, BATCH_SIZE=None):
        # make batch content id includes leaf node content id for each tree belongs to batch
        batch_num_node = []
        batch_sentence_list = []
        batch_label_list = []
        batch_original_pos = []
        batch_composition_info = []
        batch_word_split = []
        num_tree = len(self.tree_list)

        # the series of "random" are information about randomly generated tree for
        # the scoring of span
        random_num_node = []
        random_original_pos = []
        random_composition_info = []
        random_negative_node_id = []
        batch_random_num_node = []
        batch_random_composition_info = []
        batch_random_original_pos = []
        batch_random_negative_node_id = []
        # generate random binary tree for all sentence in training data each epoch
        for tree in self.tree_list:
            random_tree_info = tree.generate_random_tree()
            random_num_node.append(random_tree_info[0])
            random_composition_info.append(
                torch.tensor(
                    random_tree_info[1],
                    dtype=torch.long,
                    device=self.device))
            random_original_pos.append(
                torch.tensor(
                    random_tree_info[2],
                    dtype=torch.long,
                    device=self.device))
            random_negative_node_id.append(
                torch.tensor(
                    random_tree_info[3],
                    dtype=torch.long,
                    device=self.device))
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
            batch_word_split.append(list(itemgetter(
                *batch_tree_id_list)(self.word_split)))
            batch_random_num_node.append(list(itemgetter(*batch_tree_id_list)(random_num_node)))
            batch_random_composition_info.append(
                list(itemgetter(*batch_tree_id_list)(random_composition_info)))
            batch_random_original_pos.append(
                list(itemgetter(*batch_tree_id_list)(random_original_pos)))
            batch_random_negative_node_id.append(
                list(itemgetter(*batch_tree_id_list)(random_negative_node_id)))
        else:
            # shuffle the tree_id in tree_list
            shuffled_tree_id = self.make_shuffled_tree_id()
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
                batch_word_split.append(list(itemgetter(
                    *batch_tree_id_list)(self.word_split)))
                batch_random_num_node.append(list(itemgetter(*batch_tree_id_list)(random_num_node)))
                batch_random_composition_info.append(
                    list(itemgetter(*batch_tree_id_list)(random_composition_info)))
                batch_random_original_pos.append(
                    list(itemgetter(*batch_tree_id_list)(random_original_pos)))
                batch_random_negative_node_id.append(
                    list(itemgetter(*batch_tree_id_list)(random_negative_node_id)))
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
            batch_word_split.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(self.word_split)))
            batch_random_num_node.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(random_num_node)))
            batch_random_composition_info.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(random_composition_info)))
            batch_random_original_pos.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(random_original_pos)))
            batch_random_negative_node_id.append(
                list(itemgetter(*shuffled_tree_id[idx + BATCH_SIZE:])(random_negative_node_id)))

        for idx in range(len(batch_composition_info)):
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

        for idx in range(len(batch_random_composition_info)):
            composition_list = batch_random_composition_info[idx]
            # set mask for composition info in each batch
            max_num_composition = max([len(i) for i in composition_list])
            # make dummy compoisition info to fill blank in batch
            dummy_compositin_info = [
                torch.ones(
                    max_num_composition - len(i),
                    4,
                    dtype=torch.long,
                    device=self.device) * -1 for i in composition_list]
            batch_random_composition_info[idx] = torch.stack(
                [torch.cat((i, j)) for (i, j) in zip(composition_list, dummy_compositin_info)])
        # return zipped batch information, when training, extract each batch from zip itteration
        batch_list = list(zip(
            batch_num_node,
            batch_sentence_list,
            batch_original_pos,
            batch_composition_info,
            batch_label_list,
            batch_word_split,
            batch_random_num_node,
            batch_random_composition_info,
            batch_random_original_pos,
            batch_random_negative_node_id))
        np.random.shuffle(batch_list)
        return batch_list

    def set_vector(self, tree_net):
        with tqdm(total=len(self.tree_list)) as pbar:
            pbar.set_description("setting vector...")
            for tree in self.tree_list:
                sentence = [" ".join(tree.sentence)]
                word_split = [tree.word_split]
                vector_list = tree_net.embed(sentence, word_split=word_split)
                for pos in tree.original_pos:
                    node_id = pos[0]
                    original_pos = pos[1]
                    node = tree.node_list[node_id]
                    node.vector = torch.squeeze(vector_list[original_pos])
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
                pbar.update(1)


class Tree_Net(nn.Module):
    def __init__(
            self,
            num_word_cat,
            num_phrase_cat,
            model,
            tokenizer=None,
            learn_embedder=True,
            embedding_dim=1024,
            model_dim=300,
            ff_dropout=0.2,
            device=torch.device('cpu')):
        super(Tree_Net, self).__init__()
        self.num_word_cat = num_word_cat
        self.num_phrase_cat = num_phrase_cat
        self.model = model
        self.tokenizer = tokenizer
        self.embedding_dim = embedding_dim
        self.model_dim = model_dim
        self.learn_embedder = learn_embedder
        # the list which to record the modules to set separated lr
        self.base_modules = []
        self.base_params = []

        self.transform_word_rep = FeedForward(
            self.embedding_dim, self.embedding_dim, self.model_dim)
        self.word_ff = FeedForward(
            self.model_dim,
            self.model_dim,
            self.num_word_cat,
            dropout=ff_dropout)
        self.phrase_ff = FeedForward(
            self.model_dim,
            self.model_dim,
            self.num_phrase_cat,
            dropout=ff_dropout)
        self.span_ff = FeedForward(self.model_dim, self.model_dim, 1, dropout=ff_dropout)
        self.base_modules.append(self.transform_word_rep)
        self.base_modules.append(self.word_ff)
        self.base_modules.append(self.phrase_ff)
        self.base_modules.append(self.span_ff)
        for module in self.base_modules:
            for params in module.parameters():
                self.base_params.append(params)
        self.base_params = iter(self.base_params)
        self.device = device

    # input batch as tuple of training info
    def forward(self, batch):
        num_node = batch[0]
        sentence = batch[1]
        original_pos = batch[2]
        composition_info = batch[3]
        batch_label = batch[4]
        word_split = batch[5]
        random_num_node = batch[6]
        random_composition_info = batch[7]
        random_original_pos = batch[8]
        random_negative_node_id = batch[9]

        if self.learn_embedder:
            vector_list, lengths = self.embed(sentence, word_split)
        # when not train word embedder, the computation of gradient is not needed
        else:
            with torch.no_grad():
                vector_list, lengths = self.embed(sentence, word_split)

        # compose word vectors and fed them into FFNN
        original_vector = self.set_leaf_node_vector(
            num_node, vector_list, lengths, original_pos)
        # compose word vectors for randomly generated trees
        random_vector = self.set_leaf_node_vector(
            random_num_node, vector_list, lengths, random_original_pos)
        original_vector_shape = original_vector.shape
        random_vector_shape = random_vector.shape
        original_vector = original_vector.view(-1, self.embedding_dim)
        random_vector = random_vector.view(-1, self.embedding_dim)
        vector = torch.cat((original_vector, random_vector))
        vector = standardize(self.transform_word_rep(vector))
        original_vector = vector[:original_vector_shape[0] * original_vector_shape[1],
                                 :].view(original_vector_shape[0], original_vector_shape[1], self.model_dim)
        random_vector = vector[original_vector_shape[0] * original_vector_shape[1]:, :].view(random_vector_shape[0], random_vector_shape[1], self.model_dim)
        composed_vector = self.compose(original_vector, composition_info)
        random_composed_vector = self.compose(random_vector, random_composition_info)
        word_vector, phrase_vector, word_label, phrase_label = self.devide_word_phrase(
            composed_vector, batch_label, original_pos)
        span_vector, span_label = self.extract_span_vector(
            phrase_vector, random_composed_vector, random_negative_node_id)

        word_output = self.word_ff(word_vector)
        phrase_output = self.phrase_ff(phrase_vector)
        span_output = self.span_ff(span_vector)
        return word_output, phrase_output, span_output, word_label, phrase_label, span_label

    # embedding word vector
    def embed(self, sentence, word_split):
        input = self.tokenizer(
            sentence,
            padding=True,
            return_tensors='pt').to(self.device)
        output = self.model(**input).last_hidden_state[:, 1:-1]
        vector_list = []
        lengths = []
        for vector, info in zip(output, word_split):
            temp = []
            for start_idx, end_idx in info:
                temp.append(torch.mean(vector[start_idx:end_idx], dim=0))
            vector_list.append(torch.stack(temp))
            lengths.append(len(temp))
        vector_list = pad_sequence(vector_list, batch_first=True)
        lengths = torch.tensor(lengths, device=torch.device('cpu'))
        return vector_list, lengths

    def set_leaf_node_vector(self, num_node, vector_list, lengths, original_pos):
        leaf_node_vector = torch.zeros(
            (len(num_node),
             torch.tensor(max(num_node)),
             self.embedding_dim), device=self.device)
        for idx in range(len(num_node)):
            batch_id = torch.tensor([idx for _ in range(lengths[idx])])
            # target_id is node.self_id
            target_id = torch.squeeze(original_pos[idx][:, 0])
            # source_id is node.original_pos
            source_id = torch.squeeze(original_pos[idx][:, 1])
            leaf_node_vector[(batch_id, target_id)] = vector_list[(batch_id, source_id)]
        return leaf_node_vector

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

    def devide_word_phrase(self, vector, batch_label, original_pos):
        word_vector = []
        phrase_vector = []
        word_label = []
        phrase_label = []
        for i in range(vector.shape[0]):
            word_idx = torch.zeros(len(batch_label[i]), dtype=torch.bool, device=self.device)
            word_idx[original_pos[i][:, 0]] = True
            phrase_idx = torch.logical_not(word_idx)
            word_vector.append(vector[i, :len(batch_label[i])][word_idx])
            phrase_vector.append(vector[i, :len(batch_label[i])][phrase_idx])
            word_label.append(
                torch.tensor(
                    batch_label[i],
                    dtype=torch.long,
                    device=self.device)[word_idx])
            phrase_label.append(
                torch.tensor(
                    batch_label[i],
                    dtype=torch.long,
                    device=self.device)[phrase_idx])
        word_vector = torch.cat(word_vector)
        phrase_vector = torch.cat(phrase_vector)
        word_label = torch.squeeze(torch.vstack(word_label))
        phrase_label = torch.squeeze(torch.vstack(phrase_label))
        return word_vector, phrase_vector, word_label, phrase_label

    def extract_span_vector(self, phrase_vector, random_composed_vector, random_negative_node_id):
        # the label for gold spans
        positive_label = torch.ones(phrase_vector.shape[0], dtype=torch.float, device=self.device)

        # the list to contain vectors for negative spans
        negative_span_vector = []
        for i in range(random_composed_vector.shape[0]):
            negative_span_idx = random_negative_node_id[i]
            if negative_span_idx.shape[0] > 0:
                negative_span_vector.append(
                    torch.index_select(
                        random_composed_vector[i],
                        0,
                        negative_span_idx))
        negative_span_vector = torch.cat(negative_span_vector)
        negative_label = torch.zeros(
            negative_span_vector.shape[0],
            dtype=torch.float,
            device=self.device)

        span_vector = torch.cat([phrase_vector, negative_span_vector])
        span_label = torch.cat([positive_label, negative_label]).view(-1, 1)

        return span_vector, span_label

    def set_word_split(self, sentence):
        tokenizer = self.tokenizer
        sentence = " ".join(sentence)
        tokens = tokenizer.tokenize(sentence)
        tokenized_pos = 0
        word_split = []
        for original_pos in range(len(sentence.split())):
            word = sentence.split()[original_pos]
            length = 1
            while True:
                temp = tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length])
                temp = temp.replace(" ", "")
                temp = temp.replace("\"", "``")
                if word == temp or word.lower() == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        self.word_split = word_split
        return word_split


class FeedForward(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        kaiming_uniform_(self.linear1.weight)
        kaiming_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.linear2(self.dropout(self.relu(self.layer_norm(self.linear1(x)))))
        return x
