import os
import pickle
import csv
import random
import numpy as np
import torch
from torch import conj
from torch.fft import fft, ifft


def circular_correlation(a, b):
    a = conj(fft(a))
    b = fft(b)
    c = ifft(a * b).real
    norm = c.norm(dim=1, keepdim=True) + 1e-6
    return c / norm


def single_circular_correlation(a, b):
    a = conj(fft(a))
    b = fft(b)
    c = ifft(a * b).real
    return c / (torch.norm(c) + 1e-6)


def load_weight_matrix(PATH_TO_WEIGHT_MATRIX):
    with open(PATH_TO_WEIGHT_MATRIX, 'r') as f:
        reader = csv.reader(f)
        weight_matrix = [row for row in reader]
    return np.array(weight_matrix).astype(np.float32)


def generate_random_weight_matrix(NUM_VOCAB, EMBEDDING_DIM):
    weight_matrix = [
        np.random.rand(EMBEDDING_DIM) for i in range(NUM_VOCAB)]
    return np.array(weight_matrix).astype(np.float32)


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def make_label_mask(batch, device=torch.device('cpu')):
    batch_label = batch[4]
    label = torch.tensor(np.squeeze(np.vstack(batch_label)),
                         dtype=torch.long, device=device)
    max_num_label = max([len(i) for i in batch_label])
    true_mask = [torch.ones(len(i), dtype=torch.bool, device=device)
                 for i in batch_label]
    false_mask = [
        torch.zeros(
            max_num_label - len(i),
            dtype=torch.bool,
            device=device) for i in batch_label]
    mask = torch.stack([torch.cat((i, j))
                        for (i, j) in zip(true_mask, false_mask)])
    return label, mask


def dump(object, path):
    with open(path, mode='wb') as f:
        pickle.dump(object, f)


def load(path):
    with open(path, mode='rb') as f:
        data = pickle.load(f)
        return data


def include_unk(content_id, unk_content_id):
    bit = 0
    for id in content_id:
        if id in unk_content_id:
            bit = 1
            return True
    if bit == 0:
        return False


def evaluate(tree_list, tree_net, unk_content_id=None):
    embedding = tree_net.embedding
    linear = tree_net.linear
    tree_list.set_vector(embedding)
    for k in [1, 5]:
        num_word = 0
        num_phrase = 0
        num_correct_word = 0
        num_correct_phrase = 0
        if unk_content_id is None:
            for tree in tree_list.tree_list:
                for node in tree.node_list:
                    output = linear(node.vector)
                    predict = torch.topk(output, k=k)[1]
                    if node.is_leaf:
                        num_word += 1
                        if node.category_id in predict:
                            num_correct_word += 1
                    else:
                        num_phrase += 1
                        if node.category_id in predict:
                            num_correct_phrase += 1
        else:
            for tree in tree_list.tree_list:
                for node in tree.node_list:
                    if include_unk(node.content_id, unk_content_id):
                        output = linear(node.vector)
                        predict = torch.topk(output, k=k)[1]
                        if node.is_leaf:
                            num_word += 1
                            if node.category_id in predict:
                                num_correct_word += 1
                        else:
                            num_phrase += 1
                            if node.category_id in predict:
                                num_correct_phrase += 1
        print('-' * 50)
        print('overall top-{}: {}'.format(k, (num_correct_word +
                                              num_correct_phrase) / (num_word + num_phrase)))
        print('word top-{}: {}'.format(k, num_correct_word / num_word))
        print('phrase top-{}: {}'.format(k, num_correct_phrase / num_phrase))


class History:
    def __init__(self, tree_net, tree_list, criteria):
        self.tree_net = tree_net
        self.tree_list = tree_list
        self.criteria = criteria
        self.loss_history = np.array([])
        self.acc_history = np.array([])

    def update(self):
        self.min_loss = np.min(self.loss_history)
        self.min_loss_idx = np.argmin(self.loss_history)
        self.max_acc = np.max(self.acc_history)
        self.max_acc_idx = np.argmax(self.acc_history)

    @torch.no_grad()
    def validation(self, batch_list, unk_idx, device=torch.device('cpu')):
        total_loss = 0.0
        total_acc = 0.0
        for batch in batch_list:
            label, mask = make_label_mask(batch, device=device)
            unk_cat_mask = label != unk_idx
            output = self.tree_net(batch)
            output = output[torch.nonzero(mask, as_tuple=True)]
            loss = self.criteria(output[unk_cat_mask], label[unk_cat_mask])
            acc = self.cal_top_k_acc(output[unk_cat_mask], label[unk_cat_mask])
            total_loss += loss.item()
            total_acc += acc.item()
        self.loss_history = np.append(
            self.loss_history, total_loss / len(batch_list))
        self.acc_history = np.append(
            self.acc_history, total_acc / len(batch_list))
        self.update()

    @torch.no_grad()
    def cal_top_k_acc(self, output, label, k=5):
        output = torch.topk(output, k=k)[1]
        label = torch.reshape(label, (output.shape[0], -1))
        comparison = output - label
        num_sample = output.shape[0]
        num_false = torch.count_nonzero(torch.all(comparison, dim=1))
        num_true = num_sample - num_false
        return num_true / num_sample

    def print_current_stat(self, name):
        print('{}-loss: {}'.format(name, self.loss_history[-1]))
        print('{}-acc: {}'.format(name, self.acc_history[-1]))

    def print_best_stat(self, name):
        print('{}-min loss: {}'.format(name, self.min_loss))
        print('{}-best acc: {}'.format(name, self.max_acc))

    def save(self, path):
        with open(path, 'w') as f:
            writer = csv.writer(f, lineterminator='\n')
            writer.writerow(self.loss_history)
            writer.writerow(self.acc_history)

    def export_stat_list(self, path):
        stat_list = []
        stat_list.append(str(self.min_loss))
        stat_list.append(str(self.max_acc))
        with open(path, 'a') as f:
            f.write(', '.join(stat_list) + '\n')


class Condition_Setter:
    def __init__(self, PATH_TO_DIR=None, set_embedding_type=True):
        if PATH_TO_DIR is None:
            PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
            self.PATH_TO_DIR = PATH_TO_DIR
        if set_embedding_type:
            self.param_list = []
            embedding_type = int(input("GloVe(0) or random(1): "))
            if embedding_type == 0:
                self.embedding_type = 'GloVe'
                self.param_list.append('GloVe')
            elif embedding_type == 1:
                self.embedding_type = 'random'
                self.param_list.append('random')
            else:
                print("Error: embedding_type")
                exit()
            embedding_dim = input("embedding_dim: ")
            if self.embedding_type == 'GloVe':
                if embedding_dim in ["50", "100", "300"]:
                    self.embedding_dim = int(embedding_dim)
                else:
                    print("Error: embedding_dim")
                    exit()
            else:
                if embedding_dim in ["10", "50", "100", "300"]:
                    self.embedding_dim = int(embedding_dim)
                else:
                    print("Error: embedding_dim")
                    exit()
            self.param_list.append(str(self.embedding_dim))
        self.set_path(set_embedding_type)

    def set_path(self, set_embedding_type=True):
        PATH_TO_DIR = self.PATH_TO_DIR
        # ******************** the path not depend on the embedding type********************
        # path to the tree data
        self.path_to_train_data = PATH_TO_DIR + "CCGbank/converted/train.txt"
        self.path_to_dev_data = PATH_TO_DIR + "CCGbank/converted/dev.txt"
        self.path_to_test_data = PATH_TO_DIR + "CCGbank/converted/test.txt"
        self.path_to_train_tree_list = PATH_TO_DIR + \
            "Hol-CCG/data/tree_list/train_tree_list.pickle"
        self.path_to_dev_tree_list = PATH_TO_DIR + \
            "Hol-CCG/data/tree_list/dev_tree_list.pickle"
        self.path_to_test_tree_list = PATH_TO_DIR + \
            "Hol-CCG/data/tree_list/test_tree_list.pickle"

        # path to counters, vocab
        self.path_to_word_counter = PATH_TO_DIR + \
            "Hol-CCG/data/counter/word_counter.pickle"
        self.path_to_train_word_counter = PATH_TO_DIR + \
            "Hol-CCG/data/counter/train_word_counter.pickle"
        self.path_to_category_counter = PATH_TO_DIR + \
            "Hol-CCG/data/counter/category_counter.pickle"
        self.path_to_content_vocab = PATH_TO_DIR + \
            "Hol-CCG/data/parsing/content_vocab.txt"

        # path_to_rule
        self.path_to_grammar = PATH_TO_DIR + \
            "CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar"
        self.path_to_binary_rule = PATH_TO_DIR + "Hol-CCG/data/parsing/binary_rule.txt"
        self.path_to_unary_rule = PATH_TO_DIR + "Hol-CCG/data/parsing/unary_rule.txt"

        # ******************** the path depend on the embedding type********************
        if set_embedding_type:
            # path_to_weight_matrix
            self.path_to_initial_weight_matrix = PATH_TO_DIR + \
                "Hol-CCG/data/initial_weight/{}_{}d_initial_weight.csv".format(
                    self.embedding_type, self.embedding_dim)
            if self.embedding_type == "GloVe":
                self.path_to_weight_with_regression = PATH_TO_DIR + \
                    "Hol-CCG/result/data/weight_matrix/{}_{}d_weight_with_regression.csv".format(
                        self.embedding_type, self.embedding_dim)

            # path_to_model
            self.path_to_model = PATH_TO_DIR + \
                "Hol-CCG/result/data/model/{}_{}d_model.pth".format(
                    self.embedding_type, self.embedding_dim)
            if self.embedding_type == "GloVe":
                self.path_to_model_with_regression = PATH_TO_DIR + \
                    "Hol-CCG/result/data/model/{}_{}d_model_with_regression.pth".format(
                        self.embedding_type, self.embedding_dim)

            # path_to_history
            self.path_to_train_history = PATH_TO_DIR + \
                "Hol-CCG/result/data/history/{}_{}d_train_history.csv".format(
                    self.embedding_type, self.embedding_dim)
            self.path_to_dev_history = PATH_TO_DIR + \
                "Hol-CCG/result/data/history/{}_{}d_dev_history.csv".format(
                    self.embedding_type, self.embedding_dim)
            self.path_to_history_fig = PATH_TO_DIR + \
                "Hol-CCG/result/fig/history/{}_{}d_history.png".format(
                    self.embedding_type, self.embedding_dim)

            # path_to_figures
            self.path_to_map = PATH_TO_DIR + \
                "Hol-CCG/result/fig/map/{}_{}d".format(
                    self.embedding_type, self.embedding_dim)
            self.fig_name = "{} {}d".format(
                self.embedding_type, self.embedding_dim)

    def export_param_list(self, path, roop_count):
        if roop_count == 0:
            with open(path, 'w') as f:
                f.write(', '.join(self.param_list) + '\n')
        else:
            with open(path, 'a') as f:
                f.write(', '.join(self.param_list) + '\n')
