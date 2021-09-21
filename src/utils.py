import torch.nn as nn
from tqdm import tqdm
import os
import pickle
import csv
import random
import numpy as np
import torch
from torch import conj
from torch.fft import fft, ifft
from torch.nn.functional import normalize


def circular_correlation(a, b):
    a_ = conj(fft(a))
    b_ = fft(b)
    c_ = a_ * b_
    c = ifft(c_).real
    return normalize(c, dim=1)


def single_circular_correlation(a, b):
    a_ = conj(fft(a))
    b_ = fft(b)
    c_ = a_ * b_
    c = ifft(c_).real
    return normalize(c, dim=-1)


def load_weight_matrix(PATH_TO_WEIGHT_MATRIX):
    with open(PATH_TO_WEIGHT_MATRIX, 'r') as f:
        reader = csv.reader(f)
        weight_matrix = [row for row in reader]
    return np.array(weight_matrix).astype(np.float32)


def generate_random_weight_matrix(NUM_VOCAB, EMBEDDING_DIM):
    weight_matrix = [
        np.random.randn(EMBEDDING_DIM) for i in range(NUM_VOCAB)]
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


@torch.no_grad()
def evaluate_tree_list(tree_list, tree_net, k_list=[1, 5]):
    if tree_list.embedder == 'bert':
        for tree in tree_list.tree_list:
            tree.set_word_split(tree_net.tokenizer)
    tree_list.set_vector(tree_net)
    word_classifier = tree_net.word_classifier
    phrase_classifier = tree_net.phrase_classifier
    word_dropout = tree_net.word_dropout
    phrase_dropout = tree_net.phrase_dropout
    for k in k_list:
        num_word = 0
        num_phrase = 0
        num_correct_word = 0
        num_correct_phrase = 0
        with tqdm(total=len(tree_list.tree_list)) as pbar:
            pbar.set_description("evaluating...")
            for tree in tree_list.tree_list:
                for node in tree.node_list:
                    if node.is_leaf:
                        output = word_classifier(word_dropout(node.vector))
                        predict = torch.topk(output, k=k)[1]
                        num_word += 1
                        if node.category_id in predict and node.category_id != 0:
                            num_correct_word += 1
                    else:
                        output = phrase_classifier(phrase_dropout(node.vector))
                        predict = torch.topk(output, k=k)[1]
                        num_phrase += 1
                        if node.category_id in predict and node.category_id != 0:
                            num_correct_phrase += 1
                pbar.update(1)
        total_acc = (num_correct_word + num_correct_phrase) / (num_word + num_phrase)
        word_acc = num_correct_word / num_word
        phrase_acc = num_correct_phrase / num_phrase
        print('-' * 50)
        print('overall top-{}: {}'.format(k, total_acc))
        print('word top-{}: {}'.format(k, word_acc))
        print('phrase top-{}: {}'.format(k, phrase_acc))
        if len(k_list) == 1:
            return total_acc, word_acc, phrase_acc


@torch.no_grad()
def evaluate_batch_list(batch_list, tree_net, criteria=nn.CrossEntropyLoss()):
    print("evaluating...")
    num_word = 0
    num_phrase = 0
    num_correct_word = 0
    num_correct_phrase = 0
    word_loss = 0
    phrase_loss = 0
    with tqdm(total=len(batch_list), unit="batch") as pbar:
        for batch in batch_list:
            word_output, phrase_output, word_label, phrase_label = tree_net(batch)
            num_word += word_output.shape[0]
            num_phrase += phrase_output.shape[0]

            word_loss += criteria(word_output, word_label)
            phrase_loss += criteria(phrase_output, phrase_label)

            # remove unknown categories
            word_output = word_output[word_label != 0]
            phrase_output = phrase_output[phrase_label != 0]
            word_label = word_label[word_label != 0]
            phrase_label = phrase_label[phrase_label != 0]

            num_correct_word += torch.count_nonzero(torch.argmax(word_output, dim=1) == word_label)
            num_correct_phrase += torch.count_nonzero(
                torch.argmax(phrase_output, dim=1) == phrase_label)
            pbar.update(1)
    word_loss = word_loss / len(batch_list)
    phrase_loss = phrase_loss / len(batch_list)
    total_loss = word_loss + phrase_loss
    total_acc = (num_correct_word + num_correct_phrase) / (num_word + num_phrase)
    word_acc = num_correct_word / num_word
    phrase_acc = num_correct_phrase / num_phrase
    stat = {
        "total_acc": total_acc,
        "word_acc": word_acc,
        "phrase_acc": phrase_acc,
        "total_loss": total_loss,
        "word_loss": word_loss,
        "phrase_loss": phrase_loss}
    print("total_acc:{}\nword_acc:{}\nphrase_acc:{}".format(
        stat["total_acc"], stat["word_acc"], stat["phrase_acc"]))
    return stat


@torch.no_grad()
def evaluate_beta(tree_list, tree_net, beta):
    tree_list.set_vector(tree_net)
    word_classifier = tree_net.word_classifier
    phrase_classifier = tree_net.phrase_classifier
    num_word = 0
    num_phrase = 0
    num_correct_word = 0
    num_correct_phrase = 0
    with tqdm(total=len(tree_list.tree_list)) as pbar:
        pbar.set_description("evaluating...")
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    output = word_classifier(node.vector)
                    max_output = torch.max(output)
                    predict = list(range(len(output)))[output > max_output * beta]
                    num_word += 1
                    if node.category_id in predict and node.category_id != 0:
                        num_correct_word += 1
                else:
                    output = phrase_classifier(node.vector)
                    max_output = torch.max(output)
                    predict = list(range(len(output)))[output > max_output * beta]
                    num_phrase += 1
                    if node.category_id in predict and node.category_id != 0:
                        num_correct_phrase += 1
            pbar.update(1)
    print('-' * 50)
    print('overall top-{}: {}'.format(beta, (num_correct_word +
                                             num_correct_phrase) / (num_word + num_phrase)))
    print('word top-{}: {}'.format(beta, num_correct_word / num_word))
    print('phrase top-{}: {}'.format(beta, num_correct_phrase / num_phrase))


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

    @ torch.no_grad()
    def evaluation(self, batch_list):
        sum_loss = 0.0
        sum_acc = 0.0
        with tqdm(total=len(batch_list), unit="batch") as pbar:
            pbar.set_description("evaluating...")
            for batch in batch_list:
                word_output, phrase_output, word_label, phrase_label = self.tree_net(batch)
                # for frequency cut-off
                word_output = word_output[word_label != 0]
                phrase_output = phrase_output[phrase_label != 0]
                word_label = word_label[word_label != 0]
                phrase_label = phrase_label[phrase_label != 0]
                word_loss = self.criteria(word_output, word_label)
                phrase_loss = self.criteria(phrase_output, phrase_label)
                total_loss = word_loss + phrase_loss
                word_acc = self.cal_top_k_acc(word_output, word_label)
                phrase_acc = self.cal_top_k_acc(phrase_output, phrase_label)
                total_acc = (
                    word_acc * word_output.shape[0] + phrase_acc * phrase_output.shape[0]) / (
                    word_output.shape[0] + phrase_output.shape[0])
                sum_loss += total_loss.item()
                sum_acc += total_acc.item()
                pbar.update(1)
        self.loss_history = np.append(
            self.loss_history, sum_loss / len(batch_list))
        self.acc_history = np.append(
            self.acc_history, sum_acc / len(batch_list))
        self.update()
        return sum_loss / len(batch_list), sum_acc / len(batch_list)

    @ torch.no_grad()
    def cal_top_k_acc(self, output, label, k=1):
        output = torch.topk(output, k=k)[1]
        label = torch.reshape(label, (output.shape[0], -1))
        comparison = output - label
        num_sample = output.shape[0]
        num_false = torch.count_nonzero(comparison)
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

        self.path_to_elmo_options = PATH_TO_DIR + "Hol-CCG/data/elmo/elmo_options.json"
        self.path_to_elmo_weights = PATH_TO_DIR + "Hol-CCG/data/elmo/elmo_weights.hdf5"

        # path to counters, vocab
        self.path_to_word_category_vocab = PATH_TO_DIR + \
            "Hol-CCG/data/vocab/word_category_vocab.pickle"
        self.path_to_phrase_category_vocab = PATH_TO_DIR + \
            "Hol-CCG/data/vocab/phrase_category_vocab.pickle"
        self.path_to_whole_category_vocab = PATH_TO_DIR + \
            "Hol-CCG/data/vocab/whole_category_vocab.pickle"
        self.path_to_evalb_category_vocab = PATH_TO_DIR + \
            "Hol-CCG/data/vocab/evalb_category_vocab.pickle"
        self.path_to_word_to_whole = PATH_TO_DIR + \
            "Hol-CCG/data/vocab/word_to_whole.pickle"
        self.path_to_whole_to_phrase = PATH_TO_DIR + \
            "Hol-CCG/data/vocab/whole_to_phrase.pickle"

        # path_to_rule
        self.path_to_grammar = PATH_TO_DIR + \
            "CCGbank/ccgbank_1_1/data/GRAMMAR/CCGbank.02-21.grammar"
        self.path_to_binary_rule = PATH_TO_DIR + "Hol-CCG/data/parsing/binary_rule.txt"
        self.path_to_unary_rule = PATH_TO_DIR + "Hol-CCG/data/parsing/unary_rule.txt"

        # path_for_visualization
        self.path_to_vis_dict = PATH_TO_DIR + \
            "Hol-CCG/result/data/visualize/vis_dict.pickle"
        self.path_to_idx_dict = PATH_TO_DIR + \
            "Hol-CCG/result/data/visualize/idx_dict.pickle"
        self.path_to_color_list = PATH_TO_DIR + \
            "Hol-CCG/result/data/visualize/color_list.pickle"

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
                "Hol-CCG/result/fig/history/{}_{}d_history.pdf".format(
                    self.embedding_type, self.embedding_dim)

            # path_for_visualization
            self.path_to_visualize_weight = PATH_TO_DIR + \
                "Hol-CCG/result/data/visualize/{}_{}d".format(
                    self.embedding_type, self.embedding_dim)
            self.path_to_map = PATH_TO_DIR + \
                "Hol-CCG/result/fig/map/{}_{}d".format(
                    self.embedding_type, self.embedding_dim)
            self.path_to_wave = PATH_TO_DIR + \
                "Hol-CCG/result/fig/wave/{}_{}d".format(
                    self.embedding_type, self.embedding_dim)
            self.path_to_freq = PATH_TO_DIR + \
                "Hol-CCG/result/fig/freq/{}_{}d_freq.pdf".format(
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
