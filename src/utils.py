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


def circular_correlation(a, b):
    a_ = conj(fft(a))
    b_ = fft(b)
    c_ = a_ * b_
    c = standardize(ifft(c_).real)
    return c


def single_circular_correlation(a, b):
    a_ = conj(fft(a))
    b_ = fft(b)
    c_ = a_ * b_
    c = standardize(ifft(c_).real)
    return c


def circular_convolution(a, b):
    a_ = fft(a)
    b_ = fft(b)
    c_ = a_ * b_
    c = standardize(ifft(c_).real)
    return c


def single_circular_convolution(a, b):
    a_ = fft(a)
    b_ = fft(b)
    c_ = a_ * b_
    c = standardize(ifft(c_).real)
    return c


def standardize(v):
    original_shape = v.shape
    v = v.view(-1, v.shape[-1])
    mean = torch.mean(v, dim=-1).view(-1, 1)
    std = torch.std(v, dim=-1, unbiased=False).view(-1, 1)
    dim = v.shape[-1]
    v = (v - mean) / (std * np.sqrt(dim))
    v = v.view(original_shape)
    return v


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
def evaluate_tree_list(tree_list, tree_net):
    if tree_list.embedder == 'transformer':
        for tree in tree_list.tree_list:
            tree.set_word_split(tree_net.tokenizer)
    tree_list.set_vector(tree_net)
    word_ff = tree_net.word_ff
    phrase_ff = tree_net.phrase_ff
    num_word = 0
    num_phrase = 0
    num_correct_word = 0
    num_correct_phrase = 0
    with tqdm(total=len(tree_list.tree_list)) as pbar:
        pbar.set_description("evaluating...")
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    output = word_ff(node.vector)
                    predict = torch.topk(output, k=1)[1]
                    num_word += 1
                    if node.category_id in predict and node.category_id != 0:
                        num_correct_word += 1
                else:
                    output = phrase_ff(node.vector)
                    predict = torch.topk(output, k=1)[1]
                    num_phrase += 1
                    if node.category_id in predict and node.category_id != 0:
                        num_correct_phrase += 1
            pbar.update(1)
    word_acc = num_correct_word / num_word
    phrase_acc = num_correct_phrase / num_phrase
    print('-' * 50)
    print('word acc: {}'.format(word_acc))
    print('phrase acc: {}'.format(phrase_acc))
    return word_acc, phrase_acc


@torch.no_grad()
def evaluate_batch_list(
        batch_list,
        tree_net,
        cat_criteria=nn.CrossEntropyLoss(),
        span_criteria=nn.BCELoss()):
    print("evaluating...")
    num_word = 0
    num_phrase = 0
    num_span = 0
    num_correct_word = 0
    num_correct_phrase = 0
    num_correct_span = 0
    word_loss = 0
    phrase_loss = 0
    span_loss = 0
    with tqdm(total=len(batch_list), unit="batch") as pbar:
        for batch in batch_list:
            word_output, phrase_output, span_output, word_label, phrase_label, span_label = tree_net(
                batch)
            span_output = torch.sigmoid(span_output)

            num_word += word_output.shape[0]
            num_phrase += phrase_output.shape[0]
            num_span += span_output.shape[0]

            word_loss += cat_criteria(word_output, word_label)
            phrase_loss += cat_criteria(phrase_output, phrase_label)
            span_loss += span_criteria(span_output, span_label)

            # remove unknown categories
            word_output = word_output[word_label != 0]
            phrase_output = phrase_output[phrase_label != 0]
            word_label = word_label[word_label != 0]
            phrase_label = phrase_label[phrase_label != 0]

            num_correct_word += torch.count_nonzero(torch.argmax(word_output, dim=1) == word_label)
            num_correct_phrase += torch.count_nonzero(
                torch.argmax(phrase_output, dim=1) == phrase_label)
            span_output[span_output >= 0.5] = 1.0
            span_output[span_output < 0.5] = 0.0
            num_correct_span += torch.count_nonzero(span_output == span_label)

            pbar.update(1)

    word_loss = word_loss / len(batch_list)
    phrase_loss = phrase_loss / len(batch_list)
    span_loss = span_loss / len(batch_list)
    total_loss = word_loss + phrase_loss + span_loss
    word_acc = num_correct_word / num_word
    phrase_acc = num_correct_phrase / num_phrase
    span_acc = num_correct_span / num_span

    stat = {
        "total_loss": total_loss,
        "word_acc": word_acc,
        "phrase_acc": phrase_acc,
        "span_acc": span_acc,
        "word_loss": word_loss,
        "phrase_loss": phrase_loss,
        "span_loss": span_loss}

    print("word_acc:{}\nphrase_acc:{}\nspan_acc:{}".format(
        stat["word_acc"], stat["phrase_acc"], stat["span_acc"]))

    return stat


@torch.no_grad()
def evaluate_beta(tree_list, tree_net, beta=0.0005, alpha=10):
    # if tree_list.embedder == 'transformer':
    #     for tree in tree_list.tree_list:
    #         tree.set_word_split(tree_net.tokenizer)
    # tree_list.set_vector(tree_net)
    word_ff = tree_net.word_ff
    phrase_ff = tree_net.phrase_ff
    num_word = 0
    num_phrase = 0
    num_correct_word = 0
    num_correct_phrase = 0
    num_predicted_word = 0
    num_predicted_phrase = 0
    with tqdm(total=len(tree_list.tree_list)) as pbar:
        pbar.set_description("evaluating...")
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if node.is_leaf:
                    # probability distribution
                    output = torch.softmax(word_ff(node.vector), dim=-1)
                    predict_prob, predict_idx = torch.sort(output[1:], dim=-1, descending=True)
                    # add one to index for the removing of zero index of "<UNK>"
                    predict_idx += 1
                    predict_idx = predict_idx[predict_prob > beta][:alpha]
                    num_word += 1
                    num_predicted_word += predict_idx.shape[0]
                    if node.category_id in predict_idx and node.category_id != 0:
                        num_correct_word += 1
                else:
                    output = torch.softmax(phrase_ff(node.vector), dim=-1)
                    predict_prob, predict_idx = torch.sort(output[1:], dim=-1, descending=True)
                    predict_idx += 1
                    predict_idx = predict_idx[predict_prob > beta][:alpha]
                    num_phrase += 1
                    num_predicted_phrase += predict_idx.shape[0]
                    if node.category_id in predict_idx and node.category_id != 0:
                        num_correct_phrase += 1
            pbar.update(1)
    print('-' * 50)
    print('beta={},alpha={}'.format(beta, alpha))
    print('word: {}'.format(num_correct_word / num_word))
    print('cat per word: {}'.format(num_predicted_word / num_word))
    print('phrase: {}'.format(num_correct_phrase / num_phrase))
    print('cat per phrase: {}'.format(num_predicted_phrase / num_phrase))

    return num_correct_word / num_word, num_predicted_word / num_word


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
        self.path_to_head_info = PATH_TO_DIR + "Hol-CCG/data/parsing/head_info.txt"

        # path_for_visualization
        self.path_to_vis_dict = PATH_TO_DIR + \
            "Hol-CCG/result/data/visualize/vis_dict.pickle"
        self.path_to_idx_dict = PATH_TO_DIR + \
            "Hol-CCG/result/data/visualize/idx_dict.pickle"
        self.path_to_color_list = PATH_TO_DIR + \
            "Hol-CCG/result/data/visualize/color_list.pickle"

        # path to trained_model
        self.path_to_model = PATH_TO_DIR + "Hol-CCG/result/data/model/"

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

            # path_to_glove_model
            self.path_to_glove_model = PATH_TO_DIR + \
                "Hol-CCG/result/data/model/{}_{}d_model.pth".format(
                    self.embedding_type, self.embedding_dim)
            if self.embedding_type == "GloVe":
                self.path_to_glove_model_with_regression = PATH_TO_DIR + \
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
