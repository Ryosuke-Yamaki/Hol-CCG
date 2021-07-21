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
        np.random.normal(
            loc=0.0,
            scale=1 /
            np.sqrt(EMBEDDING_DIM),
            size=EMBEDDING_DIM) for i in range(NUM_VOCAB)]
    return np.array(weight_matrix).astype(np.float32)


def cal_norm_mean_std(tree):
    norm_list = []
    mean_list = []
    std_list = []
    for node in tree.node_list:
        norm_list.append(torch.norm(node.vector))
        mean_list.append(torch.mean(node.vector))
        std_list.append(torch.std(node.vector))
    norm = sum(norm_list) / len(norm_list)
    mean = sum(mean_list) / len(mean_list)
    std = sum(std_list) / len(std_list)
    return norm, mean, std


def set_random_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def original_loss(output, label, criteria, tree):
    loss = criteria(output, label)
    vector = tree.make_node_vector_tensor()
    norm = torch.norm(vector, dim=1)
    norm_base_line = torch.ones_like(norm)
    norm_loss = torch.sum(torch.abs(norm - norm_base_line))
    return loss + norm_loss


def make_label_mask(batch, device=torch.device('cpu')):
    batch_label = batch[4]
    label = torch.tensor(np.squeeze(np.vstack(batch_label)), dtype=torch.long, device=device)
    max_num_label = max([len(i) for i in batch_label])
    true_mask = [torch.ones(len(i), dtype=torch.bool, device=device) for i in batch_label]
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
    def validation(self, batch_list, device=torch.device('cpu')):
        total_loss = 0.0
        total_acc = 0.0
        for batch in batch_list:
            label, mask = make_label_mask(batch, device=device)
            output = self.tree_net(batch)
            output = output[torch.nonzero(mask, as_tuple=True)]
            loss = self.criteria(output, label)
            acc = self.cal_top_k_acc(output, label)
            total_loss += loss.item()
            total_acc += acc.item()
        self.loss_history = np.append(self.loss_history, total_loss / len(batch_list))
        self.acc_history = np.append(
            self.acc_history, total_acc / len(batch_list))
        self.update()

    @torch.no_grad()
    def cal_acc(self, batch_output, batch_label, batch_label_mask):
        # extract the element over threshold
        prediction = batch_output > self.THRESHOLD
        # matching the prediction and label, dummy node always become correct
        matching = (prediction * batch_label_mask) == batch_label
        # the number of nodes the prediction is correct
        num_correct_node = torch.count_nonzero(
            torch.all(matching, dim=2)).item()
        # the number of actually existing nodes(not a dummy nodes for fulling batch)
        num_existing_node = torch.count_nonzero(
            torch.all(batch_label_mask, dim=2)).item()
        # the number of dummy nodes
        num_dummy_node = batch_label_mask.shape[0] * \
            batch_label_mask.shape[1] - num_existing_node
        # for the dummy nodes the prediction always become correct because they are zero
        # so, subtruct them when calculate the acc
        return float((num_correct_node - num_dummy_node) / num_existing_node)

    @torch.no_grad()
    def cal_f1(self, output, n_hot_label):
        prediction = output > self.THRESHOLD
        precision = torch.count_nonzero(
            n_hot_label[torch.nonzero(prediction, as_tuple=True)]) / torch.count_nonzero(prediction)
        recall = torch.count_nonzero(prediction[torch.nonzero(
            n_hot_label, as_tuple=True)]) / torch.count_nonzero(n_hot_label)
        f1 = (2 * precision * recall) / (precision + recall)
        return precision, recall, f1

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
    def __init__(self, PATH_TO_DIR):
        self.param_list = []
        embedding_type = int(input("random(0) or GloVe(1) or FastText(2): "))
        if embedding_type == 1:
            self.RANDOM = False
            self.embedding_type = 'GloVe'
            self.param_list.append('GloVe')
        elif embedding_type == 2:
            self.RANDOM = False
            self.embedding_type = 'FastText'
            self.param_list.append('FastText')
        else:
            self.RANDOM = True
            self.embedding_type = 'random'
            self.param_list.append('random')
        self.REGULARIZED = True
        embedding_dim = input("embedding_dim(default=100d): ")
        if embedding_dim != "":
            self.embedding_dim = int(embedding_dim)
        else:
            self.embedding_dim = 100
        self.param_list.append(str(self.embedding_dim))
        self.set_path(PATH_TO_DIR)

    def set_path(self, PATH_TO_DIR):
        self.path_to_train_data = PATH_TO_DIR + "CCGbank/converted/train.txt"
        self.path_to_dev_data = PATH_TO_DIR + "CCGbank/converted/dev.txt"
        self.path_to_test_data = PATH_TO_DIR + "CCGbank/converted/test.txt"
        self.path_to_pretrained_weight_matrix = PATH_TO_DIR + \
            "Hol-CCG/data/{}_{}d.csv".format(self.embedding_type, self.embedding_dim)
        path_to_initial_weight_matrix = PATH_TO_DIR + "Hol-CCG/result/data/"
        path_to_model = PATH_TO_DIR + "Hol-CCG/result/model/"
        path_to_train_data_history = PATH_TO_DIR + "Hol-CCG/result/data/"
        path_to_test_data_history = PATH_TO_DIR + "Hol-CCG/result/data/"
        path_to_history_fig = PATH_TO_DIR + "Hol-CCG/result/fig/"
        path_to_map = PATH_TO_DIR + "Hol-CCG/result/fig/"
        fig_name = ""
        path_list = [
            path_to_initial_weight_matrix,
            path_to_model,
            path_to_train_data_history,
            path_to_test_data_history,
            path_to_history_fig,
            path_to_map,
            fig_name]
        for i in range(len(path_list)):
            path_list[i] += self.embedding_type
            if self.REGULARIZED:
                path_list[i] += "_reg"
            else:
                path_list[i] += "_not_reg"
            path_list[i] += "_" + str(self.embedding_dim) + "d"
        self.path_to_initial_weight_matrix = path_list[0] + \
            "_initial_weight_matrix.csv"
        self.path_to_model = path_list[1] + "_model.pth"
        self.path_to_train_data_history = path_list[2] + "_train_history.csv"
        self.path_to_test_data_history = path_list[3] + "_test_history.csv"
        self.path_to_history_fig = path_list[4] + "_history.png"
        self.path_to_map = path_list[5]
        self.fig_name = path_list[6].replace('_', ' ')

    def export_param_list(self, path, roop_count):
        if roop_count == 0:
            with open(path, 'w') as f:
                f.write(', '.join(self.param_list) + '\n')
        else:
            with open(path, 'a') as f:
                f.write(', '.join(self.param_list) + '\n')
