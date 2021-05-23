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
    return c / torch.norm(c)


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


def make_n_hot_label(batch_label, num_category, device=torch.device('cpu')):
    max_num_label = max([len(i) for i in batch_label])
    batch_n_hot_label_list = []
    for label_list in batch_label:
        n_hot_label_list = []
        for label in label_list:
            n_hot_label = torch.zeros(num_category, dtype=torch.float, device=device)
            n_hot_label[label] = 1.0
            n_hot_label_list.append(n_hot_label)
        batch_n_hot_label_list.append(torch.stack(n_hot_label_list))

    true_mask = [torch.ones((len(i), num_category), dtype=torch.bool, device=device)
                 for i in batch_n_hot_label_list]
    false_mask = [
        torch.zeros(
            (max_num_label - len(i),
             num_category),
            dtype=torch.bool,
            device=device) for i in batch_n_hot_label_list]
    mask = torch.stack([torch.cat((i, j)) for (i, j) in zip(true_mask, false_mask)])
    dummy_label = [
        torch.zeros(
            max_num_label - len(i),
            i.shape[1],
            dtype=torch.float,
            device=device) for i in batch_n_hot_label_list]
    batch_n_hot_label_list = torch.stack([torch.cat((i, j))
                                          for (i, j) in zip(batch_n_hot_label_list, dummy_label)])
    return batch_n_hot_label_list, mask


class History:
    def __init__(self, tree_net, tree_list, criteria, THRESHOLD):
        self.tree_net = tree_net
        self.tree_list = tree_list
        self.criteria = criteria
        self.THRESHOLD = THRESHOLD
        self.loss_history = np.array([])
        self.acc_history = np.array([])

    def update(self):
        self.min_loss = np.min(self.loss_history)
        self.min_loss_idx = np.argmin(self.loss_history)
        self.max_acc = np.max(self.acc_history)
        self.max_acc_idx = np.argmax(self.acc_history)

    def validation(self, batch_list, device=torch.device('cpu')):
        with torch.no_grad():
            total_loss = 0.0
            total_acc = 0.0
            total = 0
            for batch in batch_list:
                output = self.tree_net(batch)
                n_hot_label, mask = make_n_hot_label(batch[4], output.shape[-1], device=device)
                loss = self.criteria(output * mask, n_hot_label)
                acc = self.cal_acc(output, n_hot_label, mask)
                total_loss += loss.item()
                total_acc += acc
                total += output.shape[0]
        self.loss_history = np.append(self.loss_history, total_loss / total)
        self.acc_history = np.append(self.acc_history, total_acc / total)
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

    def print_current_stat(self, name):
        print('{}-loss: {}'.format(name, self.loss_history[-1]))
        print('{}-acc: {}'.format(name, self.acc_history[-1]))

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
        if int(input("random(0) or GloVe(1): ")) == 1:
            self.RANDOM = False
            self.param_list.append('GloVe')
        else:
            self.RANDOM = True
            self.param_list.append('random')
        self.REGULARIZED = True
        # if int(input("reg(0) or not_reg(1): ")) == 1:
        #     self.REGULARIZED = False
        #     self.param_list.append('not_reg')
        # else:
        #     self.REGULARIZED = True
        #     self.param_list.append('reg')
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
            "Hol-CCG/data/glove_{}d.csv".format(self.embedding_dim)
        path_to_initial_weight_matrix = PATH_TO_DIR + "Hol-CCG/result/data/"
        path_to_model = PATH_TO_DIR + "Hol-CCG/result/model/"
        path_to_train_data_history = PATH_TO_DIR + "Hol-CCG/result/data/"
        path_to_test_data_history = PATH_TO_DIR + "Hol-CCG/result/data/"
        path_to_history_fig = PATH_TO_DIR + "Hol-CCG/result/fig/"
        fig_name = ""
        path_list = [
            path_to_initial_weight_matrix,
            path_to_model,
            path_to_train_data_history,
            path_to_test_data_history,
            path_to_history_fig,
            fig_name]
        for i in range(len(path_list)):
            if self.RANDOM:
                path_list[i] += "random"
            else:
                path_list[i] += "GloVe"
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
        self.fig_name = path_list[5].replace('_', ' ')

    def export_param_list(self, path, roop_count):
        if roop_count == 0:
            with open(path, 'w') as f:
                f.write(', '.join(self.param_list) + '\n')
        else:
            with open(path, 'a') as f:
                f.write(', '.join(self.param_list) + '\n')
