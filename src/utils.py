import numpy as np
import torch
import csv
import random
from torch.fft import fft, ifft
from torch import conj, mul


def circular_correlation(a, b):
    a = conj(fft(a))
    b = fft(b)
    c = mul(a, b)
    c = ifft(c).real
    return c.div(c.norm(dim=1, keepdim=True) + 1e-6)


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
    return np.array([i / j for (i, j) in zip(weight_matrix,
                                             np.linalg.norm(weight_matrix, axis=1))]).astype(np.float32)


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
