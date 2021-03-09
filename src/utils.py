import numpy as np
import torch
import csv


def load_weight_matrix(path_to_weight_matrix):
    weight_matrix = []
    with open(path_to_weight_matrix, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            weight_matrix.append(row)

    weight_matrix = np.array(weight_matrix).astype(np.float32)
    weight_matrix = torch.from_numpy(weight_matrix).clone()
    return weight_matrix


def cal_acc(output, label):
    num_True = 0
    num_False = 0

    for i in range(output.shape[0]):
        pred = torch.argmax(output[i])
        correct = label[i]
        if pred == correct:
            num_True += 1
        else:
            num_False += 1
    acc = num_True / (num_True + num_False)
    return acc
