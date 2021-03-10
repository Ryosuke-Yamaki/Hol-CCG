import numpy as np
import torch
import csv


def load_weight_matrix(FROM_RANDOM, REGURALIZED, PATH_TO_GLOVE):
    glove_weight_matrix = []
    with open(PATH_TO_GLOVE, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            glove_weight_matrix.append(row)

    if FROM_RANDOM:
        weight_matrix = np.empty_like(glove_weight_matrix)
        EMBEDDING_DIM = len(glove_weight_matrix[0])
        for i in range(weight_matrix.shape[0]):
            weight_matrix[i] = np.random.normal(
                loc=0.0, scale=1 / np.sqrt(EMBEDDING_DIM), size=EMBEDDING_DIM)
        weight_matrix = weight_matrix.astype(np.float32)
    else:
        weight_matrix = np.array(glove_weight_matrix).astype(np.float32)

    if REGURALIZED:
        for i in range(weight_matrix.shape[0]):
            weight_matrix[i] = weight_matrix[i] / np.linalg.norm(weight_matrix[i], ord=2)

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
