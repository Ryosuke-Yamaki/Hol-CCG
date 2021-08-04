from utils import load, load_weight_matrix
import os
import torch
from utils import Condition_Setter
from torch.nn import Embedding
from torchtext.vocab import Vocab
import numpy as np


def most_similar(positive, negative, n, embedding, vocab):
    v0 = embedding(torch.tensor(vocab[positive[0]])).detach().numpy()
    v1 = embedding(torch.tensor(vocab[positive[1]])).detach().numpy()
    v2 = embedding(torch.tensor(vocab[negative[0]])).detach().numpy()
    v_target = v0 - v2 + v1
    cos_sim = []
    for v in embedding.weight:
        v = v.detach().numpy()
        cos_sim.append(np.dot(v_target, v) /
                       (np.linalg.norm(v_target) * np.linalg.norm(v)))
    top_n_idx = torch.topk(torch.tensor(cos_sim), k=n)[1]
    print('-' * 50)
    for idx in top_n_idx:
        print(vocab.itos[idx], cos_sim[idx])


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

path_to_word_counter = PATH_TO_DIR + "Hol-CCG/data/word_counter.pickle"
word_counter = load(path_to_word_counter)
content_vocab = Vocab(word_counter, specials=[])

weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim))
weight_matrix = torch.tensor(weight_matrix)

NUM_VOCAB = weight_matrix.shape[0]
embedding = Embedding(NUM_VOCAB, condition.embedding_dim, _weight=weight_matrix)

most_similar(["tokyo", "france"], ["japan"], 5, embedding, content_vocab)
most_similar(["king", "woman"], ["man"], 5, embedding, content_vocab)
most_similar(["big", "small"], ["bigger"], 5, embedding, content_vocab)
most_similar(["think", "read"], ["thinking"], 5, embedding, content_vocab)
