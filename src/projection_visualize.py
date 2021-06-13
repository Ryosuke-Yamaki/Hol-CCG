from utils import load_weight_matrix, load
from sklearn.decomposition import PCA
import os
import matplotlib.pyplot as plt
from utils import set_random_seed, Condition_Setter
from torchtext.vocab import Vocab


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

set_random_seed(0)

print("loading_weight_matrix...")
initial_weight_matrix = load_weight_matrix(
    condition.path_to_pretrained_weight_matrix)
trained_weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim))

path_to_word_counter = PATH_TO_DIR + "Hol-CCG/data/word_counter.pickle"
word_counter = load(path_to_word_counter)
content_vocab = Vocab(word_counter, specials=[])

unk_word_list = [
    "edmonton",
    "thirty-five",
    "distaste",
    "outstripping"]
known_word_list = [
    ["calgary", "ottawa", "winnipeg"],
    ["twenty-five", "thirty-four", "twenty-one"],
    ["disdain", "unhappiness", "ambivalence"],
    ["outstripped", "outstrips", "outstrip"]]
known_id_list = []
unk_id_list = []

for unk_word, idx in zip(unk_word_list, range(len(unk_word_list))):
    unk_id_list.append(content_vocab[unk_word])
    temp = []
    for known_word in known_word_list[idx]:
        temp.append(content_vocab[known_word])
    known_id_list.append(temp)


def scatter(unk_id, known_id_list, ax, embedded, vocab, text):
    xy = embedded[unk_id]
    ax.scatter(xy[0], xy[1], color='b', s=10)
    text = annotate([unk_id], [xy], ax, vocab, text)
    ax.scatter(embedded[known_id_list][:, 0], embedded[known_id_list][:, 1], s=10)
    text = annotate(known_id_list, embedded[known_id_list], ax, vocab, text)
    return text


def annotate(id_list, xy_list, ax, vocab, text):
    for id, xy in zip(id_list, xy_list):
        text.append(ax.annotate(vocab.itos[id], tuple(xy), fontsize='large'))
    return text


pca = PCA(n_components=2)
print("PCA working...")
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax1.set_xlim(-2.1, 2.1)
ax1.set_ylim(-2.1, 2.1)
ax1.set_title('Initial state', fontsize='large')
ax1.grid()
ax2 = fig.add_subplot(1, 2, 2)
ax2.set_xlim(-2.1, 2.1)
ax2.set_ylim(-2.1, 2.1)
ax2.set_title('After training', fontsize='large')
ax2.grid()
text = []
embedded = pca.fit_transform(initial_weight_matrix)
for unk_id, known_id in zip(unk_id_list, known_id_list):
    text = scatter(unk_id, known_id, ax1, embedded, content_vocab, text)
embedded = pca.fit_transform(trained_weight_matrix)
for unk_id, known_id in zip(unk_id_list, known_id_list):
    text = scatter(unk_id, known_id, ax2, embedded, content_vocab, text)
# adjust_text(text, arrowprops=dict(arrowstyle='-', color='gray', lw=0.5))
plt.show()
