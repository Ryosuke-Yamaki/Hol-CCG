from utils import load, set_random_seed, Condition_Setter
from collections import Counter
import torch
import matplotlib.pyplot as plt
from torch.fft import fft


def freq_mean_std(vector_list):
    vector_list = fft(vector_list)
    amp = torch.sqrt(torch.pow(vector_list.real, 2) + torch.pow(vector_list.imag, 2)).squeeze()
    half = int(amp.shape[1] / 2)
    amp = amp[:, 1:half + 1]
    mean = []
    std = []
    for idx in range(amp.shape[1]):
        mean.append(torch.mean(amp[:, idx]).item())
        std.append(torch.std(amp[:, idx]).item())
    return mean, std


condition = Condition_Setter()

device = torch.device('cpu')
set_random_seed(0)

print('loading tree list...')
test_tree_list = load(condition.path_to_test_tree_list)

if condition.embedding_type == 'random':
    tree_net = torch.load(condition.path_to_model,
                          map_location=device)
else:
    tree_net = torch.load(condition.path_to_model_with_regression, map_location=device)
tree_net.eval()

embedding = tree_net.embedding
test_tree_list.set_vector(embedding)

n_vector = []
np_vector = []
s_vector = []
word_vector = []
phrase_vector = []
counter = Counter()

for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if counter[tuple(node.content_id)] == 0:
            counter[tuple(node.content_id)] += 1
            k = node.category
            if 'NP' in k and ('/' not in k and '\\' not in k):
                n_vector.append(node.vector)
            elif 'N' in k and ('/' not in k and '\\' not in k):
                np_vector.append(node.vector)
            elif 'S' in k and ('/' not in k and '\\' not in k):
                s_vector.append(node.vector)
            else:
                if node.is_leaf:
                    word_vector.append(node.vector)
                else:
                    phrase_vector.append(node.vector)

n_mean, n_std = freq_mean_std(torch.stack(n_vector))
np_mean, np_std = freq_mean_std(torch.stack(np_vector))
s_mean, s_std = freq_mean_std(torch.stack(s_vector))
word_mean, word_std = freq_mean_std(torch.stack(word_vector))
phrase_mean, phrase_std = freq_mean_std(torch.stack(phrase_vector))

x = range(len(n_mean))

fig = plt.figure(figsize=(15, 10))
ax1 = fig.add_subplot(2, 3, 1)
ax2 = fig.add_subplot(2, 3, 2)
ax3 = fig.add_subplot(2, 3, 3)
ax4 = fig.add_subplot(2, 3, 4)
ax5 = fig.add_subplot(2, 3, 5)

amp_max = int(max(s_mean) * 1.25)
ax1.bar(x, n_mean)
ax1.errorbar(x, n_mean, yerr=n_std, fmt='k.', capsize=2)
ax1.set_ylim(0, amp_max)
ax1.set_title('N')

ax2.bar(x, word_mean)
ax2.errorbar(x, word_mean, yerr=word_std, fmt='k.', capsize=2)
ax2.set_ylim(0, amp_max)
ax2.set_title('Word')

ax3.bar(x, np_mean)
ax3.errorbar(x, np_mean, yerr=np_std, fmt='k.', capsize=2)
ax3.set_ylim(0, amp_max)
ax3.set_title('NP')

ax4.bar(x, phrase_mean)
ax4.errorbar(x, phrase_mean, yerr=phrase_std, fmt='k.', capsize=2)
ax4.set_ylim(0, amp_max)
ax4.set_title('Phrase')

ax5.bar(x, s_mean)
ax5.errorbar(x, s_mean, yerr=s_std, fmt='k.', capsize=2)
ax5.set_ylim(0, amp_max)
ax5.set_title('S')

fig.savefig(condition.path_to_freq)
plt.show()
