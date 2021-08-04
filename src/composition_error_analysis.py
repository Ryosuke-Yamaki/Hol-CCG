from collections import Counter
import os
import torch
import matplotlib.pyplot as plt
from utils import set_random_seed, Condition_Setter
from models import Tree_List, Tree_Net

PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

device = torch.device('cpu')

set_random_seed(0)
print('loading tree list...')
train_tree_list = Tree_List(condition.path_to_train_data, device=device)
test_tree_list = Tree_List(
    condition.path_to_test_data,
    train_tree_list.content_vocab,
    train_tree_list.category_vocab,
    device=device)
test_tree_list.clean_tree_list()

tree_net = Tree_Net(train_tree_list, condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()
linear = tree_net.linear
embedding = tree_net.embedding
test_tree_list.set_vector(embedding)

for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            node.composition_count = 0
for tree in test_tree_list.tree_list:
    for info in tree.composition_info:
        num_child = tree.composition_info[0]
        parent_node = tree.node_list[info[1]]
        if num_child == 1:
            child_node = tree.node_list[info[2]]
            parent_node.composition_count = child_node.composition_count
        else:
            left_child_node = tree.node_list[info[2]]
            right_child_node = tree.node_list[info[3]]
            parent_node.composition_count = left_child_node.composition_count + right_child_node.composition_count + 1

sample_counter = Counter()
correct_counter = Counter()
for tree in test_tree_list.tree_list:
    for node in tree.node_list:
        sample_counter[node.composition_count] += 1
        output = linear(node.vector)
        predict = torch.topk(output, k=5)[1]
        if node.category_id in predict:
            correct_counter[node.composition_count] += 1
x = []
num_sample = []
num_correct = []
err = []
for key in sorted(sample_counter.keys()):
    if sample_counter[key] >= 100:
        x.append(key)
        num_sample.append(sample_counter[key])
        num_correct.append(correct_counter[key])
        err.append(1 - correct_counter[key] / sample_counter[key])

fig1 = plt.figure()
ax1 = fig1.add_subplot(1, 1, 1)
ax2 = ax1.twinx()

ax1.plot(x, num_sample, label='number of sample')
ax1.legend()
ax1.set_xlabel('times of composition')
ax1.set_ylabel('number of samples')

ax2.plot(x, err, label='Error Rate', linestyle=':')
ax2.set_ylabel('Error Rate')
ax2.set_ylim(0.0, 1.0)
plt.xticks(fontsize='large')
plt.yticks(fontsize='large')
handler1, label1 = ax1.get_legend_handles_labels()
handler2, label2 = ax2.get_legend_handles_labels()
ax1.legend(handler1 + handler2, label1 + label2, loc=2, borderaxespad=0.)

plt.show()
