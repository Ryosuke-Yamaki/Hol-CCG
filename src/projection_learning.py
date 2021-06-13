from collections import Counter
import numpy as np
from utils import load
import os
import torch
import torch.nn as nn
import torch.optim as optim
from models import Tree_Net
from utils import load_weight_matrix, set_random_seed, Condition_Setter
from tqdm import tqdm


class DataSet:
    def __init__(self, input, output):
        self.input = input
        self.output = output

    def __len__(self):
        return self.input.shape[0]

    def __getitem__(self, idx):
        return self.input[idx], self.output[idx]


class Net(nn.Module):
    def __init__(self, embedding_dim):
        super(Net, self).__init__()
        self.embedding_dim = embedding_dim
        self.linear1 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.linear2 = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, batch):
        batch = self.tanh(self.linear1(batch))
        batch = self.tanh(self.linear2(batch))
        return batch


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

set_random_seed(0)

print('loading tree list...')
train_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
dev_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/dev_tree_list.pickle")
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

counter = Counter()
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            counter[node.content] += 1

unk_counter = Counter()
for tree in dev_tree_list.tree_list + test_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf and counter[node.content] == 0:
            unk_counter[node.content] += 1


if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

initial_weight_matrix = torch.tensor(load_weight_matrix(
    condition.path_to_pretrained_weight_matrix), device=device)

EPOCHS = 100
BATCH_SIZE = 25
NUM_VOCAB = len(test_tree_list.content_vocab)
NUM_CATEGORY = len(test_tree_list.category_vocab)
tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY, condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()
trained_weight_matrix = tree_net.embedding.weight

initial_weight_matrix = initial_weight_matrix / initial_weight_matrix.norm(dim=1, keepdim=True)
trained_weight_matrix = trained_weight_matrix / initial_weight_matrix.norm(dim=1, keepdim=True)


model = Net(condition.embedding_dim)

dataset = DataSet(initial_weight_matrix, trained_weight_matrix)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False)
criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(1, EPOCHS):
    with tqdm(total=len(dataloader), unit="batch") as pbar:
        pbar.set_description(f"Epoch[{epoch}/{EPOCHS}]")
        epoch_loss = 0.0
        num_batch = 0
        for data in dataloader:
            num_batch += 1
            optimizer.zero_grad()
            input = data[0]
            target = data[1]
            output = model(input)
            loss = criteria(input, target)
            loss.backward(retain_graph=True)
            epoch_loss += loss.item()
            optimizer.step()
            pbar.set_postfix({"loss": epoch_loss / num_batch})
            pbar.update(1)

dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=1,
    shuffle=False,
    drop_last=False)

cos = nn.CosineSimilarity()

for data in dataloader:
    input = data[0]
    target = data[1]
    output = model(input)
    print(cos(output, target))
