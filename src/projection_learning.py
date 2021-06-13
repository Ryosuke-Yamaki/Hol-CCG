import csv
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

    def forward(self, batch):
        batch = self.linear1(batch)
        return batch


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

set_random_seed(0)

print('loading tree list...')
train_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/train_tree_list.pickle")
dev_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/dev_tree_list.pickle")
test_tree_list = load(PATH_TO_DIR + "Hol-CCG/data/test_tree_list.pickle")

content_id_in_train = []
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            content_id_in_train.append(node.content_id[0])
content_id_in_train = list(set(content_id_in_train))
unk_content_id = []
for tree in dev_tree_list.tree_list + test_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf and node.content_id[0] not in content_id_in_train:
            unk_content_id.append(node.content_id[0])
unk_content_id = list(set(unk_content_id))

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

initial_weight_matrix = torch.tensor(load_weight_matrix(
    condition.path_to_pretrained_weight_matrix), device=device)

EPOCHS = 5
BATCH_SIZE = 25
NUM_VOCAB = len(train_tree_list.content_vocab)
NUM_CATEGORY = len(train_tree_list.category_vocab)
tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()
trained_weight_matrix = tree_net.embedding.weight.detach()

initial_weight_of_train = []
trained_weight_of_train = []

with torch.no_grad():
    for content_id in content_id_in_train:
        initial_weight_of_train.append(initial_weight_matrix[content_id])
        trained_weight_of_train.append(trained_weight_matrix[content_id])

    initial_weight_of_unk = []
    for content_id in unk_content_id:
        initial_weight_of_unk.append(initial_weight_matrix[content_id])

    initial_weight_of_train = torch.stack(initial_weight_of_train)
    trained_weight_of_train = torch.stack(trained_weight_of_train)
    initial_weight_of_unk = torch.stack(initial_weight_of_unk)

model = Net(condition.embedding_dim).to(device)
dataset = DataSet(
    initial_weight_of_train, trained_weight_of_train)
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=True)
criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(1, EPOCHS + 1):
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
            loss = criteria(output, target)
            loss.backward(retain_graph=True)
            epoch_loss += loss.item()
            optimizer.step()
            pbar.set_postfix({"loss": epoch_loss / num_batch})
            pbar.update(1)

torch.save(model, PATH_TO_DIR +
           "Hol-CCG/result/model/{}d_projection.pth".format(condition.embedding_dim))

# predict projected vector from initial state of vector of unknown words
for content_id, weight in zip(unk_content_id, initial_weight_of_unk):
    trained_weight_matrix[content_id] = model(weight)
trained_weight_matrix = trained_weight_matrix.cpu().detach().numpy()

with open(PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim), 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(trained_weight_matrix)

cos = nn.CosineSimilarity()
cos_list = []

for data in dataloader:
    input = data[0]
    target = data[1]
    output = model(input)
    cos_list.append(torch.mean(cos(output, target)).item())
print(np.mean(cos_list))
