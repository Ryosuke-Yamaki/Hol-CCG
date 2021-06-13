import csv
from collections import Counter
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

counter = Counter()
for tree in train_tree_list.tree_list:
    for node in tree.node_list:
        if node.is_leaf:
            counter[node.content] += 1

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

initial_weight_matrix = torch.tensor(load_weight_matrix(
    condition.path_to_pretrained_weight_matrix), device=device)

EPOCHS = 10
BATCH_SIZE = 25
NUM_VOCAB = len(train_tree_list.content_vocab)
NUM_CATEGORY = len(train_tree_list.category_vocab)
tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY,
                    condition.embedding_dim).to(device)
tree_net = torch.load(condition.path_to_model,
                      map_location=device)
tree_net.eval()
trained_weight_matrix = tree_net.embedding.weight

# normalize the norm of embedding vector
initial_weight_matrix = initial_weight_matrix / \
    initial_weight_matrix.norm(dim=1, keepdim=True)
trained_weight_matrix = trained_weight_matrix / \
    initial_weight_matrix.norm(dim=1, keepdim=True)

num_vocab_in_train = len(counter)

model = Net(condition.embedding_dim)
dataset = DataSet(
    initial_weight_matrix[:num_vocab_in_train], trained_weight_matrix[:num_vocab_in_train])
dataloader = torch.utils.data.DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    drop_last=False)
criteria = nn.MSELoss()
optimizer = optim.Adam(model.parameters())

for epoch in range(1, EPOCHS+1):
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

trained_embeddings_of_train = trained_weight_matrix[:num_vocab_in_train]
initial_embeddings_of_dev_test = initial_weight_matrix[num_vocab_in_train:]
# predict projected vector from initial state of vector of unknown words
trained_embeddings_of_dev_test = model(initial_embeddings_of_dev_test)
trained_embeddings_of_dev_test = trained_embeddings_of_dev_test / \
    (trained_embeddings_of_dev_test.norm(dim=1, keepdim=True)+1e-6)
trained_weight_matrix = torch.cat(
    [trained_embeddings_of_train, trained_embeddings_of_dev_test])
trained_weight_matrix = trained_weight_matrix.cpu().detach().numpy()

with open(PATH_TO_DIR + "Hol-CCG/result/data/{}d_weight_matrix_with_projection_learning.csv".format(condition.embedding_dim), 'w') as f:
    writer = csv.writer(f, lineterminator='\n')
    writer.writerows(trained_weight_matrix)
