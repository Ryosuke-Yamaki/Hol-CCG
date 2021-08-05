import csv
import numpy as np
from utils import load
import torch
import torch.nn as nn
import torch.optim as optim
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
        self.linear1 = nn.Linear(self.embedding_dim, 2 * self.embedding_dim)
        self.linear2 = nn.Linear(2 * self.embedding_dim, self.embedding_dim)
        self.tanh = nn.Tanh()

    def forward(self, batch):
        batch = self.linear1(batch)
        batch = self.linear2(self.tanh(batch))
        return batch


condition = Condition_Setter(set_embedding_type=False)
condition.embedding_type = 'GloVe'

print('loading tree list...')
train_tree_list = load(condition.path_to_train_tree_list)
dev_tree_list = load(condition.path_to_dev_tree_list)
test_tree_list = load(condition.path_to_test_tree_list)

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

set_random_seed(0)

for embedding_dim in [50, 100, 300]:
    condition.embedding_dim = embedding_dim
    condition.set_path()

    initial_weight_matrix = torch.tensor(load_weight_matrix(
        condition.path_to_initial_weight_matrix), device=device)

    EPOCHS = 10
    BATCH_SIZE = 25
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
    train_size = int(len(dataset) * 0.8)
    val_size = int(len(dataset) - train_size)
    train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True)
    criteria = nn.MSELoss()
    optimizer = optim.Adam(model.parameters())

    for epoch in range(1, EPOCHS + 1):
        with tqdm(total=len(train_dataloader), unit="batch") as pbar:
            pbar.set_description(f"Epoch[{epoch}/{EPOCHS}]")
            epoch_loss = 0.0
            num_batch = 0
            for data in train_dataloader:
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
        cos_list = []
        for input, output in val_dataset:
            predict = model(input).detach().numpy()
            output = output.detach().numpy()
            cos_list.append(np.dot(output, predict) /
                            (np.linalg.norm(output) * np.linalg.norm(predict)))
        print("val_cos_sim: ", np.mean(cos_list))

    torch.save(model, condition.path_to_model_with_regression)

    # predict projected vector from initial state of vector of unknown words
    for content_id, weight in zip(unk_content_id, initial_weight_of_unk):
        trained_weight_matrix[content_id] = model(weight)
    trained_weight_matrix = trained_weight_matrix.cpu().detach().numpy()

    with open(condition.path_to_weight_with_regression, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(trained_weight_matrix)
