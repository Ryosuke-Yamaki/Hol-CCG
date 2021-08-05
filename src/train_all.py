import csv
from utils import load
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from models import Tree_Net
from utils import load_weight_matrix, set_random_seed, make_label_mask, evaluate, Condition_Setter, History
from tqdm import tqdm

condition = Condition_Setter(set_embedding_type=False)

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

set_random_seed(0)

print('loading tree list...')
train_tree_list = load(condition.path_to_train_tree_list)
dev_tree_list = load(condition.path_to_dev_tree_list)
test_tree_list = load(condition.path_to_test_tree_list)

unk_idx = train_tree_list.category_vocab.unk_index

train_tree_list.device = device
dev_tree_list.device = device

train_tree_list.set_info_for_training()
dev_tree_list.set_info_for_training()

EPOCHS = 100
BATCH_SIZE = 25
PATIENCE = 3
NUM_VOCAB = len(train_tree_list.content_vocab)
NUM_CATEGORY = len(train_tree_list.category_vocab)

for embedding_type in ['GloVe', 'random']:
    if embedding_type == 'GloVe':
        dim_list = [50, 100, 300]
    else:
        dim_list = [10, 50, 100, 300]
    for embedding_dim in dim_list:
        print("Start training of {}_{}".format(embedding_type, embedding_dim))
        condition.embedding_type = embedding_type
        condition.embedding_dim = embedding_dim
        condition.set_path()
        if condition.embedding_type == 'random':
            initial_weight_matrix = None
        else:
            initial_weight_matrix = load_weight_matrix(
                condition.path_to_initial_weight_matrix)

        tree_net = Tree_Net(NUM_VOCAB, NUM_CATEGORY, condition.embedding_dim,
                            initial_weight_matrix).to(device)
        if condition.embedding_type == 'random':
            # save random weight matrix as initial state
            with open(condition.path_to_initial_weight_matrix, 'w') as f:
                writer = csv.writer(f, lineterminator='\n')
                writer.writerows(tree_net.embedding.weight.detach().numpy())

        criteria = nn.CrossEntropyLoss()
        optimizer = optim.Adam(tree_net.parameters())

        train_history = History(tree_net, train_tree_list, criteria)
        dev_history = History(tree_net, dev_tree_list, criteria)

        train_batch_list = train_tree_list.make_batch(BATCH_SIZE)
        dev_batch_list = dev_tree_list.make_batch(BATCH_SIZE)

        print('calculation initial status...')
        train_history.validation(train_batch_list, unk_idx, device=device)
        dev_history.validation(dev_batch_list, unk_idx, device=device)
        print('*** initial status ***')
        train_history.print_current_stat('train')
        dev_history.print_current_stat('dev')

        for epoch in range(1, EPOCHS):
            train_batch_list = train_tree_list.make_batch(BATCH_SIZE)
            epoch_loss = 0.0
            epoch_acc = 0.0
            num_tree = 0
            num_batch = 0
            with tqdm(total=len(train_batch_list), unit="batch") as pbar:
                pbar.set_description(f"Epoch[{epoch}/{EPOCHS}]")
                for batch in train_batch_list:
                    optimizer.zero_grad()
                    label, mask = make_label_mask(batch, device=device)
                    unk_cat_mask = label != unk_idx
                    output = tree_net(batch)
                    output = output[torch.nonzero(mask, as_tuple=True)]
                    loss = criteria(output[unk_cat_mask], label[unk_cat_mask])
                    loss.backward()
                    optimizer.step()
                    acc = train_history.cal_top_k_acc(output, label)
                    epoch_loss += loss.item()
                    epoch_acc += acc.item()
                    num_batch += 1
                    pbar.set_postfix({"loss": epoch_loss / num_batch,
                                      "acc": epoch_acc / num_batch})
                    pbar.update(1)
            train_history.loss_history = np.append(
                train_history.loss_history, epoch_loss / num_batch)
            train_history.acc_history = np.append(
                train_history.acc_history, epoch_acc / num_batch)
            train_history.update()
            dev_history.validation(dev_batch_list, unk_idx, device=device)
            dev_history.print_current_stat('dev')
            train_history.save(condition.path_to_train_history)
            dev_history.save(condition.path_to_dev_history)

            if dev_history.min_loss_idx == epoch:
                torch.save(tree_net, condition.path_to_model)
            elif epoch - dev_history.min_loss_idx >= PATIENCE:
                train_history.print_best_stat('train')
                dev_history.print_best_stat('dev')
                print('train max acc: {}'.format(train_history.max_acc))
                print('dev max acc: {}'.format(dev_history.max_acc))
                # loading best model
                tree_net = torch.load(condition.path_to_model,
                                      map_location=torch.device('cpu'))
                tree_net.eval()
                # validatin model on test data
                evaluate(test_tree_list, tree_net)
                break

        fig_name = condition.fig_name
        fig = plt.figure(figsize=(10, 5))

        ax1 = fig.add_subplot(1, 2, 1)
        x = range(len(train_history.loss_history))
        ax1.plot(x, train_history.loss_history, label='train')
        ax1.plot(x, dev_history.loss_history, label='dev')
        ax1.set_ylim(0.0, np.max(train_history.loss_history) * 1.1)
        ax1.set_xlabel('epoch')
        ax1.set_ylabel('loss')
        ax1.legend()
        ax1.set_title(fig_name + ' - ' + 'loss history')

        ax2 = fig.add_subplot(1, 2, 2)
        x = range(len(train_history.acc_history))
        ax2.plot(x, train_history.acc_history, label='train')
        ax2.plot(x, dev_history.acc_history, label='dev')
        ax2.set_ylim(0.0, 1.0)
        ax2.set_xlabel('epoch')
        ax2.set_ylabel('acc')
        ax2.legend()
        ax2.set_title(fig_name + ' - ' + 'acc history')

        plt.savefig(
            condition.path_to_history_fig,
            dpi=300,
            orientation='portrait',
            transparent=False,
            pad_inches=0.0)
