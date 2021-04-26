import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
import matplotlib.pyplot as plt
from models import Tree_List, Tree_Net, Condition_Setter, History
from utils import load_weight_matrix, set_random_seed
from parsing import CCG_Category_List, Linear_Classifier, Parser
import time

for roop_count in range(8):
    PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"
    PATH_TO_STAT = PATH_TO_DIR + 'result/data/stat.txt'
    condition = Condition_Setter(PATH_TO_DIR)

    condition.export_param_list(PATH_TO_STAT, roop_count)
    # initialize tree_list from toy_data
    train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED)
    test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
    # match the vocab and category between train and test data
    test_tree_list.replace_vocab_category(train_tree_list)

    EPOCHS = 1000
    BATCH_SIZE = 5
    THRESHOLD = 0.3
    PATIENCE = 30
    NUM_VOCAB = len(train_tree_list.content_to_id)

    set_random_seed(0)

    if condition.RANDOM:
        initial_weight_matrix = None
    else:
        initial_weight_matrix = load_weight_matrix(
            condition.path_to_pretrained_weight_matrix,
            condition.REGULARIZED)

    # convert from ndarray to torch.tensor
    tree_net = Tree_Net(train_tree_list, condition.embedding_dim, initial_weight_matrix)
    criteria = nn.BCELoss(reduction='sum')
    optimizer = optim.Adam(tree_net.parameters())

    # save weight matrix as initial state
    with open(condition.path_to_initial_weight_matrix, 'w') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerows(tree_net.embedding.weight)

    # calclate initial statistics of model
    train_history = History(tree_net, train_tree_list, criteria, THRESHOLD)
    train_history.cal_stat()
    train_history.print_current_stat('train')
    test_history = History(tree_net, test_tree_list, criteria, THRESHOLD)
    test_history.cal_stat()
    test_history.print_current_stat('test')

    start = time.time()
    for epoch in range(1, EPOCHS + 1):
        batch_tree_list = train_tree_list.make_batch(BATCH_SIZE)
        for tree_list in batch_tree_list:
            optimizer.zero_grad()
            loss = 0.0
            for tree in tree_list:
                leaf_node_info = tree.leaf_node_info
                label_list = tree.label_list
                composition_info = tree.composition_info
                output = tree_net(leaf_node_info, composition_info)
                loss = loss + criteria(output, label_list)
            loss.backward()
            optimizer.step()
        train_history.cal_stat()
        test_history.cal_stat()
        # each 10 epoch print training status
        if epoch % 10 == 0:
            print('*' * 50)
            print('epoch {}'.format(epoch))
            train_history.print_current_stat('train')
            test_history.print_current_stat('test')
            end = time.time()
            print('time per epoch: {}'.format((end - start) / 10))
            start = time.time()

        if test_history.min_loss_idx == epoch:
            best_model = tree_net.state_dict()
        elif epoch - test_history.min_loss_idx > PATIENCE:
            print('')
            print('train max acc: {}'.format(train_history.max_acc))
            print('test max acc: {}'.format(test_history.max_acc))
            tree_net.load_state_dict(best_model)
            tree_net.eval()
            torch.save(tree_net, condition.path_to_model)
            break

    train_history.save(condition.path_to_train_data_history)
    test_history.save(condition.path_to_test_data_history)

    # after training, parse test data and print statistics
    ccg_category_list = CCG_Category_List(test_tree_list)
    linear_classifier = Linear_Classifier(tree_net)
    weight_matrix = tree_net.embedding.weight
    parser = Parser(
        test_tree_list,
        ccg_category_list,
        weight_matrix,
        linear_classifier,
        THRESHOLD)
    parser.validation()
    print('')
    print('parsing score taward test data')
    parser.print_stat()

    train_history.export_stat_list(PATH_TO_STAT)
    test_history.export_stat_list(PATH_TO_STAT)
    parser.export_stat_list(PATH_TO_STAT)

    fig_name = condition.fig_name
    fig = plt.figure(figsize=(10, 5))

    ax1 = fig.add_subplot(1, 2, 1)
    x = range(len(train_history.loss_history))
    ax1.plot(x, train_history.loss_history, label='train')
    ax1.plot(x, test_history.loss_history, label='test')
    ax1.set_ylim(0.0, np.max(train_history.loss_history) * 1.1)
    ax1.set_xlabel('epoch')
    ax1.set_ylabel('loss')
    ax1.legend()
    ax1.set_title(fig_name + ' - ' + 'loss history')

    ax2 = fig.add_subplot(1, 2, 2)
    x = range(len(train_history.acc_history))
    ax2.plot(x, train_history.acc_history, label='train')
    ax2.plot(x, test_history.acc_history, label='test')
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
