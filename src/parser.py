import numpy as np
import time
import re
import torch
from utils import single_circular_correlation


class Parser:
    @torch.no_grad()
    def __init__(self, tree_net, content_vocab, binary_rule, unary_rule):
        self.embedding = tree_net.embedding
        self.linear = tree_net.linear
        self.content_vocab = content_vocab
        self.binary_rule = binary_rule
        self.unary_rule = unary_rule
        self.softmax = torch.nn.Softmax(dim=-1)

    @torch.no_grad()
    def tokenize(self, sentence):
        vector_list = []
        for word in sentence.split():
            word = word.lower()
            word = re.sub(r'\d+', '0', word)
            word = re.sub(r'\d,\d', '0', word)
            word_id = torch.tensor(self.content_vocab[word])
            vector = self.embedding(word_id)
            vector_list.append(vector / torch.norm(vector))
        return vector_list

    @torch.no_grad()
    def parse(self, sentence, num_category, vector_dim=100, mergin=5):
        vector_list = self.tokenize(sentence)
        n = len(vector_list)
        category_table = [[[] for i in range(n + 1)] for j in range(n + 1)]
        prob = torch.zeros((n + 1, n + 1, num_category))
        backpointer = torch.zeros((n + 1, n + 1, num_category, 3), dtype=torch.int)
        vector = torch.zeros((n + 1, n + 1, num_category, vector_dim))

        for i in range(n):
            output = self.softmax(self.linear(vector_list[i]))
            predict = torch.topk(output, k=mergin)
            for P, A in zip(predict[0], predict[1]):
                category_table[i][i + 1].append(A)
                prob[(i, i + 1, A)] = P
                vector[(i, i + 1, A)] = vector_list[i]
        binary_time = 0
        unary_time = 0
        cut_off_time = 0
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                start = time.time()
                j = i + length
                for k in range(i + 1, j):
                    for S1 in category_table[i][k]:
                        for S2 in category_table[k][j]:
                            # list of gramatically possible category
                            possible_cat = self.binary_rule[S1][S2]
                            if possible_cat == []:
                                continue
                            else:
                                composed_vector = single_circular_correlation(
                                    vector[(i, k, S1)], vector[(k, j, S2)])
                                prob_dist = self.softmax(self.linear(composed_vector))
                                possible_cat_prob = torch.index_select(
                                    input=prob_dist, dim=-1, index=torch.tensor(possible_cat))
                                for A, P in zip(possible_cat, possible_cat_prob):
                                    if A not in category_table[i][j]:
                                        category_table[i][j].append(A)
                                    P = P * prob[(i, k, S1)] * prob[(k, j, S2)]
                                    if P > prob[(i, j, A)]:
                                        prob[(i, j, A)] = P
                                        backpointer[(i, j, A)] = torch.tensor([k, S1, S2])
                                        vector[(i, j, A)] = composed_vector
                binary_time += time.time() - start
                start = time.time()
                again = True
                while again:
                    again = False
                    for S in category_table[i][j]:
                        possible_cat = self.unary_rule[S]
                        if possible_cat == []:
                            continue
                        else:
                            prob_dist = self.softmax(self.linear(vector[(i, j, S)]))
                            possible_cat_prob = torch.index_select(
                                input=prob_dist, dim=-1, index=torch.tensor(possible_cat))
                            for A, P in zip(possible_cat, possible_cat_prob):
                                if A not in category_table[i][j]:
                                    category_table[i][j].append(A)
                                P = P * prob[(i, j, S)]
                                if P > prob[(i, j, A)]:
                                    prob[(i, j, A)] = P
                                    backpointer[(i, j, A)] = torch.tensor([0, S, 0])
                                    vector[(i, j, A)] = vector[(i, j, S)]
                                    again = True
                    unary_time += time.time() - start
                    start = time.time()
                    category_table = self.cut_off(category_table, prob, i, j)
                    cut_off_time += time.time() - start
        print(binary_time, unary_time, cut_off_time)
        node_list = self.reconstruct_tree(category_table, backpointer, n)
        return node_list

    # remove the candidate of low probability for beam search
    @ torch.no_grad()
    def cut_off(self, category_table, prob, i, j, width=5):
        if len(category_table[i][j]) > width:
            top_5_cat = torch.topk(prob[i][j], k=width)[1]
            category_table[i][j] = list(top_5_cat)
        return category_table

    @ torch.no_grad()
    def reconstruct_tree(self, category_table, backpointer, n):
        waiting_node_list = []
        node_list = []
        # when parsing was completed
        if category_table[0][n] != []:
            top_cat = category_table[0][n][0].item()
            if torch.any(backpointer[(0, n, top_cat)]):
                waiting_node_list.append((0, n, top_cat))
                while waiting_node_list != []:
                    node_info = waiting_node_list.pop()
                    node_list.append(node_info)
                    start_idx = node_info[0]
                    end_idx = node_info[1]
                    cat = node_info[2]
                    child_cat_info = backpointer[(start_idx, end_idx, cat)]
                    divide_idx = child_cat_info[0].item()
                    left_child_cat = child_cat_info[1].item()
                    right_child_cat = child_cat_info[2].item()
                    # when one child
                    if divide_idx == 0:
                        child_info = (start_idx, end_idx, left_child_cat)
                        # when the node is not leaf
                        if torch.any(backpointer[child_info]):
                            waiting_node_list.append(child_info)
                    # when two children
                    else:
                        left_child_info = (start_idx, divide_idx, left_child_cat)
                        right_child_info = (divide_idx, end_idx, right_child_cat)
                        # when the node is not leaf
                        if torch.any(backpointer[left_child_info]):
                            waiting_node_list.append(left_child_info)
                        # when the node is not leaf
                        if torch.any(backpointer[right_child_info]):
                            waiting_node_list.append(right_child_info)
            else:
                node_list.append((0, n, top_cat))
        return node_list

    @ torch.no_grad()
    def cal_f1_score(self, pred_node_list, correct_node_list):
        if len(pred_node_list) != 0:
            precision = 0.0
            for node in pred_node_list:
                if node in correct_node_list:
                    precision += 1.0
            precision = precision / len(pred_node_list)

            recall = 0.0
            for node in correct_node_list:
                if node in pred_node_list:
                    recall += 1.0
            recall = recall / len(correct_node_list)
            f1 = (2 * precision * recall) / (precision + recall + 1e-10)
        # when failed parsing
        else:
            f1 = 0.0
            precision = 0.0
            recall = 0.0
        return f1, precision, recall

    @ torch.no_grad()
    def validation(self, test_sentence, test_tree_list, max_length=50):
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        for sentence, tree in zip(test_sentence, test_tree_list.tree_list):
            sentence = sentence.rstrip()
            if len(sentence.split()) <= max_length:
                print(sentence)
                correct_node_list = tree.make_correct_node_list()
                pred_node_list = self.parse(sentence, len(test_tree_list.category_vocab))
                score = self.cal_f1_score(pred_node_list, correct_node_list)
                print('f1:{}, precicion:{}, recall:{}'.format(score[0], score[1], score[2]))
                f1 += score[0]
                precision += score[1]
                recall += score[2]
        self.f1 = f1 / len(test_sentence)
        self.precision = precision / len(test_sentence)
        self.recall = recall / len(test_sentence)
        return self.f1, self.precision, self.recall

    @ torch.no_grad()
    def export_stat_list(self, path):
        stat_list = []
        stat_list.append(str(self.f1))
        stat_list.append(str(self.precision))
        stat_list.append(str(self.recall))
        with open(path, 'a') as f:
            f.write(', '.join(stat_list) + '\n\n')

    @ torch.no_grad()
    def print_stat(self):
        print('f1: {}'.format(self.f1))
        print('precision: {}'.format(self.precision))
        print('recall: {}'.format(self.recall))


def extract_rule(path_to_grammar, category_vocab):
    binary_rule = [[[] for i in range(len(category_vocab))] for j in range(len(category_vocab))]
    unary_rule = [[] for i in range(len(category_vocab))]

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    for rule in data:
        tokens = rule.split()
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]]
            left_cat = category_vocab[tokens[4]]
            right_cat = category_vocab[tokens[5]]
            binary_rule[left_cat][right_cat].append(parent_cat)
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]]
            child_cat = category_vocab[tokens[4]]
            unary_rule[child_cat].append(parent_cat)
    return binary_rule, unary_rule
