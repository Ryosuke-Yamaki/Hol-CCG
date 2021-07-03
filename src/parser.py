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
    def parse(self, sentence, mergin=10):
        prob = {}
        backpointer = {}
        vector = {}

        vector_list = self.tokenize(sentence)
        n = len(vector_list)

        for i in range(n):
            output = self.softmax(self.linear(vector_list[i]))
            predict = torch.topk(output, k=mergin)
            key = (i, i + 1)
            prob[key] = {}
            backpointer[key] = {}
            vector[key] = {}
            for P, A in zip(predict[0], predict[1]):
                P = P.item()
                A = A.item()
                prob[key][A] = P
                backpointer[key][A] = None
                vector[key][A] = vector_list[i]
        binary_time = 0
        unary_time = 0
        cut_off_time = 0
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                start = time.time()
                j = i + length
                key = (i, j)
                for k in range(i + 1, j):
                    if (i, k) in prob and (k, j) in prob:
                        for S1 in prob[(i, k)].keys():
                            for S2 in prob[(k, j)].keys():
                                composed_vector = single_circular_correlation(
                                    vector[(i, k)][S1], vector[(k, j)][S2])
                                prob_dist = self.softmax(self.linear(composed_vector))
                                # set candidate of composed category based on probability
                                candidate = torch.topk(prob_dist, k=mergin)[1]
                                # list of gramatically possible category
                                possible_cat = self.binary_rule.get((S1, S2))
                                if possible_cat is not None:
                                    num_match_cat = 0
                                    for A in candidate:
                                        A = A.item()
                                        if A in possible_cat:
                                            num_match_cat += 1
                                            P = prob_dist[A] * prob[(i, k)][S1] * prob[(k, j)][S2]
                                            P = P.item()
                                            if key not in prob:
                                                prob[key] = {A: P}
                                                backpointer[key] = {A: (k, S1, S2)}
                                                vector[key] = {A: composed_vector}
                                            elif A not in prob[key] or P > prob[key][A]:
                                                prob[key][A] = P
                                                backpointer[key][A] = (k, S1, S2)
                                                vector[key][A] = composed_vector
                                            if num_match_cat == len(possible_cat):
                                                break

                binary_time += time.time() - start
                if key in prob:
                    start = time.time()
                    again = True
                    while again:
                        again = False
                        for S in list(prob[key].keys()):
                            temp_vector = vector[key][S]
                            prob_dist = self.softmax(self.linear(temp_vector))
                            # set candidate of composed category based on probability
                            candidate = torch.topk(prob_dist, k=mergin)[1]
                            # list of gramatically possible category
                            possible_cat = self.unary_rule.get(S)
                            if possible_cat is not None:
                                num_match_cat = 0
                                for A in candidate:
                                    A = A.item()
                                    if A in possible_cat:
                                        num_match_cat += 1
                                        P = prob_dist[A] * prob[key][S]
                                        P = P.item()
                                        if A not in prob[key] or P > prob[key][A]:
                                            prob[key][A] = P
                                            backpointer[key][A] = (None, S, None)
                                            vector[key][A] = temp_vector
                                            again = True
                                        if num_match_cat == len(possible_cat):
                                            break
                    unary_time += time.time() - start
                    start = time.time()
                    prob, backpointer, vector = self.cut_off(prob, backpointer, vector, key)
                    cut_off_time += time.time() - start
        # print(binary_time, unary_time, cut_off_time)
        node_list = self.reconstruct_tree(prob, backpointer, n)
        return node_list

    # remove the candidate of low probability for beam search
    @torch.no_grad()
    def cut_off(self, prob, backpointer, vector, key, width=5):
        prediction = sorted(prob[key].items(), key=lambda x: x[1], reverse=True)
        top_prob = {}
        top_backpointer = {}
        top_vector = {}
        for idx in range(min(width, len(prediction))):
            cat = prediction[idx][0]
            top_prob[cat] = prob[key][cat]
            top_backpointer[cat] = backpointer[key][cat]
            top_vector[cat] = vector[key][cat]
        prob[key] = top_prob
        backpointer[key] = top_backpointer
        vector[key] = top_vector
        return prob, backpointer, vector

    @torch.no_grad()
    def reconstruct_tree(self, prob, backpointer, len_sentence):
        waiting_node_list = []
        node_list = []
        # when parsing was completed
        if (0, len_sentence) in prob:
            top_cat = list(prob[(0, len_sentence)].items())[0][0]
            node_list.append((0, len_sentence, top_cat))
            if backpointer[(0, len_sentence)][top_cat] is not None:
                child_cat_info = backpointer[(0, len_sentence)][top_cat]
                divide_idx = child_cat_info[0]
                left_child_cat = child_cat_info[1]
                right_child_cat = child_cat_info[2]
                if divide_idx is None:
                    child_info = (0, len_sentence, left_child_cat)
                    waiting_node_list.append(child_info)
                else:
                    left_child_info = (0, divide_idx, left_child_cat)
                    right_child_info = (divide_idx, len_sentence, right_child_cat)
                    waiting_node_list.append(left_child_info)
                    waiting_node_list.append(right_child_info)

            while len(waiting_node_list) != 0:
                node_info = waiting_node_list.pop()
                node_list.append(node_info)
                start_idx = node_info[0]
                end_idx = node_info[1]
                cat = node_info[2]
                if backpointer[(start_idx, end_idx)][cat] is not None:
                    child_cat_info = backpointer[(start_idx, end_idx)][cat]
                    divide_idx = child_cat_info[0]
                    left_child_cat = child_cat_info[1]
                    right_child_cat = child_cat_info[2]
                    if divide_idx is None:
                        child_info = (start_idx, end_idx, left_child_cat)
                        waiting_node_list.append(child_info)
                    else:
                        left_child_info = (start_idx, divide_idx, left_child_cat)
                        right_child_info = (divide_idx, end_idx, right_child_cat)
                        waiting_node_list.append(left_child_info)
                        waiting_node_list.append(right_child_info)
        return node_list

    @torch.no_grad()
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
            f1 = (2 * precision * recall) / (precision + recall + 1e-6)
        # when failed parsing
        else:
            f1 = 0.0
            precision = 0.0
            recall = 0.0
        return f1, precision, recall

    @torch.no_grad()
    def validation(self, test_sentence, test_tree_list, max_length=50):
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        for sentence, tree in zip(test_sentence, test_tree_list.tree_list):
            sentence = sentence.rstrip()
            if len(sentence.split()) <= max_length:
                print(sentence)
                correct_node_list = tree.make_correct_node_list()
                pred_node_list = self.parse(sentence)
                score = self.cal_f1_score(pred_node_list, correct_node_list)
                print('f1:{}, precicion:{}, recall:{}'.format(score[0], score[1], score[2]))
                f1 += score[0]
                precision += score[1]
                recall += score[2]
        self.f1 = f1 / len(test_sentence)
        self.precision = precision / len(test_sentence)
        self.recall = recall / len(test_sentence)
        return self.f1, self.precision, self.recall

    @torch.no_grad()
    def export_stat_list(self, path):
        stat_list = []
        stat_list.append(str(self.f1))
        stat_list.append(str(self.precision))
        stat_list.append(str(self.recall))
        with open(path, 'a') as f:
            f.write(', '.join(stat_list) + '\n\n')

    @torch.no_grad()
    def print_stat(self):
        print('f1: {}'.format(self.f1))
        print('precision: {}'.format(self.precision))
        print('recall: {}'.format(self.recall))


def extract_rule(path_to_grammar, category_vocab):
    binary_rule = {}
    unary_rule = {}

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    for rule in data:
        tokens = rule.split()
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]]
            left_cat = category_vocab[tokens[4]]
            right_cat = category_vocab[tokens[5]]
            if (left_cat, right_cat) in binary_rule:
                binary_rule[(left_cat, right_cat)].append(parent_cat)
            else:
                binary_rule[(left_cat, right_cat)] = [parent_cat]
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]]
            child_cat = category_vocab[tokens[4]]
            if child_cat in unary_rule:
                unary_rule[child_cat].append(parent_cat)
            else:
                unary_rule[child_cat] = [parent_cat]

    return binary_rule, unary_rule
