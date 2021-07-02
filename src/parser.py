import copy
from os import wait
import re
from numpy.core.einsumfunc import _parse_possible_contraction
import torch
import torch.nn as nn
from utils import circular_correlation, single_circular_correlation


class CCG_Category:
    def __init__(self, category, category_to_id):
        self.self_category_id = category_to_id[category]
        if '(' in category:  # for the complex categories including brackets
            self.self_category = category[1:-1]
        else:  # for the primitive categories like NP or S
            self.self_category = category
        self.set_composition_info(self.self_category, category_to_id)

    def set_composition_info(self, category, category_to_id):
        level = 0
        for idx in range(len(category)):
            char = category[idx]
            if char == '(':
                level += 1
            elif char == ')':
                level -= 1
            if level == 0:
                if char == '\\':
                    self.direction_of_slash = 'L'
                    self.parent_category = category[:idx]
                    self.parent_category_id = category_to_id[self.parent_category]
                    self.sibling_category = category[idx + 1:]
                    self.sibling_category_id = category_to_id[self.sibling_category]
                    return
                elif char == '/':
                    self.direction_of_slash = 'R'
                    self.parent_category = category[:idx]
                    self.parent_category_id = category_to_id[self.parent_category]
                    self.sibling_category = category[idx + 1:]
                    self.sibling_category_id = category_to_id[self.sibling_category]
                    return
        self.direction_of_slash = None
        self.parent_category = None
        self.parent_category_id = None
        self.sibling_category = None
        self.sibling_category_id = None


class Category_info:
    def __init__(self, vector, prob, backpointer):
        self.vector = vector
        self.prob = prob
        self.backpointer = backpointer


class Cell:
    def __init__(self, start_idx, end_idx):
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.category_list = {}

    def add_category(self, category_id, vector, prob, backpointer=None):
        self.category_list[category_id] = Category_info(vector, prob, backpointer)


class CCG_Category_List:
    def __init__(self, tree_list):
        self.id_to_category = tree_list.id_to_category
        self.category_to_id = tree_list.category_to_id
        # the dictionary includes the information of each CCG category, key is category_id
        self.set_category_info()

    def set_category_info(self):
        self.category_info = {}
        for category_id, category in self.id_to_category.items():
            self.category_info[category_id] = CCG_Category(category, self.category_to_id)


class Parser:
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
    def parse(self, sentence, k=5):
        prob = {}
        backpointer = {}
        vector = {}

        vector_list = self.tokenize(sentence)
        n = len(vector_list)

        for i in range(n):
            output = self.softmax(self.linear(vector_list[i]))
            predict = torch.topk(output, k=k)
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

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                key = (i, j)
                for k in range(i + 1, j):
                    if (i, k) in prob and (k, j) in prob:
                        for S1 in prob[(i, k)].keys():
                            for S2 in prob[(k, j)].keys():
                                composed_vector = single_circular_correlation(
                                    vector[(i, k)][S1], vector[(k, j)][S2])
                                prob_dist = self.softmax(self.linear(composed_vector))
                                if self.binary_rule.get((S1, S2)) is not None:
                                    for A in self.binary_rule.get((S1, S2)):
                                        P = prob_dist[A] * prob[(i, k)][S1] * prob[(k, j)][S2]
                                        if key not in prob:
                                            prob[key] = {A: P}
                                            backpointer[key] = {A: (k, S1, S2)}
                                            vector[key] = {A: composed_vector}
                                        elif A not in prob[key] or P > prob[key][A]:
                                            prob[key][A] = P
                                            backpointer[key][A] = (k, S1, S2)
                                            vector[key][A] = composed_vector
                if key in prob:
                    again = True
                    while again:
                        again = False
                        for S in list(prob[key].keys()):
                            temp_vector = vector[key][S]
                            prob_dist = self.softmax(self.linear(temp_vector))
                            if self.unary_rule.get(S) is not None:
                                for A in self.unary_rule.get(S):
                                    P = prob_dist[A] * prob[key][S]
                                    if A not in prob[key] or P > prob[key][A]:
                                        prob[key][A] = P
                                        backpointer[key][A] = (None, S, None)
                                        vector[key][A] = temp_vector
                                        again = True
                    prob, backpointer, vector = self.cut_off(prob, backpointer, vector, key)
        node_list = self.recunstruct_tree(prob, backpointer, n)
        return node_list

    # remove the candidate of low probability for beam search
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

    def recunstruct_tree(self, prob, backpointer, len_sentence):
        waiting_node_list = []
        node_list = []
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

    def make_leaf_node_cell(self, vector):
        output = self.linear_classifier(vector)
        cell = {}
        for i in range(len(output)):
            if output[i] >= self.THRESHOLD:
                pred_category_info = {}
                pred_category_info['vector'] = vector
                pred_category_info['prob'] = output[i]
                pred_category_info['category'] = self.ccg_category_list.id_to_category[i]
                cell[i] = pred_category_info
        return cell

    def validation(self):
        f1 = 0.0
        precision = 0.0
        recall = 0.0
        for tree in self.tree_list.tree_list:
            chart = self.parse(tree.sentence)
            parsed_tree = Parsed_Tree(tree.sentence, chart, self.tree_list.id_to_category)
            score = parsed_tree.cal_f1_score(tree.convert_node_list_for_eval())
            f1 += score[0]
            precision += score[1]
            recall += score[2]
            self.chart_list.append((chart, score[0]))
        self.f1 = f1 / len(self.tree_list.tree_list)
        self.precision = precision / len(self.tree_list.tree_list)
        self.recall = recall / len(self.tree_list.tree_list)
        return self.f1, self.precision, self.recall

    def export_stat_list(self, path):
        stat_list = []
        stat_list.append(str(self.f1))
        stat_list.append(str(self.precision))
        stat_list.append(str(self.recall))
        with open(path, 'a') as f:
            f.write(', '.join(stat_list) + '\n\n')

    def print_stat(self):
        print('f1: {}'.format(self.f1))
        print('precision: {}'.format(self.precision))
        print('recall: {}'.format(self.recall))


class Node:
    def __init__(self, self_id, scope, content, cell, category_id=None, top=False):
        self.self_id = self_id
        self.scope = scope
        self.content = content
        self.cell = cell
        self.find_max_prob()
        if top:
            self.category_id = self.max_prob_category_id
        else:
            self.category_id = category_id

    def find_max_prob(self):
        self.max_prob = 0.0
        for possible_category_id, possible_category_info in self.cell.items():
            if possible_category_info['prob'] > self.max_prob:
                self.max_prob = possible_category_info['prob']
                self.max_prob_category_id = possible_category_id

    def extract_back_pointer(self):
        if 'back_pointer' in self.cell[self.category_id]:
            backpointer = self.cell[self.category_id]['back_pointer']
            left_pointer = backpointer[0]
            right_pointer = backpointer[1]
            return left_pointer, right_pointer
        else:
            return None, None


class Parsed_Tree:
    def __init__(self, sentence, chart, id_to_category):
        self.sentence = sentence
        self.length_of_sentence = len(sentence.split())
        self.chart = chart
        self.sentence = sentence.split()
        self.id_to_category = id_to_category
        self.node_list = []
        self.pointers_before_define = []
        self.define_top_node()
        self.define_other_nodes()
        self.convert_node_list_for_eval()

    def define_top_node(self):
        if self.chart[(0, self.length_of_sentence)] is not None:
            top_node = Node(0, (0, self.length_of_sentence), (' ').join(self.sentence),
                            self.chart[(0, self.length_of_sentence)], top=True)
            left_pointer, right_pointer = top_node.extract_back_pointer()
            self.node_list.append(top_node)
            self.pointers_before_define.append(left_pointer)
            self.pointers_before_define.append(right_pointer)
        # when parsing is not done well
        else:
            top_node = None
            self.node_list.append(top_node)

    def define_other_nodes(self):
        node_id = 1
        if self.pointers_before_define != []:
            while True:
                pointer = self.pointers_before_define.pop(0)
                node = Node(node_id,
                            scope=pointer[:2],
                            content=(' ').join(self.sentence[pointer[0]:pointer[1]]),
                            cell=self.chart[pointer[:2]],
                            category_id=pointer[2])
                left_pointer, right_pointer = node.extract_back_pointer()
                self.node_list.append(node)
                if left_pointer is not None and right_pointer is not None:
                    self.pointers_before_define.append(left_pointer)
                    self.pointers_before_define.append(right_pointer)
                    node_id += 1
                if self.pointers_before_define == []:
                    break

    def convert_node_list_for_eval(self):
        converted_node_list = []
        if len(self.node_list) > 1:
            for node in self.node_list:
                scope = node.scope
                category_id = node.category_id
                converted_node_list.append((scope[0], scope[1], category_id))
            self.converted_node_list = converted_node_list
        else:
            self.converted_node_list = None

    def cal_f1_score(self, correct_node_list):
        pred_node_list = self.converted_node_list
        if pred_node_list is not None:
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
            # avoid zero division
            if precision == 0.0 and recall == 0.0:
                f1 = 0.0
            else:
                f1 = (2 * precision * recall) / (precision + recall)
        # when failed parsing
        else:
            f1 = 0.0
            precision = 0.0
            recall = 0.0

        return f1, precision, recall

    def visualize_parsing_result(self):
        for node in self.node_list:
            print('content: {}'.format(node.content))
            print('category :{}'.format(self.id_to_category[node.category_id]))
            print()


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
