import torch.nn as nn
from utils import circular_correlation


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
    def __init__(
            self,
            tree_list,
            ccg_category_list,
            weight_matrix,
            linear_classifier,
            THRESHOLD):
        self.tree_list = tree_list
        self.ccg_category_list = ccg_category_list
        self.category_info = ccg_category_list.category_info
        self.content_to_id = tree_list.content_to_id
        self.weight_matrix = weight_matrix
        self.linear_classifier = linear_classifier
        self.THRESHOLD = THRESHOLD
        self.chart_list = []

    def parse(self, sentence):
        chart = {}
        sentence = sentence.split(' ')
        n = len(sentence)
        # set the cells of leaf nodes with their vectors and possible category_id
        for i in range(n):
            word = sentence[i]
            word_id = self.content_to_id[word]
            vector = self.weight_matrix[word_id]
            cell = self.make_leaf_node_cell(vector)
            chart[(i, i + 1)] = cell

        for length in range(2, n + 1):
            for i in range(n - length + 1):
                j = i + length
                cell = {}
                for k in range(i + 1, j):
                    left_cell = chart[(i, k)]
                    right_cell = chart[(k, j)]
                    if left_cell is not None and right_cell is not None:
                        # extract category_id of left cell
                        for left_category_id in left_cell.keys():
                            # define ccg category in left cell
                            left_category = self.category_info[left_category_id]
                            # extract category_id of right cell
                            for right_category_id in right_cell.keys():
                                # define ccg category in right cell
                                right_category = self.category_info[right_category_id]
                                # output possible category id from left and right ccg category
                                possible_category_id = self.compose_categories(
                                    left_category, right_category)
                                if possible_category_id is not None:
                                    possible_category_info = {}
                                    left_vector = left_cell[left_category_id]['vector']
                                    left_prob = left_cell[left_category_id]['prob']
                                    right_vector = right_cell[right_category_id]['vector']
                                    right_prob = right_cell[right_category_id]['prob']
                                    vector = circular_correlation(left_vector, right_vector, True)
                                    prob = self.linear_classifier(vector)[possible_category_id]
                                    possible_category_info['vector'] = vector
                                    possible_category_info['prob'] = prob * left_prob * right_prob
                                    possible_category_info['back_pointer'] = (
                                        (i, k, left_category_id), (k, j, right_category_id))
                                    # when the category is subscribed to cell first time
                                    if possible_category_id not in cell:
                                        cell[possible_category_id] = possible_category_info
                                    # when the probability is higher than existing one
                                    elif possible_category_info['prob'] > \
                                            cell[possible_category_id]['prob']:
                                        cell[possible_category_id] = possible_category_info
                # when composition candidate founded
                if len(cell) > 0:
                    chart[(i, j)] = cell
                # when no composition candidate founded
                else:
                    chart[(i, j)] = None
        return chart

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

    def compose_categories(self, left_category, right_category):
        if left_category.direction_of_slash == 'R'\
                and left_category.sibling_category_id == right_category.self_category_id:
            return left_category.parent_category_id
        elif right_category.direction_of_slash == 'L'\
                and right_category.sibling_category_id == left_category.self_category_id:
            return right_category.parent_category_id

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

    def print_stat(self):
        print('f1: {}'.format(self.f1))
        print('precision: {}'.format(self.precision))
        print('recall: {}'.format(self.recall))


class Linear_Classifier(nn.Module):
    def __init__(self, tree_net):
        super(Linear_Classifier, self).__init__()
        self.linear = tree_net.linear
        self.sigmoid = tree_net.sigmoid

    def forward(self, vector):
        output = self.sigmoid(self.linear(vector))
        return output


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
