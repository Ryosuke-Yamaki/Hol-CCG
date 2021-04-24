import torch
from models import Tree_List, Tree_Net, Condition_Setter
from utils import load_weight_matrix, circular_correlation
import numpy as np


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
    def __init__(self, ccg_category_list, content_to_id, parsing_info, weight_matrix):
        self.ccg_category_list = ccg_category_list
        self.category_info = ccg_category_list.category_info
        self.content_to_id = content_to_id
        self.parsing_info = parsing_info
        self.weight_matrix = weight_matrix

    def parse(self, sentence):
        chart = {}
        sentence = sentence.split(' ')
        n = len(sentence)
        # set the cells of leaf nodes with their vectors and possible category_id
        for i in range(n):
            word = sentence[i]
            word_id = self.content_to_id[word]
            cell = {}
            cell['vector'] = self.weight_matrix[word_id]
            cell['possible_category_id_list'] = self.parsing_info[word_id]
            cell['possible_category'] = []
            for category_id in cell['possible_category_id_list']:
                cell['possible_category'].append(self.ccg_category_list.id_to_category[category_id])
            chart[(i, i + 1)] = cell

        for l in range(2, n + 1):
            for i in range(n - l + 1):
                j = i + l
                cell = {}
                cell['possible_category_id_list'] = []
                for k in range(i + 1, j):
                    left_cell = chart[(i, k)]
                    right_cell = chart[(k, j)]
                    if left_cell is not None and right_cell is not None:
                        # set parent node's vector
                        # cell['vector'] = circular_correlation(
                        # left_cell['vector'], right_cell['vector'], True)

                        for left_category_id in left_cell['possible_category_id_list']:
                            left_category = self.category_info[left_category_id]
                            for right_category_id in right_cell['possible_category_id_list']:
                                right_category = self.category_info[right_category_id]
                                possible_category_id = self.compose_categories(
                                    left_category, right_category)
                                if possible_category_id is not None:
                                    cell['possible_category_id_list'].append(possible_category_id)

                if cell['possible_category_id_list'] == []:
                    chart[(i, j)] = None
                else:
                    cell['possible_category'] = []
                    for category_id in cell['possible_category_id_list']:
                        cell['possible_category'].append(
                            self.ccg_category_list.id_to_category[category_id])
                    chart[(i, j)] = cell
        return chart, chart[(0, n)]['possible_category']

    def compose_categories(self, left_category, right_category):
        if left_category.direction_of_slash == 'R'\
                and left_category.sibling_category_id == right_category.self_category_id:
            return left_category.parent_category_id
        elif right_category.direction_of_slash == 'L'\
                and right_category.sibling_category_id == left_category.self_category_id:
            return right_category.parent_category_id


PATH_TO_DIR = "/home/yryosuke0519/Hol-CCG/"

condition = Condition_Setter(PATH_TO_DIR)

train_tree_list = Tree_List(condition.path_to_train_data, condition.REGULARIZED)
test_tree_list = Tree_List(condition.path_to_test_data, condition.REGULARIZED)
# use same vocablary and category as train_tree_list
test_tree_list.content_to_id = train_tree_list.content_to_id
test_tree_list.category_to_id = train_tree_list.category_to_id
test_tree_list.id_to_content = train_tree_list.id_to_content
test_tree_list.id_to_category = train_tree_list.id_to_category
test_tree_list.set_content_category_id()
test_tree_list.set_info_for_training()
test_tree_list.set_info_for_parsing()

weight_matrix = torch.tensor(
    load_weight_matrix(
        condition.path_to_initial_weight_matrix,
        condition.REGULARIZED))
tree_net = Tree_Net(test_tree_list, weight_matrix)
tree_net.load_state_dict(torch.load(condition.path_to_model))
tree_net.eval()

ccg_category_list = CCG_Category_List(test_tree_list)

weight_matrix = tree_net.embedding.weight.detach()
parser = Parser(
    ccg_category_list,
    test_tree_list.content_to_id,
    test_tree_list.parsing_info,
    weight_matrix)
for tree in test_tree_list.tree_list:
    chart, possible_category = parser.parse(tree.sentence)

    print('{}:{}'.format(tree.self_id, possible_category))


for tree in test_tree_list.tree_list[1:3]:
    output = tree_net(tree.leaf_node_info, tree.composition_info)
    id_to_category = test_tree_list.id_to_category
    n = 10
    for node_id in range(len(tree.node_list)):
        node = tree.node_list[node_id]
        # if node.is_leaf:
        node.prob_dist = output[node_id]
        if node.is_leaf:
            for idx in range(len(node.prob_dist)):
                if idx not in test_tree_list.parsing_info[node.content_id]:
                    node.prob_dist[idx] = 0.0
            node.prob_dist = node.prob_dist / torch.norm(node.prob_dist)
        node.top_n_category_id = torch.argsort(node.prob_dist, descending=True)[:n]
        # node.top_n_category_id = np.argsort(node.prob_dist)[::-1][:n]
        # node.top_n_category_id = node.top_n_].detach().numpy().copy()
    for node in tree.node_list:
        print(node.content)
        if node.category_id in node.top_n_category_id:
            print('True category is included: {}'.format(node.category))
        else:
            print('True category is not included: {}'.format(node.category))
        for category_id in node.top_n_category_id:
            print('{}: {}'.format(
                id_to_category[int(category_id)], node.prob_dist[int(category_id)]))
        print('')
    print('*' * 50)

# tree_idx = 0
# for tree in train_tree_list.tree_list:
#     print('*' * 50)
#     print('tree_idx = {}'.format(tree_idx))
#     node_pair_list = tree.make_node_pair_list()
#     while True:
#         i = 0
#         for node_pair in node_pair_list:
#             left_node = node_pair[0]
#             right_node = node_pair[1]
#             parent_node = tree.node_list[left_node.parent_id]
#             if left_node.ready and right_node.ready:
#                 left_category = CCG_Category(left_node.category)
#                 right_category = CCG_Category(right_node.category)
#                 composed_category = compose_categories(left_category, right_category)
#                 parent_node.category = composed_category
#                 parent_node.content = left_node.content + ' ' + right_node.content
#                 parent_node.ready = True
#                 print('{} + {} ---> {}'.format(left_node.content,
#                                                right_node.content, parent_node.content))
#                 print('{} {} ---> {}'.format(left_category.category,
#                                              right_category.category, composed_category))
#                 print('')
#                 node_pair_list.remove(node_pair)
#         if node_pair_list == []:
#             break
#     tree_idx += 1
