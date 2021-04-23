import torch
from models import Tree_List, Tree_Net, Condition_Setter
from utils import load_weight_matrix
import numpy as np


class CCG_Category:
    def __init__(self, category):
        # for the complex categories including brackets
        if '(' in category:
            self.category = category[1:-1]
        # for the primitive categories like NP or S
        else:
            self.category = category
        # self.category = category
        self.set_composition_info(self.category)

    def set_composition_info(self, category):
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
                    self.sibling_category = category[idx + 1:]
                    return
                elif char == '/':
                    self.direction_of_slash = 'R'
                    self.parent_category = category[:idx]
                    self.sibling_category = category[idx + 1:]
                    return
        self.direction_of_slash = None
        self.parent_category = None
        self.sibling_category = None


def compose_categories(left_category, right_category):
    if left_category.direction_of_slash == 'R' and left_category.sibling_category == right_category.category:
        return left_category.parent_category
    elif left_category.direction_of_slash == 'R' and left_category.sibling_category == '(' + right_category.category + ')':
        return left_category.parent_category
    elif right_category.direction_of_slash == 'L' and right_category.sibling_category == left_category.category:
        return right_category.parent_category
    elif right_category.direction_of_slash == 'L' and right_category.sibling_category == '(' + left_category.category + ')':
        return right_category.parent_category


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

weight_matrix = torch.tensor(
    load_weight_matrix(
        condition.path_to_initial_weight_matrix,
        condition.REGULARIZED))
tree_net = Tree_Net(test_tree_list, weight_matrix)
tree_net.load_state_dict(torch.load(condition.path_to_model))
tree_net.eval()

for tree in test_tree_list.tree_list[1:3]:
    output = tree_net(tree.leaf_node_info, tree.composition_info)
    id_to_category = test_tree_list.id_to_category
    n = 10
    for node_id in range(len(tree.node_list)):
        node = tree.node_list[node_id]
        # if node.is_leaf:
        node.prob_dist = output[node_id].detach().numpy().copy()
        node.top_n_category_id = np.argsort(node.prob_dist)[::-1][:n]

    for node in tree.node_list:
        print(node.content)
        if node.category_id in node.top_n_category_id:
            print('True category is included: {}'.format(node.category))
        else:
            print('True category is not included: {}'.format(node.category))
        for category_id in node.top_n_category_id:
            print('{}: {}'.format(id_to_category[category_id], node.prob_dist[category_id]))
        print('')
    print('*' * 50)

# for node in tree.node_list:
#     if node.is_leaf:
#         print(node.content)
#         for category_id in node.top_n_category_id:
#             print('{}:{}'.format(inv_category[category_id], node.prob_dist[category_id]))


# for tree in train_tree_list.tree_list:
#     for node in tree.node_list:
#         if not node.is_leaf:
#             node.category = None
#             print(node.category)

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
