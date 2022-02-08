from utils import Condition_Setter
from collections import Counter
from sys import path
from utils import load


class Combinator:
    def __init__(self, tree_list, min_freq=1):
        self.head_info = tree_list.head_info
        self.binary_counter = Counter()
        self.binary_rule = {}
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if not node.is_leaf:
                    left_child_node = tree.node_list[node.left_child_node_id]
                    right_child_node = tree.node_list[node.right_child_node_id]
                    left = left_child_node.category.split('-->')[-1]
                    right = right_child_node.category.split('-->')[-1]
                    parent = node.category
                    self.binary_counter[(left, right, parent)] += 1
        for key, freq in self.binary_counter.items():
            if freq >= min_freq:
                if (key[0], key[1]) in self.binary_rule:
                    self.binary_rule[(key[0], key[1])].append(key[2])
                else:
                    self.binary_rule[(key[0], key[1])] = [key[2]]
