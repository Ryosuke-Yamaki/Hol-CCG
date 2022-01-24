from collections import Counter
from sys import path
from utils import load


class Combinator:
    def __init__(self, tree_list):
        self.head_info = tree_list.head_info
        self.binary_rule = {}
        for tree in tree_list.tree_list:
            for node in tree.node_list:
                if not node.is_leaf:
                    left_child_node = tree.node_list[node.left_child_node_id]
                    right_child_node = tree.node_list[node.right_child_node_id]
                    left = left_child_node.category.split('-->')[-1]
                    right = right_child_node.category.split('-->')[-1]
                    prime_parent = node.prime_category
                    parent = node.category
                    composed_parent, type = self.apply_binary(left, right)
                    # for the basic combinatory rules
                    if prime_parent == composed_parent:
                        if (left, right) in self.binary_rule:
                            if [parent, type] not in self.binary_rule[(left, right)]:
                                self.binary_rule[(left, right)].append([parent, type])
                        else:
                            self.binary_rule[(left, right)] = [[parent, type]]
                    else:
                        if (left, right) in self.binary_rule:
                            if [parent, 'binary'] not in self.binary_rule[(left, right)]:
                                self.binary_rule[(left, right)].append([parent, 'binary'])
                        else:
                            self.binary_rule[(left, right)] = [[parent, 'binary']]

    def apply_binary(self, left, right):
        left = self.get_cat_info(left)
        right = self.get_cat_info(right)
        # apply forward application
        if left['lr'] == 'right' and left['arg'] == right['cat']:
            parent = left['target']
            parent = parent
            type = 'fa'
        # apply backward application
        elif right['lr'] == 'left' and left['cat'] == right['arg']:
            parent = right['target']
            parent = parent
            type = 'ba'
        # apply forward compostion
        elif left['lr'] == 'right' and right['lr'] == 'right' and left['arg'] == right['target']:
            if '/' in left['target'] or '\\' in left['target']:
                parent = '(' + left['target'] + ')'
            else:
                parent = left['target']
            parent += '/'
            if '/' in right['arg'] or '\\' in right['arg']:
                parent += '(' + right['arg'] + ')'
            else:
                parent += right['arg']
            parent = parent
            type = 'fc'
        # apply backward composition
        elif left['lr'] == 'left' and right['lr'] == 'left' and left['target'] == right['arg']:
            if '/' in right['target'] or '\\' in right['target']:
                parent = '(' + right['target'] + ')'
            else:
                parent = right['target']
            parent += '\\'
            if '/' in left['arg'] or '\\' in left['arg']:
                parent += '(' + left['arg'] + ')'
            else:
                parent += left['arg']
            parent = parent
            type = 'bc'
        # apply backward cross composition
        elif left['lr'] == 'right' and right['lr'] == 'left' and left['target'] == right['arg']:
            if '/' in right['target'] or '\\' in right['target']:
                parent = '(' + right['target'] + ')'
            else:
                parent = right['target']
            parent += '/'
            if '/' in left['arg'] or '\\' in left['arg']:
                parent += '(' + left['arg'] + ')'
            else:
                parent += left['arg']
            parent = parent
            type = 'bxc'
        else:
            parent = None
            type = None
        return parent, type

    def get_cat_info(self, cat):
        level = 0
        is_primitive = True
        lr = None
        arg = None
        target = None
        for i in range(len(cat)):
            c = cat[i]
            if c == '(':
                level += 1
            elif c == ')':
                level -= 1
            elif (c == '\\' or c == '/') and level == 0:
                is_primitive = False
                arg = cat[i + 1:]
                target = cat[:i]
                if arg[0] == '(':
                    arg = arg[1:-1]
                if target[0] == '(':
                    target = target[1:-1]
                if c == '\\':
                    lr = 'left'
                elif c == '/':
                    lr = 'right'
        info = {'cat': cat, 'is_primitive': is_primitive, 'arg': arg, 'target': target, 'lr': lr}
        return info
