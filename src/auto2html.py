from models import Tree_List
import os
import argparse


class Converter:
    def __init__(self, path_to_data):
        self.load_sentence(path_to_data)

    def load_sentence(self, path_to_data):
        self.sentence_list = []
        f = open(path_to_data, 'r')
        data = f.readlines()
        f.close()
        for i in range(len(data)):
            if i % 2 == 1:
                self.sentence_list.append(data[i])

    def convert(self, sentence):
        self.comfirmed_node = []
        stack_dict = {}
        idx = 0
        level = 0
        node_id = 0
        node_info, idx = self.extract_node(sentence, idx)
        root_node = Node(node_info)
        stack_dict[level] = Node_Stack()
        stack_dict[level].push(root_node)
        if not root_node.is_leaf:
            stack_dict[level + 1] = Node_Stack(capacity=root_node.num_child)

        while True:
            char = sentence[idx]
            if char == '(':
                level += 1
                node_info, idx = self.extract_node(sentence, idx)
                node = Node(node_info)
                stack_dict[level].push(node)
                if not node.is_leaf:
                    stack_dict[level + 1] = Node_Stack(capacity=node.num_child)
            elif char == ')':
                level -= 1
            idx += 1
            stack_dict, node_id = self.search_parent_child_relation(stack_dict, node_id)
            if idx == len(sentence):
                root_node = stack_dict[0].node_stack[-1]
                root_node.self_id = node_id
                self.comfirmed_node.append(root_node)
                break

    # add node to stack corresponding to the current level, and update index
    def extract_node(self, sentence, idx):
        start_idx_of_node = idx + 2
        for idx in range(start_idx_of_node, len(sentence)):
            char = sentence[idx]
            if char == '>':
                end_idx_of_node = idx
                break
        node_info = sentence[start_idx_of_node:end_idx_of_node]
        return node_info.split(), idx

    def search_parent_child_relation(self, stack_dict, node_id):
        for level in range(1, len(stack_dict)):
            node_stack = stack_dict[level]
            node_stack.update_status()
            if node_stack.ready:
                parent_node = stack_dict[level - 1].node_stack[-1]
                if node_stack.capacity == 1:
                    child_node = node_stack.pop()
                    child_node.self_id = node_id
                    node_id += 1
                    parent_node.child_node_id = child_node.self_id
                    parent_node.ready = True
                    self.comfirmed_node.append(child_node)
                else:
                    right_child_node = node_stack.pop()
                    left_child_node = node_stack.pop()
                    left_child_node.self_id = node_id
                    node_id += 1
                    right_child_node.self_id = node_id
                    node_id += 1
                    parent_node.left_child_node_id = left_child_node.self_id
                    parent_node.right_child_node_id = right_child_node.self_id
                    parent_node.ready = True
                    self.comfirmed_node.append(left_child_node)
                    self.comfirmed_node.append(right_child_node)
                del stack_dict[level]
        return stack_dict, node_id

    def save_node_info(self, path_to_save):
        node_info_list = []
        for node in self.comfirmed_node:
            node_info = []
            node_info.append(str(node.is_leaf))
            node_info.append(str(node.self_id))
            if node.is_leaf:
                node_info.append(node.content)
                node_info.append(node.category)
                node_info.append(node.pos)
            else:
                node_info.append(node.category)
                node_info.append(str(node.num_child))
                if node.num_child == 1:
                    node_info.append(str(node.child_node_id))
                else:
                    node_info.append(str(node.left_child_node_id))
                    node_info.append(str(node.right_child_node_id))
                node_info.append(str(node.head))
            node_info_list.append(' '.join(node_info))
            node_info_list.append('\n')
        f = open(path_to_save, 'a')
        f.writelines(node_info_list)
        f.write('\n')
        f.close


class Node:
    def __init__(self, node_info):
        if node_info[0] == 'L':
            self.is_leaf = True
            self.ready = True
            self.content = node_info[4]
            self.category = node_info[1]
            self.pos = node_info[2]
        else:
            self.is_leaf = False
            self.ready = False
            self.category = node_info[1]
            self.head = int(node_info[2])
            self.num_child = int(node_info[3])

    def set_self_id(self, id):
        self.self_id = id

    def set_child_id(self, id):
        self.child_id = id


class Node_Stack:
    def __init__(self, capacity=None):
        self.node_stack = []
        self.capacity = capacity
        self.ready = False

    def push(self, node):
        self.node_stack.append(node)

    def pop(self):
        return self.node_stack.pop(-1)

    def update_status(self):
        # when the number of nodes in node_stack equals to capacity
        if len(self.node_stack) == self.capacity:
            # check wheter all node is ready or not
            check_bit = 1
            for node in self.node_stack:
                if not node.ready:
                    check_bit = 0
                    break
            # when all nodes in the node_stack is ready
            if check_bit == 1:
                self.ready = True


def flatten(mathml):
    for i in mathml:
        if isinstance(i, list):
            yield from flatten(i)
        else:
            yield i


parser = argparse.ArgumentParser()
parser.add_argument('path_to_auto', type=str)

args = parser.parse_args()

path_to_auto = args.path_to_auto
path_to_converted = 'converted.txt'

open(path_to_converted, 'w')

converter = Converter(path_to_auto)
for sentence in converter.sentence_list:
    converter.convert(sentence)
    converter.save_node_info(path_to_converted)

tree_list = Tree_List(
    path_to_converted, type='train')

for tree in tree_list.tree_list:
    tree.root_node = tree.node_list[-1]
    tree.root_node.parent_node = None
    for node in reversed(tree.node_list):
        if not node.is_leaf:
            if node.num_child == 1:
                node.child = tree.node_list[node.child_node_id]
            elif node.num_child == 2:
                node.left_child = tree.node_list[node.left_child_node_id]
                node.right_child = tree.node_list[node.right_child_node_id]

mathml_list = []
for tree in tree_list.tree_list:
    tree.root_node.mathml = []
    waiting_nodes = [tree.root_node]
    while len(waiting_nodes) > 0:
        node = waiting_nodes.pop(0)
        if node.is_leaf:
            node.mathml += ['<mfrac><mtext>',
                            node.content[0],
                            '</mtext><mtext>',
                            node.category,
                            '</mtext></mfrac>']
        else:
            if node.num_child == 1:
                node.child.mathml = []
                node.mathml += ['<mfrac>',
                                node.child.mathml,
                                '<mtext>',
                                node.category,
                                '</mtext></mfrac>']
                if node.child.is_leaf:
                    node.child.mathml += ['<mfrac><mtext>',
                                          node.child.content[0],
                                          '</mtext><mtext>',
                                          node.child.category,
                                          '</mtext></mfrac>']
                else:
                    waiting_nodes.append(node.child)
            elif node.num_child == 2:
                node.left_child.mathml = []
                node.right_child.mathml = []
                node.mathml += ['<mfrac><mrow>',
                                node.left_child.mathml,
                                node.right_child.mathml,
                                '</mrow><mtext>',
                                node.category,
                                '</mtext></mfrac>']
                if node.left_child.is_leaf:
                    node.left_child.mathml += ['<mfrac><mtext>',
                                               node.left_child.content[0],
                                               '</mtext><mtext>',
                                               node.left_child.category,
                                               '</mtext></mfrac>']
                else:
                    waiting_nodes.append(node.left_child)
                if node.right_child.is_leaf:
                    node.right_child.mathml += ['<mfrac><mtext>',
                                                node.right_child.content[0],
                                                '</mtext><mtext>',
                                                node.right_child.category,
                                                '</mtext></mfrac>']
                else:
                    waiting_nodes.append(node.right_child)
    mathml_list.append('<p>Sentence ID={}</p><math>{}</math>'.format(tree.self_id,
                       ''.join(flatten(tree.root_node.mathml))))

html = '''\
<!doctype html>
<html lang='en'>
<head>
<meta charset='UTF-8'>
<style>
    body {{
    font-size: 1em;
    }}
</style>
<script type="text/javascript"
    src="http://cdn.mathjax.org/mathjax/latest/MathJax.js?config=TeX-AMS-MML_HTMLorMML">
</script>
</head>
<body>
{}
</body>
</html>
'''.format(''.join(mathml_list))
print(html)
