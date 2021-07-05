import torch
import subprocess
from utils import single_circular_correlation
import re


class Node:
    def __init__(self, node_info, content_vocab, category_vocab, embedding, linear):
        if node_info[0] == 'L':
            self.is_leaf = True
            self.ready = True
            content = node_info[4].lower()
            content = re.sub(r'\d+', '0', content)
            content = re.sub(r'\d,\d', '0', content)
            self.content_id = content_vocab[content]
            self.category_id = category_vocab[node_info[1]]
            self.vector = embedding(torch.tensor(self.content_id))
            self.vector = self.vector / torch.norm(self.vector)
            self.score = linear(self.vector)[self.category_id]
        else:
            self.is_leaf = False
            self.ready = False
            self.category_id = category_vocab[node_info[1]]
            self.num_child = int(node_info[3])


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


class Tree:
    def __init__(self, node_list):
        self.node_list = node_list

    def cal_score(self, linear):
        while True:
            num_ready_node = 0
            for node in self.node_list:
                if node.ready:
                    num_ready_node += 1
                elif not node.is_leaf and not node.ready:
                    if node.num_child == 1:
                        child_node = self.node_list[node.child_node_id]
                        if child_node.ready:
                            node.vector = child_node.vector
                            node.score = linear(node.vector)[node.category_id] + child_node.score
                            node.ready = True
                    else:  # when node has two children
                        left_child_node = self.node_list[node.left_child_node_id]
                        right_child_node = self.node_list[node.right_child_node_id]
                        if left_child_node.ready and right_child_node.ready:
                            node.vector = single_circular_correlation(
                                left_child_node.vector, right_child_node.vector)
                            node.score = linear(
                                node.vector)[
                                node.category_id] + left_child_node.score + right_child_node.score
                            node.ready = True
            if num_ready_node == len(self.node_list):
                break
        return self.node_list[-1].score.item()

    def make_node_list_for_eval(self):
        for node in self.node_list:
            if node.is_leaf:
                node.content_id = [node.content_id]
        self.composition_info = []
        while True:
            num_ready_node = 0
            for node in self.node_list:
                if node.ready:
                    num_ready_node += 1
                elif not node.is_leaf and not node.ready:
                    if node.num_child == 1:
                        child_node = self.node_list[node.child_node_id]
                        if child_node.ready:
                            node.content_id = child_node.content_id
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, child_node.self_id, 0])
                    else:  # when node has two children
                        left_child_node = self.node_list[node.left_child_node_id]
                        right_child_node = self.node_list[node.right_child_node_id]
                        if left_child_node.ready and right_child_node.ready:
                            node.content_id = left_child_node.content_id + right_child_node.content_id
                            node.ready = True
                            self.composition_info.append(
                                [node.num_child, node.self_id, left_child_node.self_id, right_child_node.self_id])
            if num_ready_node == len(self.node_list):
                break
        eval_node_list = []
        top_node = self.node_list[-1]
        top_node.start_idx = 0
        top_node.end_idx = len(top_node.content_id)
        eval_node_list.append((0, len(top_node.content_id), top_node.category_id))
        for info in reversed(self.composition_info):
            num_child = info[0]
            if num_child == 1:
                parent_node = self.node_list[info[1]]
                child_node = self.node_list[info[2]]
                child_node.start_idx = parent_node.start_idx
                child_node.end_idx = parent_node.end_idx
                if not child_node.is_leaf:
                    eval_node_list.append(
                        (child_node.start_idx, child_node.end_idx, child_node.category_id))
            else:
                parent_node = self.node_list[info[1]]
                left_child_node = self.node_list[info[2]]
                right_child_node = self.node_list[info[3]]
                left_child_node.start_idx = parent_node.start_idx
                left_child_node.end_idx = parent_node.start_idx + len(left_child_node.content_id)
                right_child_node.start_idx = left_child_node.end_idx
                right_child_node.end_idx = parent_node.end_idx
                if not left_child_node.is_leaf:
                    eval_node_list.append(
                        (left_child_node.start_idx,
                         left_child_node.end_idx,
                         left_child_node.category_id))
                if not right_child_node.is_leaf:
                    eval_node_list.append(
                        (right_child_node.start_idx,
                         right_child_node.end_idx,
                         right_child_node.category_id))
        return eval_node_list


class Converter:
    def __init__(self, content_vocab, category_vocab, embedding, linear):
        self.content_vocab = content_vocab
        self.category_vocab = category_vocab
        self.embedding = embedding
        self.linear = linear

    def convert_to_tree(self, sentence):
        self.comfirmed_node = []
        stack_dict = {}
        idx = 0
        level = 0
        node_id = 0
        node_info, idx = self.extract_node(sentence, idx)
        root_node = Node(
            node_info,
            self.content_vocab,
            self.category_vocab,
            self.embedding,
            self.linear)
        stack_dict[level] = Node_Stack()
        stack_dict[level].push(root_node)
        if not root_node.is_leaf:
            stack_dict[level + 1] = Node_Stack(capacity=root_node.num_child)

        while True:
            char = sentence[idx]
            if char == '(':
                level += 1
                node_info, idx = self.extract_node(sentence, idx)
                node = Node(
                    node_info,
                    self.content_vocab,
                    self.category_vocab,
                    self.embedding,
                    self.linear)
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
                if not root_node.is_leaf:
                    root_node.ready = False
                self.comfirmed_node.append(root_node)
                break
        return Tree(self.comfirmed_node)

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

    # called in self.convert()
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
                    if not child_node.is_leaf:
                        child_node.ready = False
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
                    if not left_child_node.is_leaf:
                        left_child_node.ready = False
                    if not right_child_node.is_leaf:
                        right_child_node.ready = False
                    self.comfirmed_node.append(left_child_node)
                    self.comfirmed_node.append(right_child_node)
                del stack_dict[level]
        return stack_dict, node_id


def easyccg(PATH_TO_DIR, n=10):
    command = ['java', '-jar', PATH_TO_DIR + 'easyccg-0.2/easyccg.jar']
    option = [
        '-m',
        PATH_TO_DIR + 'easyccg-0.2/model',
        '-f',
        PATH_TO_DIR + 'CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw',
        '-n',
        str(n)]
    print('easyccg processing...')
    proc = subprocess.run(command + option, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    result = proc.stdout.decode("utf8").splitlines()
    return result


@torch.no_grad()
def cal_f1_score(pred_node_list, correct_node_list):
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


def convert_for_eval(sentence, category_vocab):
    if sentence == '':
        return "(2 fail)"
    else:
        partial = []
        converted = []
        pos = 0
        while True:
            char = sentence[pos]
            if char == '<':
                start_pos = pos
                while True:
                    pos += 1
                    char = sentence[pos]
                    if char == '>':
                        end_pos = pos
                        break
                partial = sentence[start_pos:end_pos + 1].split()
                if partial[0] == '<L':
                    converted.append(str(category_vocab[partial[1]]))
                    converted.append(' ')
                    converted.append(partial[4])
                else:
                    converted.append(str(category_vocab[partial[1]]))
                pos += 1
            else:
                converted.append(char)
                pos += 1
                if pos == len(sentence):
                    break
        converted = ''.join(converted).replace(' )', ')')
        return converted
