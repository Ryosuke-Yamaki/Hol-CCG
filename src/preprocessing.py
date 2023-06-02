import os
from typing import Tuple
from tree import TreeList
from utils import dump


class Converter:
    def __init__(self):
        """the class for converting constituency tree of CCG to node information
        """

    def load_sentence(self, path_to_autos: str) -> None:
        """the method for loading constituency tree of CCG

        Parameters
        ----------
        path_to_data : str
            the path to constituency tree of CCG
        """
        self.auto_list = []
        f = open(path_to_autos, 'r')
        data = f.readlines()
        f.close()
        for i in range(len(data)):
            if i % 2 == 1:
                self.auto_list.append(data[i])

    def convert_and_save(self, path_to_autos: str, path_to_save: str) -> None:
        """the method for converting constituency tree of CCG to node information

        Parameters
        ----------
        path_to_autos : str
            the path to constituency trees of CCG
        path_to_save : str
            the path to save node information
        """
        self.load_sentence(path_to_autos)
        for line in self.auto_list:
            self.convert(line)
            self.save_node_info(path_to_save)

    def convert(self, line: str) -> None:
        """convert constituency tree of CCG to node information

        Parameters
        ----------
        line : str
            the line of constituency tree of CCG
        """
        self.comfirmed_node = []
        stack_dict = {}
        idx = 0
        level = 0
        node_id = 0
        node_info, idx = self.extract_node(line, idx)
        root_node = Node(node_info)
        stack_dict[level] = Node_Stack()
        stack_dict[level].push(root_node)
        if not root_node.is_leaf:
            stack_dict[level + 1] = Node_Stack(capacity=root_node.num_child)

        while True:
            char = line[idx]
            if char == '(':
                level += 1
                node_info, idx = self.extract_node(line, idx)
                node = Node(node_info)
                stack_dict[level].push(node)
                if not node.is_leaf:
                    stack_dict[level + 1] = Node_Stack(capacity=node.num_child)
            elif char == ')':
                level -= 1
            idx += 1
            stack_dict, node_id = self.search_parent_child_relation(stack_dict, node_id)
            if idx == len(line):
                root_node = stack_dict[0].node_stack[-1]
                root_node.self_id = node_id
                self.comfirmed_node.append(root_node)
                break

    def extract_node(self, line: str, idx: int) -> Tuple[list, int]:
        """extract node information from line

        Parameters
        ----------
        line : str
            the line of constituency tree of CCG
        idx : int
            the index of node

        Returns
        -------
        node_info : list
            the list of node information
        idx : int
            the index of node
        """
        start_idx_of_node = idx + 2
        for idx in range(start_idx_of_node, len(line)):
            char = line[idx]
            if char == '>':
                end_idx_of_node = idx
                break
        node_info = line[start_idx_of_node:end_idx_of_node]
        return node_info.split(), idx

    def search_parent_child_relation(self, stack_dict: dict, node_id: int) -> Tuple[dict, int]:
        """search parent-child relation between nodes

        Parameters
        ----------
        stack_dict : dict
            the dictionary of node stack
        node_id : int
            the id of node

        Returns
        -------
        stack_dict : dict
            the dictionary of node stack
        node_id : int
            the id of node
        """
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

    def save_node_info(self, path_to_save: str) -> None:
        """save converted node information to file

        Parameters
        ----------
        path_to_save : str
            the path to save node information
        """
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
    def __init__(self, node_info: list):
        """the class for node in constituency tree of CCG

        Parameters
        ----------
        node_info : list
            the list of node information
        """
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

    def set_self_id(self, id: int) -> None:
        """set self id to node

        Parameters
        ----------
        id : int
            the id of node
        """
        self.self_id = id

    def set_child_id(self, id: int) -> None:
        """set child id to node

        Parameters
        ----------
        id : int
            the id of child node
        """
        self.child_id = id


class Node_Stack:
    def __init__(self, capacity: int = None):
        """the class for node stack to search parent-child relation
        """
        self.node_stack = []
        self.capacity = capacity
        self.ready = False

    def push(self, node: Node) -> None:
        """push node to node_stack

        Parameters
        ----------
        node : Node
            the node to push"""
        self.node_stack.append(node)

    def pop(self) -> Node:
        """pop node from node_stack

        Returns
        -------
        node : Node
            the node to pop
        """
        return self.node_stack.pop(-1)

    def update_status(self) -> None:
        """update status of node_stack
        """
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


def main():
    path_to_dataset = '../dataset/'
    path_to_ccgbank = os.path.join(path_to_dataset, 'ccgbank_1_1/')
    path_to_file_names = os.path.join(path_to_ccgbank, 'doc/file.tbl')
    path_to_converted = os.path.join(path_to_dataset, 'converted/')
    path_to_tree_list = os.path.join(path_to_dataset, 'tree_list/')
    path_to_grammar = os.path.join(path_to_dataset, 'grammar/')

    os.makedirs(path_to_converted, exist_ok=True)
    # make the file to save converted tree
    for split in ['train', 'dev', 'test']:
        path_to_save = os.path.join(path_to_converted, split + '.txt')
        open(path_to_save, 'w')
    os.makedirs(path_to_tree_list, exist_ok=True)
    os.makedirs(path_to_grammar, exist_ok=True)

    converter = Converter()

    f = open(path_to_file_names, 'r')
    file_paths = f.readlines()

    print('Converting ccgbank format...')
    for path in file_paths:
        if '.auto' in path:
            idx = int(path[-10:-6])
            if idx < 100:
                path_to_save = os.path.join(path_to_converted, 'dev.txt')
            elif 200 <= idx and idx < 2200:
                path_to_save = os.path.join(path_to_converted, 'train.txt')
            elif 2300 <= idx and idx < 2400:
                path_to_save = os.path.join(path_to_converted, 'test.txt')
            else:
                continue

            converter.convert_and_save(os.path.join(path_to_ccgbank, path.replace('\n', '')), path_to_save)
        else:
            break

    print('Initializing tree...')
    train_tree_list = TreeList(
        os.path.join(path_to_converted, 'train.txt'), type='train')

    word_category_vocab = train_tree_list.word_category_vocab
    phrase_category_vocab = train_tree_list.phrase_category_vocab
    head_info = train_tree_list.head_info

    dev_tree_list = TreeList(
        os.path.join(path_to_converted, 'dev.txt'),
        type='dev',
        word_category_vocab=word_category_vocab,
        phrase_category_vocab=phrase_category_vocab,
        head_info=head_info)

    test_tree_list = TreeList(
        os.path.join(path_to_converted, 'test.txt'),
        type='test',
        word_category_vocab=word_category_vocab,
        phrase_category_vocab=phrase_category_vocab,
        head_info=head_info)

    print('Binarizing tree...')
    train_tree_list.convert_to_binary(type='train')
    word_category_vocab = train_tree_list.word_category_vocab
    phrase_category_vocab = train_tree_list.phrase_category_vocab
    head_info = train_tree_list.head_info
    rule_counter = train_tree_list.count_rule()
    dev_tree_list.word_category_vocab = word_category_vocab
    dev_tree_list.phrase_category_vocab = phrase_category_vocab
    test_tree_list.word_category_vocab = word_category_vocab
    test_tree_list.phrase_category_vocab = phrase_category_vocab
    dev_tree_list.convert_to_binary(type='dev')
    test_tree_list.convert_to_binary(type='test')

    dump(train_tree_list, os.path.join(path_to_tree_list, 'train_tree_list.pickle'))
    dump(dev_tree_list, os.path.join(path_to_tree_list, 'dev_tree_list.pickle'))
    dump(test_tree_list, os.path.join(path_to_tree_list, 'test_tree_list.pickle'))
    dump(word_category_vocab, os.path.join(path_to_grammar, 'word_category_vocab.pickle'))
    dump(phrase_category_vocab, os.path.join(path_to_grammar, 'phrase_category_vocab.pickle'))
    dump(head_info, os.path.join(path_to_grammar, 'head_info.pickle'))
    dump(rule_counter, os.path.join(path_to_grammar, 'rule_counter.pickle'))


if __name__ == '__main__':
    main()
