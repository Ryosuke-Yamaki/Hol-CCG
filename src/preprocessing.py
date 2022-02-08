from models import Tree_List
from utils import Condition_Setter, dump


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


condition = Condition_Setter(set_embedding_type=False)

path_to_file_table = condition.PATH_TO_DIR + 'CCGbank/ccgbank_1_1/doc/file.tbl'
path_to_train = condition.path_to_train_data
path_to_dev = condition.path_to_dev_data
path_to_test = condition.path_to_test_data
open(path_to_train, 'w')
open(path_to_dev, 'w')
open(path_to_test, 'w')
f = open(path_to_file_table, 'r')
path_to_data = f.readlines()
i = 0
print('converting auto file...')
for path in path_to_data:
    if '.auto' in path:
        idx = int(path[-10:-6])
        if idx < 100:
            path_to_save = path_to_dev
        elif 200 <= idx and idx < 2200:
            path_to_save = path_to_train
        elif 2300 <= idx and idx < 2400:
            path_to_save = path_to_test
        else:
            path_to_save = None
        if path_to_save is not None:
            path = condition.PATH_TO_DIR + 'CCGbank/ccgbank_1_1/' + path.replace('\n', '')
            converter = Converter(path)
            for sentence in converter.sentence_list:
                num_node = sentence.count('<')
                converter.convert(sentence)
                converter.save_node_info(path_to_save)
    else:
        break

print('setting tree list...')
train_tree_list = Tree_List(
    condition.path_to_train_data, type='train')
word_category_vocab = train_tree_list.word_category_vocab
phrase_category_vocab = train_tree_list.phrase_category_vocab
head_info = train_tree_list.head_info
dev_tree_list = Tree_List(
    condition.path_to_dev_data,
    type='dev',
    word_category_vocab=word_category_vocab,
    phrase_category_vocab=phrase_category_vocab,
    head_info=head_info)
test_tree_list = Tree_List(
    condition.path_to_test_data,
    type='test',
    word_category_vocab=word_category_vocab,
    phrase_category_vocab=phrase_category_vocab,
    head_info=head_info)


# dump(train_tree_list, condition.path_to_train_tree_list)
# dump(dev_tree_list, condition.path_to_dev_tree_list)
# dump(test_tree_list, condition.path_to_test_tree_list)
# dump(word_category_vocab, condition.path_to_word_category_vocab)
# dump(phrase_category_vocab, condition.path_to_phrase_category_vocab)
# dump(head_info, condition.path_to_head_info)

print('setting binary tree list...')
train_tree_list.convert_to_binary(type='train')
binary_word_category_vocab = train_tree_list.word_category_vocab
binary_phrase_category_vocab = train_tree_list.phrase_category_vocab
head_info = train_tree_list.head_info
dev_tree_list.word_category_vocab = binary_word_category_vocab
dev_tree_list.phrase_category_vocab = binary_phrase_category_vocab
test_tree_list.word_category_vocab = binary_word_category_vocab
test_tree_list.phrase_category_vocab = binary_phrase_category_vocab
dev_tree_list.convert_to_binary(type='dev')
test_tree_list.convert_to_binary(type='test')

dump(train_tree_list, condition.path_to_binary_train_tree_list)
dump(dev_tree_list, condition.path_to_binary_dev_tree_list)
dump(test_tree_list, condition.path_to_binary_test_tree_list)
dump(binary_word_category_vocab, condition.path_to_binary_word_category_vocab)
dump(binary_phrase_category_vocab, condition.path_to_binary_phrase_category_vocab)
dump(head_info, condition.path_to_head_info)
