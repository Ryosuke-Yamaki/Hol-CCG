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
    def __init__(self, length_of_sentence, chart, sentence, id_to_category):
        self.length_of_sentence = length_of_sentence
        self.chart = chart
        self.sentence = sentence.split()
        self.id_to_category = id_to_category
        self.node_list = []
        self.pointers_before_define = []
        self.define_top_node()
        self.define_other_nodes()
        self.convert_node_list_for_eval()

    def define_top_node(self):
        top_node = Node(0, (0, self.length_of_sentence), (' ').join(self.sentence),
                        self.chart[(0, self.length_of_sentence)], top=True)
        left_pointer, right_pointer = top_node.extract_back_pointer()
        self.node_list.append(top_node)
        self.pointers_before_define.append(left_pointer)
        self.pointers_before_define.append(right_pointer)

    def define_other_nodes(self):
        node_id = 1
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
        for node in self.node_list:
            scope = node.scope
            category_id = node.category_id
            converted_node_list.append((scope[0], scope[1], category_id))
        self.converted_node_list = converted_node_list

    def cal_f1_score(self, correct_node_list):
        pred_node_list = self.converted_node_list
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
        f1 = (2 * precision * recall) / (precision + recall)

        return f1, precision, recall

    def visualize_parsing_result(self):
        for node in self.node_list:
            print('content: {}'.format(node.content))
            print('category :{}'.format(self.id_to_category[node.category_id]))
            print()
