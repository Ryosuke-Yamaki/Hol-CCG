import os
import argparse
from collections import Counter
import sys
from utils import load
import torch
from utils import circular_correlation, circular_convolution


class Category:
    def __init__(
            self,
            cell_id,
            cat,
            type,
            is_leaf=False,
            vector=None,
            total_ll=None,
            cat_ll=None,
            span_ll=None,
            num_child=None,
            left_child=None,
            right_child=None,
            head=None,
            word=None):
        self.cell_id = cell_id
        self.original_cat = cat
        cat_list = cat.split('-->')
        # set final category as cat
        self.cat = cat_list[-1]
        self.unary_chain = cat_list[:-1]
        self.type = type
        self.vector = vector
        self.total_ll = total_ll
        self.cat_ll = cat_ll
        self.span_ll = span_ll
        self.num_child = num_child
        self.left_child = left_child
        self.right_child = right_child
        self.head = head
        self.is_leaf = is_leaf
        self.word = word


class Cell:
    def __init__(self, content):
        self.content = content
        self.best_category = {}

    def add_category(self, category):
        # when category already exist in the cell
        if category.cat in self.best_category:
            best_category = self.best_category[category.cat]
            # only when the new category has higher probability than existing one, replace it
            if category.total_ll > best_category.total_ll:
                self.best_category[category.cat] = category
        # when firstly add category into the cell
        else:
            self.best_category[category.cat] = category
        # when category has unary chain
        if len(category.unary_chain) > 0:
            self.set_unary_chain(category)

    def set_unary_chain(self, category):
        cell_id = category.cell_id
        type = category.type
        is_leaf = category.is_leaf
        num_child = category.num_child
        left_child_cat = category.left_child
        right_child_cat = category.right_child
        head = category.head
        word = category.word
        # the list of unary children
        category.unary_chain_cat_list = []
        for cat in category.unary_chain:
            temp_cat = Category(
                cell_id=cell_id,
                cat=cat,
                type=type,
                is_leaf=is_leaf,
                num_child=num_child,
                left_child=left_child_cat,
                right_child=right_child_cat,
                head=head,
                word=word)
            category.unary_chain_cat_list.append(temp_cat)
            left_child_cat = temp_cat
            right_child_cat = None
            type = 'unary'
            is_leaf = False
            num_child = 1
            head = None
            word = None
        category.left_child = left_child_cat
        category.right_child = right_child_cat
        category.type = type
        category.is_leaf = is_leaf
        category.num_child = num_child
        category.head = head
        category.word = word


class Parser:
    def __init__(
            self,
            word_category_vocab,
            phrase_category_vocab,
            head_info,
            rule_counter,
            tree_net,
            stag_threshold,
            phrase_threshold,
            span_threshold,
            min_freq):
        self.word_category_vocab = word_category_vocab
        self.phrase_category_vocab = phrase_category_vocab
        self.head_info = head_info
        self.binary_rule = {}
        for key, freq in rule_counter.items():
            if freq >= min_freq:
                if (key[0], key[1]) in self.binary_rule:
                    self.binary_rule[(key[0], key[1])].append(key[2])
                else:
                    self.binary_rule[(key[0], key[1])] = [key[2]]
        self.tree_net = tree_net
        self.composition = tree_net.composition
        self.word_ff = tree_net.word_ff
        self.phrase_ff = tree_net.phrase_ff
        self.span_ff = tree_net.span_ff
        self.stag_threshold = stag_threshold
        self.phrase_threshold = phrase_threshold
        self.span_threshold = span_threshold

    @torch.no_grad()
    def initialize_chart(self, sentence):
        sentence = sentence.split()
        converted_sentence = []
        for i in range(len(sentence)):
            content = sentence[i]
            if content == "-LRB-":
                content = "("
            elif content == "-LCB-":
                content = "{"
            elif content == "-RRB-":
                content = ")"
            elif content == "-RCB-":
                content = "}"
            if r"\/" in content:
                content = content.replace(r"\/", "/")
            converted_sentence.append(content)
        word_split = self.tree_net.set_word_split(converted_sentence)
        word_vectors, _ = self.tree_net.encode([" ".join(converted_sentence)], [word_split])
        word_vectors = word_vectors[0]
        word_probs_list = torch.softmax(self.word_ff(word_vectors), dim=-1)
        word_predict_cats = torch.argsort(word_probs_list, descending=True)
        word_predict_cats = word_predict_cats[word_predict_cats != 0].view(
            word_probs_list.shape[0], -1)

        chart = {}

        for idx in range(len(converted_sentence)):
            word = sentence[idx]
            vector = word_vectors[idx]
            word_probs = word_probs_list[idx]
            top_cat_id = word_predict_cats[idx, 0]
            top_cat = self.word_category_vocab.itos[top_cat_id]
            top_category = Category(
                cell_id=(idx, idx + 1),
                cat=top_cat,
                type='stag',
                vector=vector,
                total_ll=torch.log(word_probs[top_cat_id]),
                cat_ll=torch.log(word_probs[top_cat_id]),
                is_leaf=True,
                word=word)
            chart[(idx, idx + 1)] = Cell(word)
            chart[(idx, idx + 1)].add_category(top_category)

            for cat_id in word_predict_cats[idx, 1:]:
                if word_probs[cat_id] > self.stag_threshold:
                    category = Category(cell_id=(idx,
                                                 idx + 1),
                                        cat=self.word_category_vocab.itos[cat_id],
                                        type='stag',
                                        vector=vector,
                                        total_ll=torch.log(word_probs[cat_id]),
                                        cat_ll=torch.log(word_probs[cat_id]),
                                        is_leaf=True,
                                        word=word)
                    chart[(idx, idx + 1)].add_category(category)
                else:
                    break
        return chart

    @torch.no_grad()
    def parse(self, sentence):
        chart = self.initialize_chart(sentence)
        n = len(chart)
        for length in range(2, n + 1):
            for left in range(n - length + 1):
                right = left + length
                chart[(left, right)] = Cell(' '.join(sentence.split()[left:right]))
                for split in range(left + 1, right):
                    for left_cat in chart[(left, split)].best_category.values():
                        for right_cat in chart[(split, right)].best_category.values():
                            # list of gramatically possible category
                            possible_cats = self.binary_rule.get(
                                (left_cat.cat, right_cat.cat))
                            # when no binary combination is available
                            if possible_cats is None:
                                continue
                            else:
                                if self.composition == 'corr':
                                    composed_vector = circular_correlation(
                                        left_cat.vector, right_cat.vector, self.tree_net.vector_norm)
                                elif self.composition == 'conv':
                                    composed_vector = circular_convolution(
                                        left_cat.vector, right_cat.vector, self.tree_net.vector_norm)
                                span_prob = torch.softmax(
                                    self.span_ff(composed_vector), dim=-1)[1]
                                if span_prob > self.span_threshold:
                                    span_ll = torch.log(span_prob)
                                    phrase_probs = torch.softmax(
                                        self.phrase_ff(composed_vector), dim=-1)
                                    for parent_cat in possible_cats:
                                        parent_cat_id = self.phrase_category_vocab[parent_cat]
                                        # when category is not <unk>
                                        if parent_cat_id != 0:
                                            cat_prob = phrase_probs[parent_cat_id]
                                            if cat_prob > self.phrase_threshold:
                                                cat_ll = torch.log(cat_prob)
                                                total_ll = cat_ll + span_ll + left_cat.total_ll + right_cat.total_ll
                                                head = self.head_info[(
                                                    left_cat.cat, right_cat.cat, parent_cat.split('-->')[0])]
                                                parent_category = Category(
                                                    cell_id=(left, right),
                                                    cat=parent_cat,
                                                    type='bin',
                                                    vector=composed_vector,
                                                    total_ll=total_ll,
                                                    cat_ll=cat_ll,
                                                    span_ll=span_ll,
                                                    num_child=2,
                                                    left_child=left_cat,
                                                    right_child=right_cat,
                                                    head=head)
                                                chart[(left, right)].add_category(
                                                    parent_category)
        return chart

    def skimmer(self, chart):
        len_sentence = list(chart.keys())[-1][1]
        found_span_id = []
        if len_sentence == 1:
            found_span_id.append((0, 1))
        else:
            cell_id_list = list(chart.keys())
            scope_list = [(0, len_sentence)]
            while scope_list != []:
                scope = scope_list.pop(0)
                for cell_id in cell_id_list:
                    if scope[0] <= cell_id[0] and cell_id[1] <= scope[1]:
                        cell = chart[cell_id]
                        max_span_length = 0
                        if len(cell.best_category) != 0:
                            span_length = cell_id[1] - cell_id[0]
                            if span_length > max_span_length:
                                max_span_length = span_length
                                max_span_id = cell_id
                found_span_id.append(max_span_id)
                cell_id_list.remove(max_span_id)
                left_scope = (scope[0], max_span_id[0])
                right_scope = (max_span_id[1], scope[1])
                if left_scope[1] - left_scope[0] > 1:
                    scope_list.append(left_scope)
                elif left_scope[1] - left_scope[0] == 1:
                    found_span_id.append(left_scope)
                if right_scope[1] - right_scope[0] > 1:
                    scope_list.append(right_scope)
                elif right_scope[1] - right_scope[0] == 1:
                    found_span_id.append(right_scope)
        autos = []
        auto_scopes = []
        for span_id in found_span_id:
            auto = self.decode(chart[span_id])
            autos.append(auto)
            auto_scopes.append(span_id)

        sorted_autos = []
        sorted_auto_scopes = []
        target = 0
        while True:
            for i in range(len(auto_scopes)):
                scope = auto_scopes[i]
                start = scope[0]
                end = scope[1]
                if start == target:
                    sorted_autos.append(autos[i])
                    sorted_auto_scopes.append(auto_scopes[i])
                    target = end
                    break
            if target == len_sentence:
                break
        return sorted_autos, sorted_auto_scopes

    def decode(self, root_cell):
        def next(waiting_cats):
            cat = waiting_cats.pop(0)
            if cat.is_leaf:
                cat.auto.append('(<L')
                cat.auto.append(cat.cat)
                cat.auto.append('POS')
                cat.auto.append('POS')
                cat.auto.append(cat.word)
                cat.auto.append(cat.cat + '>)')
            elif cat.num_child == 1:
                child_cat = cat.left_child
                child_cat.auto = []
                cat.auto.append('(<T')
                cat.auto.append(cat.cat)
                cat.auto.append('0')
                cat.auto.append('1>')
                cat.auto.append(child_cat.auto)
                cat.auto.append(')')
                waiting_cats.append(child_cat)
            elif cat.num_child == 2:
                left_child_cat = cat.left_child
                right_child_cat = cat.right_child
                left_child_cat.auto = []
                right_child_cat.auto = []
                cat.auto.append('(<T')
                cat.auto.append(cat.cat)
                cat.auto.append(str(cat.head))
                cat.auto.append('2>')
                cat.auto.append(left_child_cat.auto)
                cat.auto.append(right_child_cat.auto)
                cat.auto.append(')')
                waiting_cats.append(left_child_cat)
                waiting_cats.append(right_child_cat)
            return waiting_cats

        def flatten(auto):
            for i in auto:
                if isinstance(i, list):
                    yield from flatten(i)
                else:
                    yield i

        root_cat = list(root_cell.best_category.values())[0]
        for cat in list(root_cell.best_category.values())[1:]:
            if cat.total_ll > root_cat.total_ll:
                root_cat = cat
        root_cat.auto = []
        waiting_cats = [root_cat]
        while True:
            if len(waiting_cats) == 0:
                break
            else:
                waiting_cats = next(waiting_cats)

        auto = ' '.join(list(flatten(root_cat.auto)))
        return auto


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', type=str, required=True)
    parser.add_argument('-t', '--target', type=str, required=True)
    parser.add_argument('--stag_threshold', type=float, default=0.1)
    parser.add_argument('--phrase_threshold', type=float, default=0.01)
    parser.add_argument('--span_threshold', type=float, default=0.01)
    parser.add_argument('--min_freq', type=int, default=1)
    parser.add_argument('--skimmer_off', action='store_true')
    parser.add_argument('-d', '--device', type=torch.device, default=torch.device('cuda:0'))
    parser.add_argument('--max_num_sentence', type=int, default=None)

    args = parser.parse_args()

    model = args.model
    target = args.target
    stag_threhold = args.stag_threshold
    phrase_threshold = args.phrase_threshold
    span_threshold = args.span_threshold
    min_freq = args.min_freq
    apply_skimmer = not args.skimmer_off
    device = args.device
    max_num_sentence = args.max_num_sentence

    path_to_model = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                              '../data/model/'), model)
    path_to_word_category_vocab = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../data/grammar/word_category_vocab.pickle')
    path_to_phrase_category_vocab = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../data/grammar/phrase_category_vocab.pickle')
    path_to_head_info = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../data/grammar/head_info.pickle')
    path_to_rule_counter = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        '../data/grammar/rule_counter.pickle')
    if target == 'dev':
        path_to_sentence_list = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw')
    elif target == 'test':
        path_to_sentence_list = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            '../../CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw')
    else:
        path_to_sentence_list = target
    word_category_vocab = load(path_to_word_category_vocab)
    phrase_category_vocab = load(path_to_phrase_category_vocab)
    head_info = load(path_to_head_info)
    rule_counter = load(path_to_rule_counter)
    tree_net = torch.load(path_to_model, map_location=device)
    tree_net.device = device
    tree_net.eval()

    parser = Parser(
        word_category_vocab=word_category_vocab,
        phrase_category_vocab=phrase_category_vocab,
        head_info=head_info,
        rule_counter=rule_counter,
        tree_net=tree_net,
        stag_threshold=stag_threhold,
        phrase_threshold=phrase_threshold,
        span_threshold=span_threshold,
        min_freq=min_freq)

    with open(path_to_sentence_list, 'r') as f:
        sentence_list = f.readlines()

    num_success = 0
    num_fail = 0
    sentence_id = 0
    for sentence in sentence_list[:max_num_sentence]:
        sentence_id += 1
        sentence = sentence.rstrip()
        chart = parser.parse(sentence)
        root_cell = list(chart.values())[-1]
        # when parsing is failed
        if len(root_cell.best_category) == 0:
            num_fail += 1
            if apply_skimmer:
                autos, scope_list = parser.skimmer(chart)
                n = 0
                for auto, scope in zip(autos, scope_list):
                    print(
                        'ID={}.{} PARSER=TEST APPLY_SKIMMER=True SCOPE=({},{})'.format(
                            sentence_id, n, scope[0], scope[1]))
                    print(auto)
                    n += 1
            else:
                print('ID={} PARSER=TEST APPLY_SKIMMER=False'.format(sentence_id))
                print('(<L fail POS POS {} fail>)'.format('_'.join(sentence.split())))
        # when parsing is succesful
        else:
            num_success += 1
            auto = parser.decode(root_cell)
            print('ID={} PARSER=TEST APPLY_SKIMMER=FALSE'.format(sentence_id))
            print(auto)
    print(
        f'success ratio: {num_success / (num_success + num_fail)}({num_success}/{(num_success + num_fail)})')


if __name__ == "__main__":
    main()
