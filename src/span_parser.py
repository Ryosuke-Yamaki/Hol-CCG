import os
import argparse
from utils import load, convert_content
import torch
from utils import circular_correlation, circular_convolution, shuffled_circular_convolution
from torchtext.vocab import Vocab
from holccg import HolCCG
from typing import List, Dict, Tuple


class Category:
    def __init__(
            self,
            cell_id: Tuple[int, int],
            cat: str,
            type: str,
            is_leaf: bool = False,
            vector: torch.Tensor = None,
            total_ll: float = None,
            cat_ll: float = None,
            span_ll: float = None,
            num_child: int = None,
            left_child: 'Category' = None,
            right_child: 'Category' = None,
            head: int = None,
            word: str = None) -> None:
        """class for each category in the chart

        Parameters
        ----------
        cell_id : Tuple[int, int]
            id of the cell. (start, end)
        cat : str
            category
        type : str
            type of combinatory rule. 'bin' or 'unary'.
        is_leaf : bool, optional
            whether the category is leaf or not, by default False
        vector : torch.Tensor, optional
            vector representation of the category, by default None
        total_ll : float, optional
            total log likelihood of the category, by default None
        cat_ll : float, optional
            log likelihood of the category assignment, by default None
        span_ll : float, optional
            log likelihood of the span existence, by default None
        num_child : int, optional
            number of children, by default None
        left_child : Category, optional
            left child of the category, by default None
        right_child : Category, optional
            right child of the category, by default None
        head : int, optional
            direction of the head, by default None
        word : str, optional
            corresponding word to the category, by default None
        """
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
    def __init__(self, content: str) -> None:
        """class for each cell in the chart

        Parameters
        ----------
        content : str
            word or phrase of the cell
        """
        self.content = content
        self.best_category = {}

    def add_category(self, category: Category) -> None:
        """add category into the cell

        Parameters
        ----------
        category : Category
            category to add
        """
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

    def set_unary_chain(self, category: Category) -> None:
        """set unary chain of the category for merged category

        Parameters
        ----------
        category : Category
            category to set unary chain
        """

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


class SpanParser:
    def __init__(
            self,
            word_category_vocab: Vocab,
            phrase_category_vocab: Vocab,
            head_info: dict,
            rule_counter: dict,
            holccg: HolCCG,
            stag_threshold: float,
            phrase_threshold: float,
            span_threshold: float,
            min_freq: int = 1) -> None:
        """class for span parser using HolCCG

        Parameters
        ----------
        word_category_vocab : Vocab
            word category vocabulary
        phrase_category_vocab : Vocab
            phrase category vocabulary
        head_info : dict
            head information depends on each combinatory rule
        rule_counter : dict
            dictionary of combinatory rules and its frequency
        holccg : HolCCG
            pre-trained HolCCG model
        stag_threshold : float
            supertagging threshold
        phrase_threshold : float
            phrase category threshold
        span_threshold : float
            span threshold
        min_freq : int, optional
            minimum frequency of combinatory rules, by default 1
        """
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
        self.holccg = holccg
        self.composition = holccg.composition
        if self.composition == 's_conv':
            self.P = holccg.P
        self.word_classifier = holccg.word_classifier
        self.phrase_classifier = holccg.phrase_classifier
        self.span_classifier = holccg.span_classifier
        self.stag_threshold = stag_threshold
        self.phrase_threshold = phrase_threshold
        self.span_threshold = span_threshold

    @torch.no_grad()
    def initialize_chart(self, sentence: str) -> Dict[Tuple[int, int], Cell]:
        """initialize CKY chart

        Parameters
        ----------
        sentence : str
            sentence to be parsed

        Returns
        -------
        Dict[Tuple[int, int], Cell]
            initialized CKY chart
        """

        sentence = sentence.split()
        converted_sentence = []
        for i in range(len(sentence)):
            content = sentence[i]
            converted_sentence.append(convert_content(content))
        word_split = self.holccg.set_word_split(converted_sentence)
        word_vectors, _ = self.holccg.encode([" ".join(converted_sentence)], [word_split])
        word_vectors = word_vectors[0]
        word_probs_list = torch.softmax(self.word_classifier(word_vectors), dim=-1)
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
    def parse(self, sentence: str) -> Dict[Tuple[int, int], Cell]:
        """parse sentence using span-based CKY algorithm

        Parameters
        ----------
        sentence : str
            sentence to be parsed

        Returns
        -------
        Dict[Tuple[int, int], Cell]
            parsed CKY chart
        """

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
                                        left_cat.vector, right_cat.vector, self.holccg.vector_norm)
                                elif self.composition == 'conv':
                                    composed_vector = circular_convolution(
                                        left_cat.vector, right_cat.vector, self.holccg.vector_norm)
                                elif self.composition == 's_conv':
                                    composed_vector = shuffled_circular_convolution(
                                        left_cat.vector, right_cat.vector, self.P, self.holccg.vector_norm)
                                span_prob = torch.softmax(
                                    self.span_classifier(composed_vector), dim=-1)[1]
                                if span_prob > self.span_threshold:
                                    span_ll = torch.log(span_prob)
                                    phrase_probs = torch.softmax(
                                        self.phrase_classifier(composed_vector), dim=-1)
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

    def skimmer(self, chart: Dict[Tuple[int, int], Cell]) -> Tuple[str, Tuple[int, int]]:
        """apply skimmer mode to chart. find successfully parsed subspans.

        Parameters
        ----------
        chart : Dict[Tuple[int, int], Cell]
            parsed CKY chart

        Returns
        -------
        Tuple[str, Tuple[int, int]]
            parsed sentence in auto format and its span id
        """

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

    def decode(self, root_cell: Cell) -> str:
        """decode parsed CKY chart to auto format

        Parameters
        ----------
        root_cell : Cell
            root cell of parserd CKY chart

        Returns
        -------
        str
            auto format of parsed sentence
        """

        def next(waiting_cats: List[Category]) -> List[Category]:
            """find next category to be processed

            Parameters
            ----------
            waiting_cats : List[Category]
                list of categories waiting to be processed

            Returns
            -------
            List[Category]
                list of categories waiting to be processed
            """

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

        def flatten(auto: List) -> List:
            """flatten nested list

            Parameters
            ----------
            auto : List
                nested list

            Returns
            -------
            List
                flattened list
            """

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


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_sentence', type=str, help='path to sentence to be supertagged')
    parser.add_argument('--path_to_model', type=str, help='path to model used for supertagging')
    parser.add_argument('--path_to_dataset', type=str, default='../dataset/', help='path to dataset')
    parser.add_argument('--stag_threshold', type=float, default=0.1, help='threshold for supertagging')
    parser.add_argument('--phrase_threshold', type=float, default=0.01, help='threshold for phrase')
    parser.add_argument('--span_threshold', type=float, default=0.01, help='threshold for span')
    parser.add_argument('--min_freq', type=int, default=1, help='minimum frequency of combinatory rule to be used')
    parser.add_argument('--skimmer', action='store_true', help='use skimmer')
    parser.add_argument(
        '--device',
        type=torch.device,
        default=torch.device('cuda:0'),
        help='device to use for supertagging')
    args = parser.parse_args()
    return args


def main():
    args = arg_parse()

    word_category_vocab = load(os.path.join(args.path_to_dataset, 'grammar/word_category_vocab.pickle'))
    phrase_category_vocab = load(os.path.join(args.path_to_dataset, 'grammar/phrase_category_vocab.pickle'))
    head_info = load(os.path.join(args.path_to_dataset, 'grammar/head_info.pickle'))
    rule_counter = load(os.path.join(args.path_to_dataset, 'grammar/rule_counter.pickle'))

    with open(args.path_to_sentence, 'r') as f:
        sentence_list = f.readlines()

    holccg = torch.load(args.path_to_model, map_location=args.device)
    holccg.device = args.device
    holccg.eval()

    parser = SpanParser(
        word_category_vocab=word_category_vocab,
        phrase_category_vocab=phrase_category_vocab,
        head_info=head_info,
        rule_counter=rule_counter,
        holccg=holccg,
        stag_threshold=args.stag_threshold,
        phrase_threshold=args.phrase_threshold,
        span_threshold=args.span_threshold,
        min_freq=args.min_freq)

    sentence_id = 0
    for sentence in sentence_list:
        sentence_id += 1
        sentence = sentence.rstrip()
        chart = parser.parse(sentence)
        root_cell = list(chart.values())[-1]
        # when parsing is failed
        if len(root_cell.best_category) == 0:
            if args.skimmer:
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
            auto = parser.decode(root_cell)
            print('ID={} PARSER=TEST APPLY_SKIMMER=FALSE'.format(sentence_id))
            print(auto)


if __name__ == "__main__":
    main()
