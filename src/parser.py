from utils import load, Condition_Setter
import time
import torch
from utils import single_circular_correlation
import time


class Category:
    def __init__(
            self,
            cell_id,
            cat,
            cat_id,
            vector,
            total_score,
            label_score,
            span_score=None,
            num_child=None,
            left_child=None,
            right_child=None):
        self.cell_id = cell_id
        self.cat = cat
        self.cat_id = cat_id
        self.vector = vector
        self.total_score = total_score
        self.label_score = label_score
        self.span_score = span_score
        self.num_child = num_child
        self.left_child = left_child
        self.right_child = right_child


class Cell:
    def __init__(self, content):
        self.content = content
        self.category_list = []
        self.best_category_id = {}

    def add_category(self, category):
        # when category already exist in the cell
        if category.cat_id in self.best_category_id:
            best_category = self.category_list[self.best_category_id[category.cat_id]]
            # only when the new category has higher score than existing one, replace it
            if category.total_score > best_category.total_score:
                self.best_category_id[category.cat_id] = len(self.category_list)
                self.category_list.append(category)
        else:
            self.best_category_id[category.cat_id] = len(self.category_list)
            self.category_list.append(category)
            return self.best_category_id[category.cat_id]


class Parser:
    def __init__(
            self,
            tree_net,
            binary_rule,
            unary_rule,
            category_vocab,
            word_to_whole,
            whole_to_phrase,
            stag_threshold,
            label_threshold,
            span_threshold):
        self.tokenizer = tree_net.tokenizer
        self.encoder = tree_net.model
        self.word_ff = tree_net.word_ff.to('cpu')
        self.phrase_ff = tree_net.phrase_ff.to('cpu')
        self.span_ff = tree_net.span_ff.to('cpu')
        self.binary_rule = binary_rule
        self.unary_rule = unary_rule
        self.category_vocab = category_vocab
        self.word_to_whole = word_to_whole
        self.whole_to_phrase = whole_to_phrase
        self.stag_threshold = stag_threshold
        self.label_threshold = label_threshold
        self.span_threshold = span_threshold

    def initialize_chart(self, sentence):
        start = time.time()
        sentence = sentence.split()
        converted_sentence = []
        converted_sentence_ = []
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
            converted_sentence_.append(content)
            if r"\/" in content:
                content = content.replace(r"\/", "/")
            converted_sentence.append(content)
        tokens = self.tokenizer.tokenize(" ".join(converted_sentence))
        tokenized_pos = 0
        word_split = []
        for original_pos in range(len(converted_sentence)):
            word = converted_sentence[original_pos]
            length = 1
            while True:
                temp = self.tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length]).replace(" ", "")
                if word == temp or word.lower() == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        print('set word split:{}'.format(time.time() - start))
        start = time.time()
        input = self.tokenizer(
            " ".join(converted_sentence),
            return_tensors='pt').to(self.encoder.device)
        output = self.encoder(**input).last_hidden_state[0, 1:-1].to('cpu')
        temp = []
        for start_idx, end_idx in word_split:
            temp.append(torch.mean(output[start_idx:end_idx], dim=0))
        word_vectors = torch.stack(temp)
        print('encoding:{}'.format(time.time() - start))
        start = time.time()
        word_scores = self.word_ff(word_vectors)
        word_prob = torch.softmax(word_scores, dim=-1)
        word_predict_cats = torch.argsort(word_prob, descending=True)
        word_predict_cats = word_predict_cats[word_predict_cats != 0].view(word_prob.shape[0], -1)

        chart = {}

        for idx in range(len(converted_sentence)):
            word = converted_sentence_[idx]
            vector = word_vectors[idx]
            score = word_scores[idx]
            prob = word_prob[idx]
            top_cat_id = word_predict_cats[idx, 0]
            top_category = Category(
                (idx, idx + 1),
                self.category_vocab.itos[self.word_to_whole[top_cat_id]],
                self.word_to_whole[top_cat_id],
                vector,
                total_score=score[top_cat_id],
                label_score=score[top_cat_id])
            chart[(idx, idx + 1)] = Cell(word)
            chart[(idx, idx + 1)].add_category(top_category)

            for cat_id in word_predict_cats[idx, 1:]:
                if prob[cat_id] > self.stag_threshold:
                    category = Category((idx,
                                         idx + 1),
                                        self.category_vocab.itos[self.word_to_whole[cat_id]],
                                        self.word_to_whole[cat_id],
                                        vector,
                                        score[cat_id],
                                        score[cat_id])
                    chart[(idx, idx + 1)].add_category(category)
                else:
                    break

            waiting_cat_id = list(chart[(idx, idx + 1)].best_category_id.values())
            while True:
                if waiting_cat_id == []:
                    break
                else:
                    child_cat_id = waiting_cat_id.pop(0)
                    child_cat = chart[(idx, idx + 1)].category_list[child_cat_id]
                    possible_cat_id = self.unary_rule.get(child_cat.cat_id)
                    if possible_cat_id is None:
                        continue
                    else:
                        span_score = self.span_ff(child_cat.vector)
                        span_prob = torch.sigmoid(span_score)
                        if span_prob > self.span_threshold:
                            phrase_scores = self.phrase_ff(child_cat.vector)
                            phrase_probs = torch.softmax(phrase_scores, dim=-1)
                            for parent_cat_id in possible_cat_id:
                                cat = self.category_vocab.itos[parent_cat_id]
                                label_score = phrase_scores[self.whole_to_phrase[parent_cat_id]]
                                label_prob = phrase_probs[self.whole_to_phrase[parent_cat_id]]
                                if label_prob > self.label_threshold:
                                    total_score = label_score + span_score + child_cat.total_score
                                    parent_category = Category(
                                        (idx, idx + 1),
                                        cat,
                                        parent_cat_id,
                                        child_cat.vector,
                                        total_score=total_score,
                                        label_score=label_score,
                                        span_score=span_score,
                                        num_child=1,
                                        left_child=child_cat)
                                    new_cat_id = chart[(idx, idx + 1)].add_category(
                                        parent_category)
                                    if new_cat_id is None:
                                        continue
                                    else:
                                        waiting_cat_id.append(new_cat_id)
        print('initialize chart:{}'.format(time.time() - start))
        return chart

    @torch.no_grad()
    def parse(self, sentence):
        start = time.time()
        chart = self.initialize_chart(sentence)
        print(time.time() - start)
        n = len(chart)
        for length in range(2, n + 1):
            for left in range(n - length + 1):
                right = left + length
                chart[(left, right)] = Cell(' '.join(sentence.split()[left:right]))
                for split in range(left + 1, right):
                    for left_cat_id in chart[(left, split)].best_category_id.values():
                        left_cat = chart[(left, split)].category_list[left_cat_id]
                        for right_cat_id in chart[(split, right)].best_category_id.values():
                            right_cat = chart[(split, right)].category_list[right_cat_id]
                            # list of gramatically possible category
                            possible_cat_id = self.binary_rule.get(
                                (left_cat.cat_id, right_cat.cat_id))
                            if possible_cat_id is None:
                                continue
                            else:
                                composed_vector = single_circular_correlation(
                                    left_cat.vector, right_cat.vector)
                                span_score = self.span_ff(composed_vector)
                                span_prob = torch.sigmoid(span_score)
                                # print(chart[(left, split)].content,
                                #       chart[(split, right)].content, span_score)
                                if span_prob > self.span_threshold:
                                    phrase_scores = self.phrase_ff(composed_vector)
                                    phrase_probs = torch.softmax(phrase_scores, dim=-1)
                                    for parent_cat_id in possible_cat_id:
                                        cat = self.category_vocab.itos[parent_cat_id]
                                        label_score = phrase_scores[self.whole_to_phrase[parent_cat_id]]
                                        label_prob = phrase_probs[self.whole_to_phrase[parent_cat_id]]
                                        if label_prob > self.label_threshold:
                                            total_score = label_score + span_score + left_cat.total_score + right_cat.total_score
                                            parent_category = Category(
                                                (left, right),
                                                cat,
                                                parent_cat_id,
                                                composed_vector,
                                                total_score=total_score,
                                                label_score=label_score,
                                                span_score=span_score,
                                                num_child=2,
                                                left_child=left_cat,
                                                right_child=right_cat)
                                            chart[(left, right)].add_category(parent_category)

                waiting_cat_id = list(chart[(left, right)].best_category_id.values())
                while True:
                    if waiting_cat_id == []:
                        break
                    else:
                        child_cat_id = waiting_cat_id.pop(0)
                        child_cat = chart[(left, right)].category_list[child_cat_id]
                        possible_cat_id = self.unary_rule.get(child_cat.cat_id)
                        if possible_cat_id is None:
                            continue
                        else:
                            span_score = self.span_ff(child_cat.vector)
                            span_prob = torch.sigmoid(span_score)
                            if span_prob > self.span_threshold:
                                phrase_scores = self.phrase_ff(child_cat.vector)
                                phrase_probs = torch.softmax(phrase_scores, dim=-1)
                                for parent_cat_id in possible_cat_id:
                                    cat = self.category_vocab.itos[parent_cat_id]
                                    label_score = phrase_scores[self.whole_to_phrase[parent_cat_id]]
                                    label_prob = phrase_probs[self.whole_to_phrase[parent_cat_id]]
                                    if label_prob > self.label_threshold:
                                        total_score = label_score + span_score + child_cat.total_score
                                        parent_category = Category(
                                            (left, right),
                                            cat,
                                            parent_cat_id,
                                            child_cat.vector,
                                            total_score=total_score,
                                            label_score=label_score,
                                            span_score=span_score,
                                            num_child=1,
                                            left_child=child_cat)
                                        new_cat_id = chart[(left, right)].add_category(
                                            parent_category)
                                        if new_cat_id is None:
                                            continue
                                        else:
                                            waiting_cat_id.append(new_cat_id)
        print(time.time() - start)
        total_cat = 0
        for cell in chart.values():
            total_cat += len(cell.category_list)
        print(total_cat)
        return chart

    # remove the candidate of low probability for beam search
    @ torch.no_grad()
    def cut_off(self, category_table, prob, i, j, width=5):
        if len(category_table[i][j]) > width:
            top_5_cat = torch.topk(prob[i][j], k=width)[1]
            category_table[i][j] = list(top_5_cat)
        return category_table

    @ torch.no_grad()
    def reconstruct_tree(self, category_table, backpointer, n):
        waiting_node_list = []
        node_list = []
        # when parsing was completed
        if category_table[0][n] != []:
            top_cat = category_table[0][n][0].item()
            if torch.any(backpointer[(0, n, top_cat)]):
                waiting_node_list.append((0, n, top_cat))
                while waiting_node_list != []:
                    node_info = waiting_node_list.pop()
                    node_list.append(node_info)
                    start_idx = node_info[0]
                    end_idx = node_info[1]
                    cat = node_info[2]
                    child_cat_info = backpointer[(start_idx, end_idx, cat)]
                    divide_idx = child_cat_info[0].item()
                    left_child_cat = child_cat_info[1].item()
                    right_child_cat = child_cat_info[2].item()
                    # when one child
                    if divide_idx == 0:
                        child_info = (start_idx, end_idx, left_child_cat)
                        # when the node is not leaf
                        if torch.any(backpointer[child_info]):
                            waiting_node_list.append(child_info)
                    # when two children
                    else:
                        left_child_info = (start_idx, divide_idx, left_child_cat)
                        right_child_info = (divide_idx, end_idx, right_child_cat)
                        # when the node is not leaf
                        if torch.any(backpointer[left_child_info]):
                            waiting_node_list.append(left_child_info)
                        # when the node is not leaf
                        if torch.any(backpointer[right_child_info]):
                            waiting_node_list.append(right_child_info)
            else:
                node_list.append((0, n, top_cat))
        return node_list


def extract_rule(path_to_grammar, category_vocab):
    binary_rule = {}
    unary_rule = {}

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    for rule in data:
        tokens = rule.split()
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]]
            left_cat = category_vocab[tokens[4]]
            right_cat = category_vocab[tokens[5]]
            if (left_cat, right_cat) in binary_rule:
                binary_rule[(left_cat, right_cat)].append(parent_cat)
            else:
                binary_rule[(left_cat, right_cat)] = [parent_cat]
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]]
            child_cat = category_vocab[tokens[4]]
            if child_cat in unary_rule:
                unary_rule[child_cat].append(parent_cat)
            else:
                unary_rule[child_cat] = [parent_cat]
    return binary_rule, unary_rule


def main():
    condition = Condition_Setter(set_embedding_type=False)

    device = torch.device('cuda')

    model = "roberta-large_phrase(a).pth"

    tree_net = torch.load(condition.path_to_model + model,
                          map_location=device)
    tree_net.device = device
    tree_net.eval()

    category_vocab = load(condition.path_to_whole_category_vocab)
    word_to_whole = load(condition.path_to_word_to_whole)
    whole_to_phrase = load(condition.path_to_whole_to_phrase)

    binary_rule, unary_rule = extract_rule(condition.path_to_grammar, category_vocab)
    parser = Parser(
        tree_net,
        binary_rule,
        unary_rule,
        category_vocab,
        word_to_whole,
        whole_to_phrase,
        stag_threshold=0.075,
        label_threshold=0.001,
        span_threshold=0.1)
    sentence0 = "My sister loves to eat ."
    sentence1 = "Pierre Vinken , 61 years old , will join the board as a nonexecutive director Nov. 29 ."
    chart = parser.parse(sentence1)
    a = 0


if __name__ == "__main__":
    main()
