from utils import load, Condition_Setter
import time
import torch
from utils import single_circular_correlation
import time


class Cell:
    def __init__(self, content, category):
        self.content = content
        self.category_list = [category]

    def add_category(self, category):
        self.category_list.append(category)


class Category:
    def __init__(
            self,
            cat,
            cat_id,
            vector,
            cat_score,
            span_score=None,
            num_child=None,
            left_child=None,
            right_child=None):
        self.cat = cat
        self.cat_id = cat_id
        self.vector = vector
        self.cat_score = cat_score
        self.span_score = span_score
        self.num_child = num_child
        self.left_child = left_child
        self.right_child = right_child


class Parser:
    @torch.no_grad()
    def __init__(
            self,
            tree_net,
            binary_rule,
            unary_rule,
            category_vocab,
            word_to_whole,
            whole_to_word):
        self.tokenizer = tree_net.tokenizer
        self.encoder = tree_net.model
        self.word_ff = tree_net.word_ff
        self.phrase_ff = tree_net.phrase_ff
        self.span_ff = tree_net.span_ff
        self.binary_rule = binary_rule
        self.unary_rule = unary_rule
        self.category_vocab = category_vocab
        self.word_to_whole = word_to_whole
        self.whole_to_word = whole_to_word

    def initialize_chart(self, sentence, beta=0.075):
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

        input = self.tokenizer(
            " ".join(converted_sentence),
            return_tensors='pt').to(self.encoder.device)
        output = self.encoder(**input).last_hidden_state[0, 1:-1]
        temp = []
        for start_idx, end_idx in word_split:
            temp.append(torch.mean(output[start_idx:end_idx], dim=0))
        word_vectors = torch.stack(temp)
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
                self.category_vocab.itos[self.word_to_whole[top_cat_id]],
                self.word_to_whole[top_cat_id],
                vector,
                score[top_cat_id])
            chart[(idx, idx + 1)] = Cell(word, top_category)

            for cat_id in word_predict_cats[idx, 1:]:
                if prob[cat_id] > beta:
                    category = Category(
                        self.category_vocab.itos[self.word_to_whole[cat_id]], self.word_to_whole[cat_id], vector, score[cat_id])
                    chart[(idx, idx + 1)].add_category(category)

        return chart

    @torch.no_grad()
    def parse(self, sentence):
        chart = self.initialize_chart(sentence)
        vector_list = self.tokenize(sentence)

        for i in range(n):
            output = self.softmax(self.linear(vector_list[i]))
            predict = torch.topk(output, k=mergin)
            for P, A in zip(predict[0], predict[1]):
                category_table[i][i + 1].append(A)
                prob[(i, i + 1, A)] = P
                vector[(i, i + 1, A)] = vector_list[i]
        binary_time = 0
        unary_time = 0
        cut_off_time = 0
        for length in range(2, n + 1):
            for i in range(n - length + 1):
                start = time.time()
                j = i + length
                for k in range(i + 1, j):
                    for S1 in category_table[i][k]:
                        for S2 in category_table[k][j]:
                            # list of gramatically possible category
                            possible_cat = self.binary_rule[S1][S2]
                            if possible_cat == []:
                                continue
                            else:
                                composed_vector = single_circular_correlation(
                                    vector[(i, k, S1)], vector[(k, j, S2)])
                                prob_dist = self.softmax(self.linear(composed_vector))
                                possible_cat_prob = torch.index_select(
                                    input=prob_dist, dim=-1, index=torch.tensor(possible_cat))
                                for A, P in zip(possible_cat, possible_cat_prob):
                                    if A not in category_table[i][j]:
                                        category_table[i][j].append(A)
                                    P = P * prob[(i, k, S1)] * prob[(k, j, S2)]
                                    if P > prob[(i, j, A)]:
                                        prob[(i, j, A)] = P
                                        backpointer[(i, j, A)] = torch.tensor([k, S1, S2])
                                        vector[(i, j, A)] = composed_vector
                binary_time += time.time() - start
                start = time.time()
                again = True
                while again:
                    again = False
                    for S in category_table[i][j]:
                        possible_cat = self.unary_rule[S]
                        if possible_cat == []:
                            continue
                        else:
                            prob_dist = self.softmax(self.linear(vector[(i, j, S)]))
                            possible_cat_prob = torch.index_select(
                                input=prob_dist, dim=-1, index=torch.tensor(possible_cat))
                            for A, P in zip(possible_cat, possible_cat_prob):
                                if A not in category_table[i][j]:
                                    category_table[i][j].append(A)
                                P = P * prob[(i, j, S)]
                                if P > prob[(i, j, A)]:
                                    prob[(i, j, A)] = P
                                    backpointer[(i, j, A)] = torch.tensor([0, S, 0])
                                    vector[(i, j, A)] = vector[(i, j, S)]
                                    again = True
                    unary_time += time.time() - start
                    start = time.time()
                    category_table = self.cut_off(category_table, prob, i, j)
                    cut_off_time += time.time() - start
        print(binary_time, unary_time, cut_off_time)
        node_list = self.reconstruct_tree(category_table, backpointer, n)
        return node_list

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
    binary_rule = [[[] for i in range(len(category_vocab))] for j in range(len(category_vocab))]
    unary_rule = [[] for i in range(len(category_vocab))]

    f = open(path_to_grammar, 'r')
    data = f.readlines()
    f.close()

    for rule in data:
        tokens = rule.split()
        if len(tokens) == 6:
            parent_cat = category_vocab[tokens[2]]
            left_cat = category_vocab[tokens[4]]
            right_cat = category_vocab[tokens[5]]
            binary_rule[left_cat][right_cat].append(parent_cat)
        elif len(tokens) == 5:
            parent_cat = category_vocab[tokens[2]]
            child_cat = category_vocab[tokens[4]]
            unary_rule[child_cat].append(parent_cat)
    return binary_rule, unary_rule


def main():
    condition = Condition_Setter(set_embedding_type=False)

    device = torch.device('cpu')

    model = "roberta-large_phrase.pth"

    tree_net = torch.load(model,
                          map_location=device)
    tree_net.device = device
    tree_net.eval()

    category_vocab = load(condition.path_to_whole_category_vocab)
    word_to_whole = load(condition.path_to_word_to_whole)
    whole_to_phrase = load(condition.path_to_whole_to_phrase)

    parser = Parser(tree_net, None, None, category_vocab, word_to_whole, whole_to_phrase)
    sentence = "My sister loves to eat ."
    parser.parse(sentence)


if __name__ == "__main__":
    main()
