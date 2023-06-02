from torch.nn.functional import normalize
from torch.nn.init import kaiming_uniform_
import numpy as np
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
from utils import circular_correlation, circular_convolution, shuffled_circular_convolution, complex_normalize


class HolCCG(nn.Module):
    def __init__(
            self,
            num_word_cat,
            num_phrase_cat,
            model,
            tokenizer,
            model_dim,
            dropout,
            normalize_type,
            vector_norm,
            composition,
            device):
        super(HolCCG, self).__init__()
        self.num_word_cat = num_word_cat
        self.num_phrase_cat = num_phrase_cat
        self.encoder = model
        self.tokenizer = tokenizer
        self.model_dim = model_dim
        self.normalize_type = normalize_type
        if self.normalize_type == 'real':
            self.vector_norm = vector_norm
        elif self.normalize_type == 'complex':
            self.vector_norm = None
        self.composition = composition
        if self.composition == 's_conv':
            self.P = torch.tensor(np.random.permutation(self.model_dim), device=device)
        # the list which to record the modules to set separated learning rate
        self.base_modules = []
        self.base_params = []

        self.linear = nn.Linear(self.model_dim, self.model_dim)
        kaiming_uniform_(self.linear.weight)
        self.base_modules.append(self.linear)
        self.word_classifier = SyntacticClassifier(
            self.model_dim,
            self.model_dim,
            self.num_word_cat,
            dropout=dropout)
        self.phrase_classifier = SyntacticClassifier(
            self.model_dim,
            self.model_dim,
            self.num_phrase_cat,
            dropout=dropout)
        self.span_classifier = SyntacticClassifier(self.model_dim, self.model_dim, 2, dropout=dropout)
        self.base_modules.append(self.word_classifier)
        self.base_modules.append(self.phrase_classifier)
        self.base_modules.append(self.span_classifier)
        for module in self.base_modules:
            for params in module.parameters():
                self.base_params.append(params)
        self.base_params = iter(self.base_params)
        self.device = device

    # input batch as tuple of training info
    def forward(self, batch):
        num_node = batch[0]
        sentence = batch[1]
        original_position = batch[2]
        composition_info = batch[3]
        batch_label = batch[4]
        word_split = batch[5]
        random_num_node = batch[6]
        random_composition_info = batch[7]
        random_original_position = batch[8]
        random_negative_node_id = batch[9]

        vector_list, lengths = self.encode(sentence, word_split)

        # compose word vectors and fed them into FFNN
        original_vector = self.set_leaf_node_vector(
            num_node, vector_list, lengths, original_position)
        # compose word vectors for randomly generated trees
        random_vector = self.set_leaf_node_vector(
            random_num_node, vector_list, lengths, random_original_position)
        original_vector_shape = original_vector.shape
        random_vector_shape = random_vector.shape
        original_vector = original_vector.view(-1, self.model_dim)
        random_vector = random_vector.view(-1, self.model_dim)
        vector = torch.cat((original_vector, random_vector))
        original_vector = vector[:original_vector_shape[0] * original_vector_shape[1],
                                 :].view(original_vector_shape[0], original_vector_shape[1], self.model_dim)
        random_vector = vector[original_vector_shape[0] * original_vector_shape[1]:,
                               :].view(random_vector_shape[0], random_vector_shape[1], self.model_dim)
        composed_vector = self.compose(original_vector, composition_info)
        random_composed_vector = self.compose(random_vector, random_composition_info)
        word_vector, phrase_vector, word_label, phrase_label = self.devide_word_phrase(
            composed_vector, batch_label, original_position)
        span_vector, span_label = self.extract_span_vector(
            phrase_vector, random_composed_vector, random_negative_node_id)

        word_output = self.word_classifier(word_vector)
        phrase_output = self.phrase_classifier(phrase_vector)
        span_output = self.span_classifier(span_vector)
        return word_output, phrase_output, span_output, word_label, phrase_label, span_label

    # encoding word vector
    def encode(self, sentence, word_split):
        input = self.tokenizer(
            sentence,
            padding=True,
            return_tensors='pt').to(self.device)
        word_vector = self.encoder(**input).last_hidden_state[:, 1:-1]
        word_vector_list = []
        lengths = []
        for vector, info in zip(word_vector, word_split):
            temp = []
            for start_idx, end_idx in info:
                temp.append(torch.mean(vector[start_idx:end_idx], dim=0))
            word_vector_list.append(torch.stack(temp))
            lengths.append(len(temp))
        word_vector = pad_sequence(word_vector_list, batch_first=True)
        word_vector = self.linear(word_vector)
        if self.normalize_type == 'real':
            word_vector = self.vector_norm * normalize(word_vector, dim=-1)
        elif self.normalize_type == 'complex':
            word_vector = complex_normalize(word_vector)
        lengths = torch.tensor(lengths, device=torch.device('cpu'))
        return word_vector, lengths

    def set_leaf_node_vector(self, num_node, vector_list, lengths, original_position):
        leaf_node_vector = torch.zeros(
            (len(num_node),
             torch.tensor(max(num_node)),
             self.model_dim), device=self.device)
        for idx in range(len(num_node)):
            batch_id = torch.tensor([idx for _ in range(lengths[idx])])
            # target_id is node.self_id
            target_id = torch.squeeze(original_position[idx][:, 0])
            # source_id is node.original_position
            source_id = torch.squeeze(original_position[idx][:, 1])
            leaf_node_vector[(batch_id, target_id)] = vector_list[(batch_id, source_id)]
        return leaf_node_vector

    def compose(self, vector, composition_info):
        # itteration of composition
        for idx in range(composition_info.shape[1]):
            # the positional index where the composition info of one child is located in batch
            one_child_compositino_idx = torch.squeeze(
                torch.nonzero(composition_info[:, idx, 0] == 1))
            one_child_composition_info = composition_info[composition_info[:, idx, 0] == 1][:, idx]
            one_child_parent_idx = one_child_composition_info[:, 1]
            # the child node index of one child composition
            child_idx = one_child_composition_info[:, 2]
            child_vector = vector[(one_child_compositino_idx, child_idx)]
            vector[(one_child_compositino_idx, one_child_parent_idx)] = child_vector
            two_child_composition_idx = torch.squeeze(
                torch.nonzero(composition_info[:, idx, 0] == 2))
            two_child_composition_info = composition_info[composition_info[:, idx, 0] == 2][:, idx]
            if len(two_child_composition_info) != 0:
                two_child_parent_idx = two_child_composition_info[:, 1]
                # left child node index of two child composition
                left_child_idx = two_child_composition_info[:, 2]
                right_child_idx = two_child_composition_info[:, 3]
                left_child_vector = vector[(two_child_composition_idx, left_child_idx)]
                right_child_vector = vector[(two_child_composition_idx, right_child_idx)]
                if self.composition == 'corr':
                    composed_vector = circular_correlation(
                        left_child_vector, right_child_vector, self.vector_norm)
                elif self.composition == 'conv':
                    composed_vector = circular_convolution(
                        left_child_vector, right_child_vector, self.vector_norm)
                elif self.composition == 's_conv':
                    composed_vector = shuffled_circular_convolution(
                        left_child_vector, right_child_vector, self.P, self.vector_norm)
                vector[(two_child_composition_idx, two_child_parent_idx)] = composed_vector
        return vector

    def devide_word_phrase(self, vector, batch_label, original_position):
        word_vector = []
        phrase_vector = []
        word_label = []
        phrase_label = []
        for i in range(vector.shape[0]):
            word_idx = torch.zeros(len(batch_label[i]), dtype=torch.bool, device=self.device)
            word_idx[original_position[i][:, 0]] = True
            phrase_idx = torch.logical_not(word_idx)
            word_vector.append(vector[i, :len(batch_label[i])][word_idx])
            phrase_vector.append(vector[i, :len(batch_label[i])][phrase_idx])
            word_label.append(
                torch.tensor(
                    batch_label[i],
                    dtype=torch.long,
                    device=self.device)[word_idx])
            phrase_label.append(
                torch.tensor(
                    batch_label[i],
                    dtype=torch.long,
                    device=self.device)[phrase_idx])
        word_vector = torch.cat(word_vector)
        phrase_vector = torch.cat(phrase_vector)
        word_label = torch.squeeze(torch.vstack(word_label))
        phrase_label = torch.squeeze(torch.vstack(phrase_label))
        return word_vector, phrase_vector, word_label, phrase_label

    def extract_span_vector(self, phrase_vector, random_composed_vector, random_negative_node_id):
        # the label for gold spans
        positive_label = torch.ones(phrase_vector.shape[0], dtype=torch.long, device=self.device)

        # the list to contain vectors for negative spans
        negative_span_vector = []
        for i in range(random_composed_vector.shape[0]):
            negative_span_idx = random_negative_node_id[i]
            if negative_span_idx.shape[0] > 0:
                negative_span_vector.append(
                    torch.index_select(
                        random_composed_vector[i],
                        0,
                        negative_span_idx))
        negative_span_vector = torch.cat(negative_span_vector)
        negative_label = torch.zeros(
            negative_span_vector.shape[0],
            dtype=torch.long,
            device=self.device)

        span_vector = torch.cat([phrase_vector, negative_span_vector])
        span_label = torch.cat([positive_label, negative_label])

        return span_vector, span_label

    def set_word_split(self, sentence):
        tokenizer = self.tokenizer
        sentence = " ".join(sentence)
        tokens = tokenizer.tokenize(sentence)
        tokenized_pos = 0
        word_split = []
        for original_position in range(len(sentence.split())):
            word = sentence.split()[original_position]
            word = word.replace(" ", "")
            word = word.replace("\"", "``")
            length = 1
            while True:
                temp = tokenizer.convert_tokens_to_string(
                    tokens[tokenized_pos:tokenized_pos + length])
                temp = temp.replace(" ", "")
                temp = temp.replace("\"", "``")
                if word == temp or word.lower() == temp:
                    word_split.append([tokenized_pos, tokenized_pos + length])
                    tokenized_pos += length
                    break
                else:
                    length += 1
        self.word_split = word_split
        return word_split


class SyntacticClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout=0.2):
        super(SyntacticClassifier, self).__init__()
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)
        self.linear2 = nn.Linear(hidden_dim, output_dim)
        kaiming_uniform_(self.linear1.weight)
        kaiming_uniform_(self.linear2.weight)

    def forward(self, x):
        x = self.linear2(self.dropout(self.relu(self.layer_norm(self.linear1(x)))))
        return x
