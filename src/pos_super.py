from utils import load, Condition_Setter
import torch
import stanza
import sys
from tqdm import tqdm

# args = sys.argv
# model = args[1]
# dev_test = args[2]
model = 'bert-base-cased_with_LSTM.pth'
dev_test = 'dev'
condition = Condition_Setter(set_embedding_type=False)
device = torch.device("cpu")
word_category_vocab = load(condition.path_to_word_category_vocab)
tree_net = torch.load(model, map_location=device)
tree_net.device = device
tree_net.eval()
word_classifier = tree_net.word_classifier
pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
softmax = torch.nn.Softmax(dim=-1)
if dev_test == 'dev':
    path_to_sentence = condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw"
elif dev_test == 'test':
    path_to_sentence = condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw"
with open(path_to_sentence, "r") as f:
    sentence_list = f.readlines()
beta = 1e-3

parser_input = []
with tqdm(total=len(sentence_list)) as pbar:
    pbar.set_description("setting up parser input")
    for sentence in sentence_list:
        sentence = sentence.split()
        # make the set of super-tags
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
            elif r"\/" in content:
                content = content.replace(r"\/", "/")
            converted_sentence.append(content)
        word_vectors = tree_net.cal_word_vectors(converted_sentence)
        word_outputs = word_classifier(word_vectors)
        word_cat_prob = softmax(word_outputs)
        predict_cat_id = torch.argsort(word_cat_prob, descending=True)
        super_tags = []
        for idx in range(word_cat_prob.shape[0]):
            temp = []
            max_prob = torch.max(word_cat_prob)
            for cat_id in predict_cat_id[idx]:
                if word_cat_prob[idx, cat_id] > max_prob * beta:
                    temp.append([word_category_vocab.itos[cat_id],
                                 word_cat_prob[idx, cat_id].item()])
                else:
                    super_tags.append(temp)
                    break

        # make the set of pos-tags for each sentence
        tagged_sentence = pos_tagger(" ".join(sentence)).sentences[0]
        pos_tags = []
        for word in tagged_sentence.words:
            pos_tags.append(word.xpos)

        # concat word|pos|super
        for word, pos, super in zip(sentence, pos_tags, super_tags):
            temp = []
            temp.append(word)
            temp.append(pos)
            temp.append(str(len(super)))
            for info in super:
                temp.append(info[0])
                temp.append(str(info[1]))
            parser_input.append('\t'.join(temp))
        parser_input.append('\n')
        pbar.update(1)

f = open("parser_input.txt", "w")
f.writelines(parser_input)
f.close()
