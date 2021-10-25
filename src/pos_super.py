from utils import load, Condition_Setter
import torch
import stanza
import sys
from tqdm import tqdm

args = sys.argv
model = args[1]
dev_test = args[2]
condition = Condition_Setter(set_embedding_type=False)
device = torch.device("cuda")
word_category_vocab = load(condition.path_to_word_category_vocab)
tree_net = torch.load(condition.path_to_model + model, map_location=device)
tree_net.device = device
tree_net.eval()
word_ff = tree_net.word_ff
pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
if dev_test == 'dev':
    path_to_sentence = condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw"
elif dev_test == 'test':
    path_to_sentence = condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw"
with open(path_to_sentence, "r") as f:
    sentence_list = f.readlines()

beta = 0.0005
alpha = 50

parser_input = []
with tqdm(total=len(sentence_list)) as pbar:
    pbar.set_description("supertagging")
    for sentence in sentence_list:
        sentence = sentence.split()
        # make the set of super-tags
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
        word_vectors = tree_net.cal_word_vectors(converted_sentence)
        word_cat_prob = torch.softmax(word_ff(word_vectors), dim=-1)
        predict_cat_id = torch.argsort(word_cat_prob, descending=True)
        max_prob, _ = torch.max(word_cat_prob, dim=1)
        super_tags = []
        for idx in range(word_cat_prob.shape[0]):
            temp = []
            for cat_id in predict_cat_id[idx, :alpha]:
                if word_cat_prob[idx, cat_id] > max_prob[idx] * beta:
                    temp.append([word_category_vocab.itos[cat_id],
                                 word_cat_prob[idx, cat_id].item()])
                else:
                    super_tags.append(temp)
                    break

        # make the set of pos-tags for each sentence
        temp = pos_tagger(" ".join(sentence)).sentences
        tagged_sentences = pos_tagger(" ".join(sentence)).sentences
        pos_tags = []
        for tagged_sentence in tagged_sentences:
            for word in tagged_sentence.words:
                pos_tags.append(word.xpos)

        for word, pos, super in zip(converted_sentence_, pos_tags, super_tags):
            temp = []
            temp.append(word)
            temp.append(pos)
            temp.append(str(len(super)))
            for info in super:
                temp.append(info[0])
                temp.append(str(info[1]))
            parser_input.append('\t'.join(temp) + '\n')
        parser_input.append('\n')
        pbar.update(1)

output_file = condition.PATH_TO_DIR + "java-candc/data/auto-stagged/" + \
    model.replace('.pth', '_') + dev_test + '.stagged'
f = open(output_file, "w")
f.writelines(parser_input)
f.close()
