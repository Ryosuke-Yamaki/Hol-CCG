from utils import load, Condition_Setter
import torch
import sys
from tqdm import tqdm

args = sys.argv
model = args[1]
dev_test = args[2]
beta = float(args[3])
alpha = args[4]
if alpha == "None":
    alpha = None
else:
    alpha = int(alpha)
# model = "roberta-large_phrase(5).pth"
# dev_test = 'dev'
condition = Condition_Setter(set_embedding_type=False)
device = torch.device("cuda")
word_category_vocab = load(condition.path_to_word_category_vocab)
tree_net = torch.load(condition.path_to_model + model, map_location=device)
tree_net.device = device
tree_net.eval()
word_ff = tree_net.word_ff
if dev_test == 'dev':
    path_to_sentence = condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw"
    path_to_auto_pos = condition.PATH_TO_DIR + "java-candc/data/auto-pos/wsj00.auto_pos"
elif dev_test == 'test':
    path_to_sentence = condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw"
    path_to_auto_pos = condition.PATH_TO_DIR + "java-candc/data/auto-pos/wsj23.auto_pos"

with open(path_to_sentence, "r") as f:
    sentence_list = f.readlines()
with open(path_to_auto_pos, "r") as f:
    auto_pos_list = f.readlines()

parser_input = []
with tqdm(total=len(sentence_list)) as pbar:
    pbar.set_description("supertagging")
    for sentence, pos in zip(sentence_list, auto_pos_list):
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
        # remove '<unk>'
        predict_cat_id = predict_cat_id[predict_cat_id != 0].view(word_cat_prob.shape[0], -1)
        super_tags = []
        for idx in range(word_cat_prob.shape[0]):
            # add top probability category
            temp = [[word_category_vocab.itos[predict_cat_id[idx, 0]],
                     word_cat_prob[idx, predict_cat_id[idx, 0]].item()]]
            for cat_id in predict_cat_id[idx, 1:alpha]:
                if word_cat_prob[idx, cat_id] > beta:
                    temp.append([word_category_vocab.itos[cat_id],
                                 word_cat_prob[idx, cat_id].item()])
                else:
                    break
            super_tags.append(temp)

        pos = pos.strip().split()
        pos_tags = []
        for info in pos:
            pos_tags.append(info.split('|')[1])

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
