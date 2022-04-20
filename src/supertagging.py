from utils import load
import torch
from tqdm import tqdm
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-t', '--target', type=str, choices=['dev', 'test'], required=True)
parser.add_argument('--stag_threshold', type=float, default=0.1)
parser.add_argument('-d', '--device', type=torch.device, default=torch.device('cuda:0'))
parser.add_argument('--failure', action='store_true')

args = parser.parse_args()

args = parser.parse_args()

MODEL = args.model
TARGET = args.target
STAG_THRESHOLD = args.stag_threshold
FAILURE = args.failure
DEVICE = args.device
MODEL_TARGET = MODEL.replace(".pth", "_" + TARGET)

path_to_word_category_vocab = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../data/grammar/word_category_vocab.pickle')
path_to_model = os.path.join(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                          '../data/model/'), MODEL)
if FAILURE:
    path_to_sentence = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../span_parsing/FAILURE/{}.raw".format(MODEL_TARGET))
    path_to_auto_pos = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                    "../span_parsing/FAILURE/{}.auto_pos".format(MODEL_TARGET))
    stagged_file = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                "../span_parsing/FAILURE/{}.stagged".format(MODEL_TARGET))
else:
    if TARGET == 'dev':
        path_to_sentence = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw")
        path_to_auto_pos = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../java-candc/data/auto-pos/wsj00.auto_pos")
        stagged_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../java-candc/data/auto-stagged/" +
            MODEL_TARGET +
            '.stagged')
    elif TARGET == 'test':
        path_to_sentence = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw")
        path_to_auto_pos = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../java-candc/data/auto-pos/wsj23.auto_pos")
        stagged_file = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "../../java-candc/data/auto-stagged/" +
            MODEL_TARGET +
            '.stagged')

word_category_vocab = load(path_to_word_category_vocab)
tree_net = torch.load(path_to_model, map_location=DEVICE)
tree_net.device = DEVICE
tree_net.eval()
word_ff = tree_net.word_ff
with open(path_to_sentence, "r") as f:
    sentence_list = f.readlines()
with open(path_to_auto_pos, "r") as f:
    auto_pos_list = f.readlines()

parser_input = []
with torch.no_grad():
    with tqdm(total=len(sentence_list)) as pbar:
        pbar.set_description("supertagging...")
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
            word_split = tree_net.set_word_split(converted_sentence)
            word_vectors, _ = tree_net.encode([" ".join(converted_sentence)], [word_split])
            word_cat_prob = torch.softmax(word_ff(word_vectors[0]), dim=-1)
            predict_cat_id = torch.argsort(word_cat_prob, descending=True)
            # remove '<unk>'
            predict_cat_id = predict_cat_id[predict_cat_id != 0].view(word_cat_prob.shape[0], -1)
            super_tags = []
            for idx in range(word_cat_prob.shape[0]):
                # add top probability category
                temp = [[word_category_vocab.itos[predict_cat_id[idx, 0]].split('-->')[0],
                        word_cat_prob[idx, predict_cat_id[idx, 0]].item()]]
                for cat_id in predict_cat_id[idx, 1:]:
                    if word_cat_prob[idx, cat_id] > STAG_THRESHOLD:
                        temp.append([word_category_vocab.itos[cat_id].split('-->')[0],
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

with open(stagged_file, "w") as f:
    f.writelines(parser_input)
