from utils import load, Condition_Setter
import torch
import stanza
from tqdm import tqdm

condition = Condition_Setter(set_embedding_type=False)
device = torch.device("cpu")
word_category_vocab = load(condition.path_to_word_category_vocab)
tree_net = torch.load("lstm_with_two_classifiers.pth", map_location=device)
tree_net.device = device
word_classifier = tree_net.word_classifier
pos_tagger = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos')
softmax = torch.nn.Softmax(dim=-1)


path_to_dev_sentence = condition.PATH_TO_DIR + "CCGbank/ccgbank_1_1/data/RAW/CCGbank.00.raw"
with open(path_to_dev_sentence, "r") as f:
    dev_sentence_list = f.readlines()

parser_input = []
with tqdm(total=len(dev_sentence_list)) as pbar:
    pbar.set_description("setting up parser input")
    for sentence in dev_sentence_list:
        sentence = sentence.split()
        # make the set of super-tags
        converted_sentence = []
        for i in range(len(sentence)):
            content = sentence[i]
            if content == "-LRB-" or content == "-LCB-":
                content = "("
            elif content == "-RRB-" or content == "-RCB-":
                content = ")"
            elif r"\/" in content:
                content = content.replace(r"\/", "/")
            converted_sentence.append(content)
        word_vectors = tree_net.cal_word_vectors(converted_sentence)
        word_outputs = word_classifier(word_vectors).to(torch.float64)
        topk_cat_id = torch.topk(word_outputs, k=5, dim=1)[1]
        word_prob_dist = softmax(word_outputs)
        max_prob_list = torch.max(word_prob_dist, dim=1)[0]
        super_tags = []
        for cat_id, prob_dist, max_prob in zip(topk_cat_id, word_prob_dist, max_prob_list):
            temp = []
            for id in cat_id:
                # if prob_dist[id] >= max_prob * 1e-4:
                #     temp.append(word_category_vocab.itos[id])
                #     temp.append(str(prob_dist[id].item()))
                # else:
                #     super_tags.append(temp)
                #     break
                temp.append(word_category_vocab.itos[id])
                temp.append(str(prob_dist[id].item()))
            super_tags.append(temp)

        # # make the set of pos-tags for each sentence
        # tagged_sentence = pos_tagger(" ".join(sentence)).sentences[0]
        # pos_tags = []
        # for word in tagged_sentence.words:
        #     pos_tags.append([word.xpos, str(1.0)])
        pos_tags = []
        for i in range(len(sentence)):
            pos_tags.append(["Pos", "1.0"])

        # concat word|pos|super
        temp = []
        for word, pos, super in zip(sentence, pos_tags, super_tags):
            temp.append("\t".join([word, "\t".join(pos), "\t".join(super)]))
        parser_input.append('|'.join(temp) + '\n')
        pbar.update(1)

f = open("parser_input.txt", "w")
f.writelines(parser_input)
f.close()
