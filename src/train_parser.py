from logging import root
import sys
from utils import load, Condition_Setter, hinge_loss
import time
import torch
from utils import load
from grammar import Combinator
import numpy as np
from tqdm import tqdm
import torch.optim as optim
from span_parser import Parser


def main():
    condition = Condition_Setter(set_embedding_type=False)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # args = sys.argv
    args = [
        '',
        'roberta-large_phrase_span_2022-01-08_08_54_41.pth',
        '1e-3',
        '1e-3',
        '1e-3',
        '5']

    model = args[1]
    stag_threhold = float(args[2])
    phrase_threshold = float(args[3])
    span_threshold = float(args[4])
    min_freq = int(args[5])

    tree_net = torch.load(condition.path_to_model + model,
                          map_location=device)
    tree_net.device = device
    tree_net.eval()

    with open(condition.PATH_TO_DIR + "span_parsing/GOLD/wsj02-21.raw", 'r') as f:
        train_sentence_list = f.readlines()[3:]
    train_tree_list = load(condition.path_to_train_tree_list)

    train_tree_list.set_info_for_training(tree_net.tokenizer)
    train_sentence_tree_list = list(zip(train_sentence_list, train_tree_list.tree_list))
    train_sentence_tree_list = train_sentence_tree_list

    combinator = Combinator(condition.PATH_TO_DIR + "span_parsing/GRAMMAR/", min_freq=min_freq)

    parser = Parser(
        tree_net,
        combinator=combinator,
        word_category_vocab=train_tree_list.word_category_vocab,
        phrase_category_vocab=train_tree_list.phrase_category_vocab,
        stag_threshold=stag_threhold,
        phrase_threshold=phrase_threshold,
        span_threshold=span_threshold)
    EPOCHS = 100
    BASE_LR = 1e-5
    FT_LR = 1e-6
    optimizer = optim.Adam([{'params': tree_net.base_params},
                            {'params': tree_net.model.parameters(), 'lr': FT_LR}], lr=BASE_LR)

    for epoch in range(EPOCHS):
        total_loss = 0.0
        n_sentence = 0
        n_fail = 0
        np.random.shuffle(train_sentence_tree_list)
        with tqdm(total=len(train_sentence_tree_list), unit="sentence") as pbar:
            pbar.set_description(f"Epoch[{epoch}/{EPOCHS}]")
            for sentence, tree in train_sentence_tree_list:
                n_sentence += 1
                optimizer.zero_grad()
                sentence = sentence.rstrip()
                chart = parser.parse(sentence)
                root_cell = list(chart.values())[-1]
                # success to parse
                if len(root_cell.best_category_id) != 0:
                    root_cat = root_cell.category_list[0]
                    for cat in root_cell.category_list[1:]:
                        if cat.total_ll > root_cat.total_ll:
                            root_cat = cat
                    predict_score = root_cat.total_ll / len(sentence.split())
                    gold_score = tree_net.cal_gold_tree_score(tree)
                    loss = hinge_loss(gold_score, predict_score)
                    loss.backward()
                    optimizer.step()
                    total_loss += loss
                    pbar.set_postfix({"loss": total_loss.item() / n_sentence, "fail": n_fail})
                    pbar.update(1)
                # when fail to parse
                else:
                    n_fail += 1
                    pbar.update(1)


if __name__ == "__main__":
    main()
