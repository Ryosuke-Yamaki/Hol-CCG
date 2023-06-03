import numpy as np
import os
import argparse
from typing import List, Tuple


def arg_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_to_dataset', type=str, default='../dataset/', help='path to dataset')

    args = parser.parse_args()
    return args


def is_number(s: str) -> bool:
    """Check if string is number.

    Parameters
    ----------
    s : str
        string to check

    Returns
    -------
    bool
        whether string is number
    """
    try:
        float(s.replace(',', '').replace(' ', ''))
    except ValueError:
        return False
    else:
        return True


def load_corpus(path_to_dataset: str) -> Tuple[List[str], int]:
    """Load corpus.

    Parameters
    ----------
    path_to_dataset : str
        path to dataset

    Returns
    -------
    Tuple[List[str], int]
        corpus and number of words in corpus
    """

    corpus = []
    N = 0
    for i in range(0, 24):
        i = str(i).zfill(2)
        path_to_raw = f'ccgbank_1_1/data/RAW/CCGbank.{i}.raw'
        path_to_raw = os.path.join(path_to_dataset, path_to_raw)
        with open(path_to_raw) as f:
            data = f.readlines()
        for sentence in data:
            temp = []
            for word in sentence.split():
                if word not in [',', '.']:
                    if is_number(word):
                        word = '0'
                    temp.append(word.lower())
            N += len(temp)
            corpus.append(' '.join(temp))
    return corpus, N


def calculate_pmi(candidates_list: List[List[str]], targets_list: List[List[str]], corpus: List[str], N: int) -> None:
    """Calculate PMI.

    Parameters
    ----------
    candidates_list : List[List[str]]
        list of candidates
    targets_list : List[List[str]]
        list of targets
    corpus : List[str]
        corpus
    N : int
        number of words in corpus"""
    pmi_list = []
    for candidates, targets in zip(candidates_list, targets_list):
        for candidate in candidates:
            candidate = candidate.split()
            total_pmi = 0
            for c_word in candidate:
                if is_number(c_word):
                    c_word = '0'
                c_word = c_word.lower()
                for target in targets:
                    target = target.split()
                    for t_word in target:
                        if is_number(t_word):
                            t_word = '0'
                        t_word = t_word.lower()
                        c_count = 0
                        t_count = 0
                        ct_count = 0
                        for sentence in corpus:
                            sentence = sentence.split()
                            if sentence.count(c_word) > 0:
                                c_count += sentence.count(c_word)
                                if t_word in sentence:
                                    ct_count += 1
                            t_count += sentence.count(t_word)
                        if ct_count == 0:
                            pmi = 0
                        else:
                            # normalized pmi
                            pmi = (np.log2(N) + np.log2(ct_count) - np.log2(c_count) -
                                   np.log2(t_count)) / (np.log2(N) - np.log2(ct_count))
                        total_pmi += pmi
            print("candidate: " + " ".join(candidate))
            print("pmi: " + str(total_pmi / (len(candidate) * len(target))))
            pmi_list.append(total_pmi / (len(candidate) * len(target)))
            print()

    print("pmi of decomposition: ", np.mean(pmi_list[:18]))
    print("pmi of roberta: ", np.mean(pmi_list[18:]))


def main():
    args = arg_parse()

    # target phrase of infilling
    # first half is from Hol-CCG, second half is from RoBERTa
    targets_list = [['Mr. Vinken'],
                    ['came out'],
                    ['at $ 374.19 an ounce'],
                    ['what they deserve'],
                    ['Despite recent declines in yields'],
                    ['to pour cash into money funds'],
                    ['Mr. Vinken'],
                    ['came out'],
                    ['at $ 374.19 an ounce'],
                    ['what they deserve'],
                    ['Despite recent declines in yields'],
                    ['to pour cash into money funds']]

    # candidate of infilling
    # first half is from Hol-CCG, second half is from RoBERTa
    candidates_list = [["Mr. Baris",
                        "Dr. Novello",
                        "Ms. Ensrud"],
                       ["turned up",
                        "sold out",
                        "sells out"],
                       ["for $ 25.50 a share",
                        "for $ 60 a bottle",
                        "at $ 51.25 a share"],
                       ["what she did",
                        "what they do",
                        "what we do"],
                       ["Despite the flap over transplants",
                        "In a victory for environmentalists",
                        "On the issue of abortion"],
                       ["to provide maintenance for other manufacturers",
                        "to share data via the telephone",
                        "to cut costs throughout the organization"],
                       ["A.P. Bates",
                        "Ms. Vinken",
                        "Dyearella Sr."],
                       ["was introduced",
                        "went open",
                        "took place"],
                       ["with $ 368.24 an ounce",
                        "as $ 368.79 an ounce",
                        "at $ 368.24 a piece"],
                       ["difficult to defend",
                        "at their views",
                        "out of themselves"],
                       ["To provide a defensive edge",
                        "In a routine shakeup",
                        "After several years of weakness"],
                       ["on a trend toward lower yields",
                        "6 ignore the quake in California",
                        "getting scared out of their lives"]
                       ]

    corpus, N = load_corpus(args.path_to_dataset)

    calculate_pmi(candidates_list, targets_list, corpus, N)


if __name__ == '__main__':
    main()
