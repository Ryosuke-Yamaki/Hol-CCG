# Holographic CCG (Hol-CCG)

# Prerequisite
- Place the CCGbank (ccgbank_1_1) in `dataset` directory.
- Preprocess CCGbank

    `python preprocessing.py`.

# Usage

- Train Hol-CCG

    `python train.py`.

- Supertagging using trained Hol-CCG

    `python supertagging.py --path_to_sentence [path to sentence to be parsed] --path_to_model [path to trained Hol-CCG]`

- Span-based Parsing using trained Hol-CCG

    `python span_parser.py --path_to_sentence [path to sentence to be parsed] --path_to_model [path to trained Hol-CCG]`

# Directory Structure
## `src/` => for source code

## `dataset/` => for dataset
### `dataset/converted/` => for converted format of CCGbank
### `dataset/grammar/` => for gramatical data 
### `dataset/tree_list/` => for converted tree list

## `model/` => for trained model