# Holographic CCG (Hol-CCG)

# Directory Structure
## `src/` => for source code

## `dataset/` => for dataset
### `dataset/converted/` => for converted format of CCGbank
### `dataset/grammar/` => for gramatical data 
### `dataset/tree_list/` => for converted tree list

## `model/` => for trained model

# Usage
1. Place the CCGbank (ccgbank_1_1) in `dataset` directory.
2. Move to `src` directory.
3. Preprocess CCGbank by `python preprocessing.py`.
3. Train Hol-CCG by `python train.py`.
