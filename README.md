# Holographic CCG (Hol-CCG)
<img src="https://github.com/Ryosuke-Yamaki/Hol-CCG/assets/71750653/8f3538ed-b228-4351-aa5a-8e09abc98bd7" width="500px">

# Prerequisite
## Place the CCGbank (ccgbank_1_1) in `dataset` directory.
## Preprocess CCGbank
```
python preprocessing.py
```

# Usage

## Train Hol-CCG
```
python train.py
```

## Supertagging using trained Hol-CCG
```
python supertagging.py --path_to_sentence [path to sentence to be parsed] --path_to_model [path to trained Hol-CCG]
```

## Span-based Parsing using trained Hol-CCG
```
python span_parser.py --path_to_sentence [path to sentence to be parsed] --path_to_model [path to trained Hol-CCG]
```

# Directory Structure
## `src/` => for source code

## `dataset/` => for dataset
### `dataset/converted/` => for converted format of CCGbank
### `dataset/grammar/` => for gramatical data 
### `dataset/tree_list/` => for converted tree list

## `model/` => for trained model
