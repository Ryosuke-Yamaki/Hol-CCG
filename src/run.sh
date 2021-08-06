#!~/Hol-CCG/src
python data_extraction.py
python make_pretrained_weight_matrix.py
python prepare_tree_list.py
python train_all.py
python regression_all.py
python wave_all.py < run.txt
python visualize_all.py