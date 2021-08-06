include("utils.jl")
include("parser.jl")
using DelimitedFiles

embedding_type, embedding_dim = condition_set()

# setting path
PATH_TO_DIR = replace(pwd(),"Hol-CCG/src"=>"")
path_to_content_vocab = string(PATH_TO_DIR, "Hol-CCG/data/parsing/content_vocab.txt")
path_to_binary_rule = string(PATH_TO_DIR, "Hol-CCG/data/parsing/binary_rule.txt")
path_to_unary_rule = string(PATH_TO_DIR, "Hol-CCG/data/parsing/unary_rule.txt")
path_to_embedding_weight = string(PATH_TO_DIR,"Hol-CCG/data/parsing/embedding_weight_",embedding_type,"_",embedding_dim,"d.csv")
path_to_linear_weight = string(PATH_TO_DIR,"Hol-CCG/data/parsing/linear_weight_",embedding_type,"_",embedding_dim,"d.csv")
path_to_linear_bias = string(PATH_TO_DIR,"Hol-CCG/data/parsing/linear_bias_",embedding_type,"_",embedding_dim,"d.csv")
path_to_raw_sentence = string(PATH_TO_DIR,"CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw")

embedding_weight = readdlm(path_to_embedding_weight, ' ', Float64, '\n')
linear_weight = readdlm(path_to_linear_weight, ' ', Float64, '\n')
linear_bias = readdlm(path_to_linear_bias, ' ', Float64, '\n')
binary_rule = load_binary_rule(path_to_binary_rule)
unary_rule = load_unary_rule(path_to_unary_rule)
content_vocab = load_content_vocab(path_to_content_vocab)
sentence_list = load_sentence_list(path_to_raw_sentence)