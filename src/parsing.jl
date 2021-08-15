include("utils.jl")
include("parser.jl")
using DelimitedFiles
using Pickle
using Statistics

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
path_to_correct_list = string(PATH_TO_DIR,"Hol-CCG/data/parsing/correct_list.pkl")

content_vocab = load_content_vocab(path_to_content_vocab)
binary_rule = load_binary_rule(path_to_binary_rule)
unary_rule = load_unary_rule(path_to_unary_rule)
embedding_weight = readdlm(path_to_embedding_weight, ' ', Float64, '\n')
linear_weight = readdlm(path_to_linear_weight, ' ', Float64, '\n')
linear_bias = readdlm(path_to_linear_bias, ' ', Float64, '\n')[:]
sentence_list = load_sentence_list(path_to_raw_sentence)
correct_list = Pickle.load(open(path_to_correct_list))

beta = 0.01
f1 = Float64[]
precision = Float64[]
recall = Float64[]

for i = 1:length(sentence_list)
    sentence = sentence_list[i]
    correct = correct_list[i]
    print(i)
    print(": ")
    println(sentence)
    category_table, prob, backpointer = cky_parse(sentence,content_vocab,embedding_weight,linear_weight,linear_bias,beta,binary_rule,unary_rule)
    predict = follow_backpointer(sentence,category_table,prob,backpointer)
    score = f1_score(predict,correct)
    print("(F1, P, R) = ")
    println(score)
    println()
    append!(f1,score[1])
    append!(precision,score[2])
    append!(recall,score[3])
    if i % 100 == 0
        print("Average(F1, P, R) = ")
        println((mean(f1),mean(precision),mean(recall)))
        println()
    end
end

print("Final average(F1, P, R) = ")
println((mean(f1),mean(precision),mean(recall)))
