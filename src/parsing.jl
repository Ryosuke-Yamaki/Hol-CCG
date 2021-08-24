include("utils.jl")
include("parser.jl")
using DelimitedFiles
using Pickle
using Statistics

# setting path
PATH_TO_DIR = replace(pwd(),"Hol-CCG/src"=>"")
path_to_binary_rule = string(PATH_TO_DIR, "Hol-CCG/data/parsing/binary_rule.txt")
path_to_unary_rule = string(PATH_TO_DIR, "Hol-CCG/data/parsing/unary_rule.txt")
path_to_word_classifier_weight = string(PATH_TO_DIR,"Hol-CCG/data/parsing/word_classifier_weight.csv")
path_to_word_classifier_bias = string(PATH_TO_DIR,"Hol-CCG/data/parsing/word_classifier_bias.csv")
path_to_phrase_classifier_weight = string(PATH_TO_DIR,"Hol-CCG/data/parsing/phrase_classifier_weight.csv")
path_to_phrase_classifier_bias = string(PATH_TO_DIR,"Hol-CCG/data/parsing/phrase_classifier_bias.csv")
path_to_raw_sentence = string(PATH_TO_DIR,"CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw")
path_to_correct_list = string(PATH_TO_DIR,"Hol-CCG/data/parsing/correct_list.pkl")

const binary_rule = load_binary_rule(path_to_binary_rule)
const unary_rule = load_unary_rule(path_to_unary_rule)
const word_classifier_weight = readdlm(path_to_word_classifier_weight, ' ', Float64, '\n')
const word_classifier_bias = readdlm(path_to_word_classifier_bias, ' ', Float64, '\n')[:]
const phrase_classifier_weight = readdlm(path_to_phrase_classifier_weight, ' ', Float64, '\n')
const phrase_classifier_bias = readdlm(path_to_phrase_classifier_bias, ' ', Float64, '\n')[:]
const sentence_list = load_sentence_list(path_to_raw_sentence)
const correct_list = Pickle.load(open(path_to_correct_list))

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
