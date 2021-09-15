include("utils.jl")
include("parser.jl")
using DelimitedFiles
using Pickle
using Statistics
using PyCall
pushfirst!(PyVector(pyimport("sys")."path"), "")
const torch = pyimport("torch")

# setting path
const PATH_TO_DIR = replace(pwd(),"Hol-CCG/src"=>"")
const path_to_model = string(PATH_TO_DIR,"Hol-CCG/src/lstm_with_two_classifiers.pth")
const path_to_binary_rule = string(PATH_TO_DIR, "Hol-CCG/data/parsing/binary_rule.txt")
const path_to_unary_rule = string(PATH_TO_DIR, "Hol-CCG/data/parsing/unary_rule.txt")
const path_to_binary_prob = string(PATH_TO_DIR, "Hol-CCG/data/parsing/binary_prob.txt")
const path_to_unary_prob = string(PATH_TO_DIR, "Hol-CCG/data/parsing/unary_prob.txt")
const path_to_word_classifier_weight = string(PATH_TO_DIR,"Hol-CCG/data/parsing/word_classifier_weight.csv")
const path_to_word_classifier_bias = string(PATH_TO_DIR,"Hol-CCG/data/parsing/word_classifier_bias.csv")
const path_to_phrase_classifier_weight = string(PATH_TO_DIR,"Hol-CCG/data/parsing/phrase_classifier_weight.csv")
const path_to_phrase_classifier_bias = string(PATH_TO_DIR,"Hol-CCG/data/parsing/phrase_classifier_bias.csv")
const path_to_raw_sentence = string(PATH_TO_DIR,"CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw")
const path_to_word_to_whole = string(PATH_TO_DIR,"Hol-CCG/data/vocab/word_to_whole.pickle")
const path_to_whole_to_phrase = string(PATH_TO_DIR,"Hol-CCG/data/vocab/whole_to_phrase.pickle")
const path_to_correct_list = string(PATH_TO_DIR,"Hol-CCG/data/parsing/correct_list.pkl")

const tree_net = torch[:load](path_to_model,map_location=torch[:device]("cpu"))
const binary_rule = load_binary_rule(path_to_binary_rule)
const unary_rule = load_unary_rule(path_to_unary_rule)
const binary_prob = load_binary_prob(path_to_binary_prob)
const unary_prob = load_unary_prob(path_to_unary_prob)
const word_classifier_weight = readdlm(path_to_word_classifier_weight, ' ', Float64, '\n')
const word_classifier_bias = readdlm(path_to_word_classifier_bias, ' ', Float64, '\n')[:]
const phrase_classifier_weight = readdlm(path_to_phrase_classifier_weight, ' ', Float64, '\n')
const phrase_classifier_bias = readdlm(path_to_phrase_classifier_bias, ' ', Float64, '\n')[:]
const sentence_list = load_sentence_list(path_to_raw_sentence)
const word_to_whole = convert(Vector{Int},Pickle.load(open(path_to_word_to_whole)))
const whole_to_phrase = convert(Vector{Int},Pickle.load(open(path_to_whole_to_phrase)))
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
    word_vectors = tree_net[:cal_word_vectors](convert_sentence(sentence))
    category_table, prob, backpointer = cky_parse(word_vectors,word_classifier_weight,word_classifier_bias,phrase_classifier_weight,phrase_classifier_bias,beta,binary_rule,unary_rule,binary_prob,unary_prob,word_to_whole,whole_to_phrase)
    predict = follow_backpointer(sentence,category_table,prob,backpointer)
    println(correct)
    println(predict)
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
