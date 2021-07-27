using Base:Float64
using FFTW
using LinearAlgebra
import Base:parse
using DelimitedFiles

function circular_correlation(a::Vector{Float64}, b::Vector{Float64})
    a = conj(fft(a))
    b = fft(b)
    return normalize(real(ifft(a .* b)), 2)
end

function tokenize(sentence::String, content_vocab::Dict{String,Int}, embedding_weight::Matrix{Float64})
    words = split(sentence)
    vector_list = zeros(Float64, size(words, 1), size(embedding_weight, 2))
    for i = 1:size(words, 1)
        word = lowercase(words[i])
        word = replace(word, r"\d+" => '0')
        word = replace(word, r"\d,\d+" => '0')
        content_id = content_vocab[word]
        vector = embedding_weight[content_id,:]
        vector_list[i,:] = vector / norm(vector, 2)
    end
    return vector_list
end

function softmax(input::Matrix{Float64})
    return exp.(input) / sum(exp.(input))
end

function top_beta(input::Matrix{Float64}, beta::Float64)
    output = zeros(Int, size(input))
    output[input .>= maximum(input) * beta] .= 1
    return output
end

function nonzero_index(input)
    return sortperm(vec(input), rev=true)[1:length(input[input .== 1])]
end
        
function cky_parsing(sentence::String, content_vocab::Dict{String,Int}, embedding_weight::Matrix{Float64}, linear_weight::Matrix{Float64}, linear_bias::Matrix{Float64}, beta::Float64, binary_rule::Array{Bool}, unary_rule::Matrix{Bool})
    vector_list = tokenize(sentence, content_vocab, embedding_weight)
    n = size(vector_list, 1)
    category_table = zeros(Int, n + 1, n + 1, size(linear_weight, 1))
    prob = zeros(Float64, n + 1, n + 1, size(linear_weight, 1))
    backpointer = zeros(Int, n + 1, n + 1, size(linear_weight, 1), 3)
    vector_table = zeros(Float64, n + 1, n + 1, size(linear_weight, 1), size(embedding_weight, 2))
    for i = 1:n
        prob_dist = softmax(linear_weight * vector_list[i,:] + linear_bias)
        A_list = nonzero_index(top_beta(prob_dist, beta))
        for A in A_list
            category_table[i,i + 1,A] = 1
            prob[i,i + 1,A] = prob_dist[A]
            vector_table[i,i + 1,A,:] = vector_list[i,:]
        end
    end

    for l = 2:n
        for i = 1:n - l + 1
            j = i + l
            for k = i + 1:j - 1
                S1_list = nonzero_index(category_table[i,k,:])
                S2_list = nonzero_index(category_table[k,j,:])
                for S1 in S1_list
                    for S2 in S2_list
                        possible_cat = binary_rule[S1,S2,:]
                        if sum(possible_cat) == 0
                            continue
                        else
                            composed_vector = circular_correlation(vector_table[i,k,S1,:], vector_table[k,j,S2,:])
                            prob_dist = softmax(linear_weight * composed_vector + linear_bias)
                            println(size(prob_dist), size(possible_cat))
                            candidate = nonzero_index(top_beta(prob_dist, beta) .== possible_cat)
                            for A in candidate
                                category_table[i,j,A] = 1
                                P = prob_dist[A] * prob[i,k,S1] * prob[k,j,S2]
                                if P > prob[i,j,A]
                                    prob[i,j,A] = P
                                    backpointer[i,j,A,:] = [k,S1,S2]
                                    vector_table[i,j,A,:] = composed_vector
                                end
                            end
                        end
                    end
                end
            end

            again = true
            while again
                again = false
                S_list = nonzero_index(category_table[i,j,:])
                for S in S_list
                    possible_cat = unary_rule[S,:]
                    if sum(possible_cat) == 0
                        continue
                    else
                        prob_dist = softmax(linear_weight * vector_table[i,j,S,:] + linear_bias)
                        candidate = nonzero_index(top_beta(prob_dist, beta) .== possible_cat)
                        for A in candidate
                            category_table[i,j,A] = 1
                            P = prob_dist[A] * prob[(i, j, S)]
                            if P > prob[i,j,A]
                                prob[i,j,A] = P
                                backpointer[i,j,A,:] = [0,S,0]
                                vector_table[i,j,A,:] = vector_table[i,j,S,:]
                                again = true
                            end
                        end
                    end
                end
            end
        end
    end                     
end

function load_content_vocab(path)
    f = open(path)
    content_vocab = Dict{String,Int}()
    for line in eachline(f)
        info = split(line)
        content_vocab[info[1]] = parse(Int, info[2])
    end
    return content_vocab
end

function load_weight_matrix(path)
    return readdlm(path, ' ', Float64, '\n')
end

function load_rule(path, binary)
    f = open(path)
    rule_set = []
    for line in eachline(f)
        push!(rule_set, split(line))
    end
    num_cat = parse(Int, rule_set[1][1])
    if binary
        binary_rule = zeros(Bool, num_cat, num_cat, num_cat)
        for info in rule_set[2:size(rule_set, 1)]
            binary_rule[parse(Int, info[1]),parse(Int, info[2]),parse(Int, info[3])] = true
        end
        return binary_rule
    else
        unary_rule = zeros(Bool, num_cat, num_cat)
        for info in rule_set[2:size(rule_set, 1)]
            unary_rule[parse(Int, info[1]),parse(Int, info[2])] = true
        end
        return unary_rule
    end
end


PATH_TO_DIR = replace(pwd(), "Hol-CCG/src" => "")
path_to_embedding_weight = PATH_TO_DIR * "Hol-CCG/data/parsing/embedding_weight_GloVe_100d.csv"
path_to_linear_weight = PATH_TO_DIR * "Hol-CCG/data/parsing/linear_weight_GloVe_100d.csv"
path_to_linear_bias = PATH_TO_DIR * "Hol-CCG/data/parsing/linear_bias_GloVe_100d.csv"
path_to_binary_rule = PATH_TO_DIR * "Hol-CCG/data/parsing/binary_rule.txt"
path_to_unary_rule = PATH_TO_DIR * "Hol-CCG/data/parsing/unary_rule.txt"
path_to_content_vocab = PATH_TO_DIR * "Hol-CCG/data/parsing/content_vocab.txt"

embedding_weight = load_weight_matrix(path_to_embedding_weight)
linear_weight = load_weight_matrix(path_to_linear_weight)
linear_bias = load_weight_matrix(path_to_linear_bias)
binary_rule = load_rule(path_to_binary_rule, true)
unary_rule = load_rule(path_to_unary_rule, false)
content_vocab = load_content_vocab(path_to_content_vocab)

println(size(linear_weight))
println(size(linear_bias))
println(size(embedding_weight))

path_to_test_sentence = PATH_TO_DIR * "CCGbank/ccgbank_1_1/data/RAW/CCGbank.23.raw"

f = open(path_to_test_sentence)
for sentence in eachline(f)
    println(sentence)
    @time cky_parsing(sentence, content_vocab, embedding_weight, linear_weight, linear_bias, 0.98, binary_rule, unary_rule)
end


println(size(embedding_weight))
println(size(linear_weight))
println(size(linear_bias))
println(size(binary_rule))
println(size(unary_rule))
println(binary_rule[510,144,185:190])



# @time for i=1:100000
#     a = softmax(rand(10))
#     sort(a,rev=true)
#     p, a = top_beta(a,0.5)
# end

# a = [1,0,0,1,0,0,1,1,0]
# @time for i = 1:10000000
#     nonzero_index(a)
# end


# @time for i = 1:100000
#     a = rand(100)
#     b = rand(100)
#     c = circular_correlation(a,b)
# end

embedding_weight = load_weight_matrix("test.csv")
println(embedding_weight[1,:])
# parse(Int,"1")
# path = "test.txt"
# content_vocab = load_content_vocab(path)
# println(content_vocab)



# sentence = "I am Yamaki"
# vocab = Dict{String,Int}()
# vocab["i"] = 1
# vocab["am"] = 2
# vocab["yamaki"] = 3
# embedding_weight = rand(3,5)
# linear_weight = rand(5,5)
# beta = 0.01

# @time for i=1:100
#     tokenize(sentence,vocab,embedding_weight)
# end

# @time for i=1:2000000
#     parse(sentence,vocab,embedding_weight,linear_weight,beta)
# end
# weight_matrix = rand(3,5)
# println(weight_matrix)
# vector_list = tokenize(sentence,vocab,weight_matrix)
# println(vector_list)

# println(weight_matrix == vector_list)