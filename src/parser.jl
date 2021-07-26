using FFTW
using LinearAlgebra

function circular_correlation(a::Vector{Float64},b::Vector{Float64})
    a = conj(fft(a))
    b = fft(b)
    return normalize(real(ifft(a.*b)),2)
end

function tokenize(sentence::String,content_vocab::Dict{String,Int},embedding_weight::Matrix{Float64})
    words = split(sentence)
    vector_list = zeros(Float64,size(words,1),size(embedding_weight,2))
    for i = 1:size(words,1)
        word = lowercase(words[i])
        word = replace(word,r"\d+"=>'0')
        word = replace(word,r"\d,\d+"=>'0')
        content_id = content_vocab[word]
        vector = embedding_weight[content_id,:]
        vector_list[i,:] = vector/norm(vector,2)
    end
    return vector_list
end

function softmax(input::Vector{Float64})
    return exp.(input)/sum(exp.(input))
end

function top_beta(input::Vector{Float64},beta::Float64)
    max_prob = maximum(input)
    sorted_prob = sort(input,rev=true)
    sorted_idx = sortperm(input,rev=true)
    idx = 0
    for i=2:size(sorted_prob,1)
        if sorted_prob[i] < max_prob * beta
            idx = i - 1
            break
        end
        idx = i
    end
    return sorted_prob[1:idx], sorted_idx[1:idx]
end

function nonzero_index(input::Vector{Int})
    return sortperm(input,rev=true)[1:length(input[input.==1])]
end
        
function cky_parse(sentence::String,content_vocab::Dict{String,Int},embedding_weight::Matrix{Float64},linear_weight::Matrix{Float64},beta::Float64,binary_rule::Matrix{Int},unary_rule::Matrix{Int})
    vector_list = tokenize(sentence,content_vocab,embedding_weight)
    n = size(vector_list,1)
    category_table = zeros(Int,n+1,n+1,size(linear_weight,1))
    prob = zeros(Float64,n+1,n+1,size(linear_weight,1))
    backpointer = zeros(Int,n+1,n+1,size(linear_weight,1),3)
    vector_table = zeros(Float64,n+1,n+1,size(linear_weight,1),size(embedding_weight,2))
    for i=1:n
        output = softmax(linear_weight*vector_list[i,:])
        P, A = top_beta(output,beta)
        for j = 1:size(P,1)
            category_table[i,i+1,A[j]] = 1
            prob[i,i+1,A[j]] = P[j]
            vector_table[i,i+1,A[j],:] = vector_list[i,:]
        end
    end

    for l=2:n+1
        for i=i+l
            j = i+l
            for k = i+1:j
                S1_list = nonzero_index(category_table[i,k,:])
                S2_list = nonzero_index(category_table[k,j,:])
                for idx1 = 1:length(S1_list)
                    S1 = S1_list[idx1]
                    for idx2 = 1:length(S2_list)
                        S2 = S2_list[idx2]
                        possible_cat = nonzero_index(binary_rule[S1,S2,:])
                        if length(possible_cat) == 0
                            continue
                        else
                            composed_vector = circular_correlation(vector_table[i,k,S1,:],vector_table[k,j,S2,:])
                            prob_dist = softmax(linear_weight*composed_vector)
                            for idx3 = 1:length(possible_cat)
                                A = possible_cat[idx3]
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
                for idx4 = 1:length(S_list)
                    S = S_list[idx4]
                    possible_cat = nonzero_index(unary_rule[S,:])
                    if length(possible_cat) == 0
                        continue
                    else
                        prob_dist = softmax(linear_weight*vector_table[i,j,S,:])
                        for idx5 = 1:length(possible_cat)
                            A = possible_cat[idx5]
                            category_table[i,j,A] = 1
                            P = prob_dist[A] * prob[(i,j,S)]
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