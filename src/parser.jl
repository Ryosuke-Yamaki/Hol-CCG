using FFTW
using LinearAlgebra

#time:ok, type:ok
function circular_correlation(a::Vector{Float16},b::Vector{Float16})
    freq_a::Vector{ComplexF32} = conj(fft(a))
    freq_b::Vector{ComplexF32} = fft(b)
    c::Vector{Float16} = normalize(real(ifft(freq_a.*freq_b)),2)
    return c
end

#time:ok, type:ok
function tokenize(sentence::String,content_vocab::Dict{String,UInt16},embedding_weight::Matrix{Float16})
    words = split(sentence)
    vector_list = zeros(Float16,size(words,1),size(embedding_weight,2))
    for i = 1:size(words,1)
        word = lowercase(words[i])
        word = replace(word,r"\d+"=>'0')
        word = replace(word,r"\d,\d+"=>'0')
        vector = embedding_weight[content_vocab[word],:]
        vector_list[i,:] = vector/norm(vector,2)
    end
    return vector_list
end

#time:ok, type:ok
function softmax(input::Vector{Float16})
    return exp.(input)/sum(exp.(input))
end

#time:ok, type:ok
function top_beta(input::Vector{Float16},beta::Float16)
    max_prob = maximum(input)
    idx = input .> max_prob * beta
    return input[idx], Vector{UInt16}(1:length(input))[idx]
end

#time:ok, type:ok
function nonzero_index(input::Vector{Bool})
    idx = input .== 1
    return Vector{UInt16}(1:length(input))[idx]
end
        
function cky_parse(sentence::String,content_vocab::Dict{String,UInt16},embedding_weight::Matrix{Float16},linear_weight::Matrix{Float16},beta::Float16,binary_rule::Array{Bool},unary_rule::Matrix{Bool})
    vector_list = tokenize(sentence,content_vocab,embedding_weight)
    n = size(vector_list,1)
    category_table = zeros(Bool,n+1,n+1,size(linear_weight,1))
    prob = zeros(Float16,n+1,n+1,size(linear_weight,1))
    backpointer = zeros(UInt16,n+1,n+1,size(linear_weight,1),3)
    vector_table = zeros(Float16,n+1,n+1,size(linear_weight,1),size(embedding_weight,2))
    for i=1:n
        output = softmax(linear_weight*vector_list[i,:])
        P, A = top_beta(output,beta)
        for j = 1:size(P,1)
            category_table[i,i+1,A[j]] = 1
            prob[i,i+1,A[j]] = P[j]
            vector_table[i,i+1,A[j],:] = vector_list[i,:]
        end
    end

    for l=2:n
        for i=1:n-l+1
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
                            P = prob_dist[A] * prob[i,j,S]
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
    if sum(category_table[1,n+1,:]) != 0
        println("success")
    else
        println("failed")
    end
end