using FFTW
using LinearAlgebra

#time:ok, type:ok
function circular_correlation(a::Vector{Float16},b::Vector{Float16})
    freq_a::Vector{ComplexF32} = conj(fft(a))
    freq_b::Vector{ComplexF32} = fft(b)
    c::Vector{Float16} = normalize(real(ifft(freq_a.*freq_b)),2)
    return c/norm(c,2)
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
    converted = convert(Vector{Float64},input)
    output = exp.(converted)/sum(exp.(converted))
    return convert(Vector{Float16},output)
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

#time:ok, type:ok msec-order
function initialize_chart(vector_list::Matrix{Float16},linear_weight::Matrix{Float16},linear_bias::Vector{Float16},beta::Float16)
    n = size(vector_list,1)
    category_table = Dict{Tuple{UInt8,UInt8},Vector{UInt16}}()
    prob = Dict{Tuple{UInt8,UInt8,UInt16},Tuple{Float16,Vector{Float16}}}()
    backpointer = Dict{Tuple{UInt8,UInt8,UInt16},Tuple{UInt8,UInt16,UInt16}}()
    for i=1:n
        vector = linear_weight*vector_list[i,:]+linear_bias
        output = softmax(vector)
        P, A = top_beta(output,beta)
        category_table[(i,i+1)] = A
        for j = 1:length(P)
            prob[(i,i+1,A[j])] = (P[j],vector_list[i,:])
        end
    end
    return category_table, prob, backpointer
end

#time:ok, type:ok msec-order
function fill_binary(category_table::Dict{Tuple{UInt8,UInt8},Vector{UInt16}},prob::Dict{Tuple{UInt8,UInt8,UInt16},Tuple{Float16,Vector{Float16}}},backpointer::Dict{Tuple{UInt8,UInt8,UInt16},Tuple{UInt8,UInt16,UInt16}},binary_rule::Array{Bool},linear_weight::Matrix{Float16},linear_bias::Vector{Float16},i::UInt8,j::UInt8)
    category_table[(i,j)] = []
    for k_ = i+1:j-1
        k = convert(UInt8,k_)
        S1_list = category_table[(i,k)]
        S2_list = category_table[(k,j)]
        for S1 in S1_list
            for S2 in S2_list
                A_list = nonzero_index(binary_rule[S1,S2,:])
                if length(A_list) != 0
                    vector = circular_correlation(prob[(i,k,S1)][2],prob[(k,j,S2)][2])
                    output = softmax(linear_weight*vector+linear_bias)
                    for A in A_list
                        P = output[A] * prob[(i,k,S1)][1] * prob[(k,j,S2)][1]
                        if A in category_table[(i,j)]
                            if P > prob[(i,j,A)][1]
                                prob[(i,j,A)] = (P,vector)
                                backpointer[(i,j,A)] = (k,S1,S2)
                            end
                        else 
                            append!(category_table[(i,j)],A)
                            prob[(i,j,A)] = (P,vector)
                            backpointer[(i,j,A)] = (k,S1,S2)
                        end
                    end
                end
            end
        end
    end
    return category_table, prob, backpointer
end

#time:ok, type:ok msec-order
function fill_unary(category_table::Dict{Tuple{UInt8,UInt8},Vector{UInt16}},prob::Dict{Tuple{UInt8,UInt8,UInt16},Tuple{Float16,Vector{Float16}}},backpointer::Dict{Tuple{UInt8,UInt8,UInt16},Tuple{UInt8,UInt16,UInt16}},unary_rule::Array{Bool},linear_weight::Matrix{Float16},linear_bias::Vector{Float16},i::UInt8,j::UInt8)
    again = true
    while again
        again = false
        S_list = category_table[(i,j)]
        for S in S_list
            A_list = nonzero_index(unary_rule[S,:])
            if length(A_list) != 0
                vector = prob[(i,j,S)][2]
                output = softmax(linear_weight*vector+linear_bias)
                for A in A_list
                    P = output[A] * prob[(i,j,S)][1]
                    if A in category_table[(i,j)]
                        if P > prob[(i,j,A)][1]
                            prob[(i,j,A)] = (P,vector)
                            backpointer[(i,j,A)] = (0,S,0)
                            again = true
                        end
                    else 
                        append!(category_table[(i,j)],A)
                        prob[(i,j,A)] = (P,vector)
                        backpointer[(i,j,A)] = (0,S,0)
                        again = true
                    end
                end
            end
        end
    end
    return category_table, prob, backpointer
end

#type:ok
function cky_parse(sentence::String,content_vocab::Dict{String,UInt16},embedding_weight::Matrix{Float16},linear_weight::Matrix{Float16},linear_bias::Vector{Float16},beta::Float16,binary_rule::Array{Bool},unary_rule::Matrix{Bool})
    vector_list = tokenize(sentence,content_vocab,embedding_weight)
    category_table, prob, backpointer = initialize_chart(vector_list,linear_weight,linear_bias,beta)
    n = size(vector_list,1)
    for l = 2:n
        for i_ = 1:n-l+1
            i = convert(UInt8,i_)
            j = convert(UInt8,i + l)
            category_table, prob, backpointer = fill_binary(category_table,prob,backpointer,binary_rule,linear_weight,linear_bias,i,j)
            category_table, prob, backpointer = fill_unary(category_table,prob,backpointer,unary_rule,linear_weight,linear_bias,i,j)
        end
    end
    return category_table, prob, backpointer
end