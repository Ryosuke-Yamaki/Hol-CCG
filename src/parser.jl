using FFTW
using LinearAlgebra

#time:ok, type:ok
function circular_correlation(a::Vector{Float64},b::Vector{Float64})
    freq_a::Vector{ComplexF64} = conj(fft(a))
    freq_b::Vector{ComplexF64} = fft(b)
    c::Vector{Float64} = normalize(real(ifft(freq_a.*freq_b)),2)
    return c/norm(c,2)
end

#time:ok, type:ok
function tokenize(sentence::String,content_vocab::Dict{String,Int},embedding_weight::Matrix{Float64})
    words = split(sentence)
    vector_list = zeros(Float64,size(words,1),size(embedding_weight,2))
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
function softmax(input::Vector{Float64})
    output = exp.(input)/sum(exp.(input))
    return output
end

#time:ok, type:ok
function top_beta(input::Vector{Float64},beta::Float64)
    max_prob = maximum(input)
    idx = input .> max_prob * beta
    return input[idx], Vector{Int}(1:length(input))[idx]
end

#time:ok, type:ok
function nonzero_index(input::Vector{Bool})
    idx = input .== 1
    return Vector{Int}(1:length(input))[idx]
end

#time:ok, type:ok msec-order
function initialize_chart(vector_list::Matrix{Float64},linear_weight::Matrix{Float64},linear_bias::Vector{Float64},beta::Float64)
    n = size(vector_list,1)
    category_table = Dict{Tuple{Int,Int},Vector{Int}}()
    prob = Dict{Tuple{Int,Int,Int},Tuple{Float64,Vector{Float64}}}()
    backpointer = Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}}()
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
function fill_binary(category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Tuple{Float64,Vector{Float64}}},backpointer::Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}},binary_rule::Array{Bool},linear_weight::Matrix{Float64},linear_bias::Vector{Float64},i::Int,j::Int)
    category_table[(i,j)] = []
    for k = i+1:j-1
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
                        # P = output[A]
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
function fill_unary(category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Tuple{Float64,Vector{Float64}}},backpointer::Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}},unary_rule::Array{Bool},linear_weight::Matrix{Float64},linear_bias::Vector{Float64},i::Int,j::Int)
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
                    # P = output[A]
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

function cut_off_beta(category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Tuple{Float64,Vector{Float64}}},beta::Float64,i::Int,j::Int)
    if length(category_table[(i,j)] ) != 0
        prob_list = Float64[]
        for category in category_table[(i,j)]
            append!(prob_list,prob[(i,j,category)][1])
        end
        idx = prob_list .> maximum(prob_list) * beta
        category_table[(i,j)] = category_table[(i,j)][idx]
    end
    return category_table
end

#type:ok
function cky_parse(sentence::String,content_vocab::Dict{String,Int},embedding_weight::Matrix{Float64},linear_weight::Matrix{Float64},linear_bias::Vector{Float64},beta::Float64,binary_rule::Array{Bool},unary_rule::Matrix{Bool})
    vector_list = tokenize(sentence,content_vocab,embedding_weight)
    category_table, prob, backpointer = initialize_chart(vector_list,linear_weight,linear_bias,beta)
    n = size(vector_list,1)
    for l = 2:n
        for i = 1:n-l+1
            j = i + l
            category_table, prob, backpointer = fill_binary(category_table,prob,backpointer,binary_rule,linear_weight,linear_bias,i,j)
            category_table, prob, backpointer = fill_unary(category_table,prob,backpointer,unary_rule,linear_weight,linear_bias,i,j)
            category_table = cut_off_beta(category_table,prob,beta,i,j)
        end
    end
    return category_table, prob, backpointer
end

function follow_backpointer(sentence::String,category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Tuple{Float64,Vector{Float64}}},backpointer::Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}})
    waiting_list = []
    predict = []
    n = length(split(sentence))
    # when parsing was failed, or one word sentence
    if length(category_table[(1,n+1)]) == 0 || n == 1
        return predict
    # find top probability for whole sentence
    else
        prob_list = Float64[]
        for category in category_table[(1,n+1)]
            append!(prob_list,prob[(1,n+1,category)][1])
        end
        max_category = category_table[(1,n+1)][prob_list.==maximum(prob_list)][1]
        push!(predict,(1,n+1,max_category))
    end
    
    # for unary_rule
    if backpointer[(1,n+1,max_category)][1] == 0
        push!(waiting_list,(1,n+1,backpointer[(1,n+1,max_category)][2]))
    # for binary_rule
    else
        divide_point = backpointer[(1,n+1,max_category)][1]
        left_cat = backpointer[(1,n+1,max_category)][2]
        right_cat = backpointer[(1,n+1,max_category)][3]
        # when not leaf node
        if divide_point - 1 > 1
            push!(waiting_list,(1,divide_point,left_cat))
        end
        # when not leaf node
        if n + 1 - divide_point > 1
            push!(waiting_list,(divide_point,n+1,right_cat))
        end
    end

    while length(waiting_list) != 0
        info = pop!(waiting_list)
        push!(predict,info)
        # for unary_rule
        if backpointer[info][1] == 0
            push!(waiting_list,(info[1],info[2],backpointer[info][2]))
        else
            divide_point = backpointer[info][1]
            left_cat = backpointer[info][2]
            right_cat = backpointer[info][3]
            # when not leaf node
            if divide_point - info[1] > 1
                push!(waiting_list,(info[1],divide_point,left_cat))
            end
            # when not leaf node
            if info[2] - divide_point > 1
                push!(waiting_list,(divide_point,info[2],right_cat))
            end
        end
    end
    return predict
end

function f1_score(predict,correct)
    if length(correct) == 0
        return 1, 1, 1
    else
        n = 0
        for info in predict
            if info in correct
                n += 1
            end
        end
        precision = n/(length(predict)+1e-6)
        
        m = 0
        for info in correct
            if info in predict
                m += 1
            end
        end
        recall = m/(length(correct)+1e-6)

        f1 = (2*precision*recall)/(precision+recall+1e-6)

        return f1, precision, recall
    end
end