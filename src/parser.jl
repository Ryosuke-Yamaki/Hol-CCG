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
    idx = input .>= max_prob * beta 
    return input[idx], Vector{Int}(1:length(input))[idx]
end

function top_k(input::Vector{Float64},k::Int)
    idx = sortperm(input,rev=true)[1:k]
    return input[idx], idx
end

#time:ok, type:ok
function nonzero_index(input::Vector{Bool})
    idx = input .== 1
    return Vector{Int}(1:length(input))[idx]
end

#time:ok, type:ok msec-order
function initialize_chart(word_vectors::Matrix{Float32},word_classifier_weight::Matrix{Float64},word_classifier_bias::Vector{Float64},word_to_whole::Vector{Int},beta::Float64)
    n = size(word_vectors,1)
    category_table = Dict{Tuple{Int,Int},Vector{Int}}()
    prob = Dict{Tuple{Int,Int,Int},Float64}()
    backpointer = Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}}()
    for i=1:n
        vector = word_classifier_weight*word_vectors[i,:]+word_classifier_bias
        output = softmax(vector)
        # P, A = top_beta(output,beta*10)
        P, A = top_k(output,5)
        category_table[(i,i+1)] = word_to_whole[A]
        for j = 1:length(P)
            prob[(i,i+1,word_to_whole[A[j]])] = P[j]
        end
    end
    return category_table, prob, backpointer
end

#time:ok, type:ok msec-order
function fill_binary(category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Float64},backpointer::Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}},binary_rule::Array{Bool},binary_prob::Dict{Tuple{Int,Int,Int},Float16},phrase_classifier_weight::Matrix{Float64}, phrase_classifier_bias::Vector{Float64},whole_to_phrase::Vector{Int},i::Int,j::Int)
    category_table[(i,j)] = []
    for k = i+1:j-1
        S1_list = category_table[(i,k)]
        S2_list = category_table[(k,j)]
        for S1 in S1_list
            for S2 in S2_list
                A_list = nonzero_index(binary_rule[S1,S2,:])
                if length(A_list) != 0
                    # vector = circular_correlation(prob[(i,k,S1)][2],prob[(k,j,S2)][2])
                    # output = softmax(phrase_classifier_weight*vector+phrase_classifier_bias)
                    for A in A_list
                        # P = output[whole_to_phrase[A]] * prob[(i,k,S1)][1] * prob[(k,j,S2)][1] * binary_prob[(A,S1,S2)]
                        P = prob[(i,k,S1)] * prob[(k,j,S2)] * binary_prob[(A,S1,S2)]
                        # P = output[A]
                        if A in category_table[(i,j)]
                            if P > prob[(i,j,A)]
                                prob[(i,j,A)] = P
                                backpointer[(i,j,A)] = (k,S1,S2)
                            end
                        else 
                            append!(category_table[(i,j)],A)
                            prob[(i,j,A)] = P
                            backpointer[(i,j,A)] = (k,S1,S2)
                        end
                    end
                end
            end
        end
    end
    return category_table, prob, backpointer
end

function fill_unary(category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Float64},backpointer::Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}},unary_rule::Array{Bool},unary_prob::Dict{Tuple{Int,Int},Float16},phrase_classifier_weight::Matrix{Float64},phrase_classifier_bias::Vector{Float64},whole_to_phrase::Vector{Int},i::Int,j::Int)
    if length(category_table[(i,j)]) > 0    
        waiting_list = copy(category_table[(i,j)])
        while true
            S = pop!(waiting_list)
            A_list = nonzero_index(unary_rule[S,:])
            if length(A_list) != 0
                # vector = prob[(i,j,S)][2]
                # output = softmax(phrase_classifier_weight*vector+phrase_classifier_bias)
                for A in A_list
                    # P = output[whole_to_phrase[A]] * prob[(i,j,S)][1] * unary_prob[(A,S)]
                    P = prob[(i,j,S)] * unary_prob[(A,S)]
                    # P = output[whole_to_phrase[A]]
                    # P = output[A]
                    if A in category_table[(i,j)]
                        if P > prob[(i,j,A)]
                            prob[(i,j,A)] = P
                            backpointer[(i,j,A)] = (0,S,0)
                        end
                    else 
                        append!(category_table[(i,j)],A)
                        prob[(i,j,A)] = P
                        backpointer[(i,j,A)] = (0,S,0)
                    end
                    if (S,A) in [(21,2),(2,3),(21,3),(62,8),(62,132)]
                        append!(waiting_list,A)
                    end
                end
            end
            if length(waiting_list) == 0
                break
            end
        end
    end
    return category_table, prob, backpointer
end

function cut_off_beta(category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Float64},beta::Float64,i::Int,j::Int)
    if length(category_table[(i,j)] ) != 0
        prob_list = Float64[]
        for category in category_table[(i,j)]
            append!(prob_list,prob[(i,j,category)])
        end
        idx = prob_list .>= maximum(prob_list) * beta
        category_table[(i,j)] = category_table[(i,j)][idx]
    end
    return category_table
end

function cut_off_k(category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Float64},k::Int,i::Int,j::Int)
    if length(category_table[(i,j)]) > k
        prob_list = Float64[]
        for category in category_table[(i,j)]
            append!(prob_list,prob[(i,j,category)])
        end
        idx = sortperm(prob_list,rev=true)[1:k]
        category_table[(i,j)] = category_table[(i,j)][idx]
    end
    return category_table
end

#type:ok
function cky_parse(word_vectors::Matrix{Float32},word_classifier_weight::Matrix{Float64},word_classifier_bias::Vector{Float64},phrase_classifier_weight::Matrix{Float64},phrase_classifier_bias::Vector{Float64},beta::Float64,binary_rule::Array{Bool},unary_rule::Matrix{Bool},binary_prob::Dict{Tuple{Int,Int,Int},Float16},unary_prob::Dict{Tuple{Int,Int},Float16},word_to_whole::Vector{Int},whole_to_phrase::Vector{Int})
    category_table, prob, backpointer = initialize_chart(word_vectors,word_classifier_weight,word_classifier_bias,word_to_whole,beta)
    n = size(word_vectors,1)
    for l = 2:n
        for i = 1:n-l+1
            j = i + l
            category_table, prob, backpointer = fill_binary(category_table,prob,backpointer,binary_rule,binary_prob,phrase_classifier_weight,phrase_classifier_bias,whole_to_phrase,i,j)
            category_table, prob, backpointer = fill_unary(category_table,prob,backpointer,unary_rule,unary_prob,phrase_classifier_weight,phrase_classifier_bias,whole_to_phrase,i,j)
            # category_table = cut_off_beta(category_table,prob,beta,i,j)
            category_table = cut_off_k(category_table,prob,5,i,j)
        end
    end
    return category_table, prob, backpointer
end

function follow_backpointer(sentence::String,category_table::Dict{Tuple{Int,Int},Vector{Int}},prob::Dict{Tuple{Int,Int,Int},Float64},backpointer::Dict{Tuple{Int,Int,Int},Tuple{Int,Int,Int}})
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
            append!(prob_list,prob[(1,n+1,category)])
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
        if length(predict) == 0
            precision = 0
        else
            precision = precision = n/(length(predict))
        end
        
        m = 0
        for info in correct
            if info in predict
                m += 1
            end
        end
        recall = m/(length(correct))

        if precision == 0 && recall == 0
            f1 = 0
        else
            f1 = (2*precision*recall)/(precision+recall)
        end

        return f1, precision, recall
    end
end