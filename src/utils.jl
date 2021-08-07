function condition_set()
    print("GloVe(0) or random(1): ")
    embedding_type = parse(UInt16,readline())
    if embedding_type != 0 && embedding_type != 1
        print("Error: embedding type")
        exit()
    end
    print("embedding dim: ")
    embedding_dim = parse(UInt16,readline())
    if embedding_type == 0
        if embedding_dim in [50,100,300]
            embedding_type = "GloVe"
        else
            print("Error: embedding dim")
            exit()
        end
    else
        if embedding_dim in [10,50,100,300]
            embedding_type = "random"
        else
            print("Error: embedding dim")
            exit()
        end
    end
    return embedding_type, embedding_dim 
end

function load_content_vocab(path_to_content_vocab::String)
    f = open(path_to_content_vocab)
    info_list = readlines(f)
    content_vocab = Dict{String,UInt16}()
    for info in info_list
        info = split(info)
        content_vocab[info[1]] = parse(UInt16,info[2])
    end 
    return content_vocab
end

function load_binary_rule(path_to_binary_rule::String)
    f = open(path_to_binary_rule)
    info_list = readlines(f)
    num_category = parse(UInt16,info_list[1])
    binary_rule = zeros(Bool,num_category,num_category,num_category)
    for info in info_list[2:end]
        info = split(info)
        left = parse(UInt16,info[1])
        right = parse(UInt16,info[2])
        parent = parse(UInt16,info[3])
        binary_rule[left,right,parent] = 1
    end
    return binary_rule
end

function load_unary_rule(path_to_unary_rule::String)
    f = open(path_to_unary_rule)
    info_list = readlines(f)
    num_category = parse(UInt16,info_list[1])
    unary_rule = zeros(Bool,num_category,num_category)
    for info in info_list[2:end]
        info = split(info)
        child = parse(UInt16,info[1])
        parent = parse(UInt16,info[2])
        unary_rule[child,parent] = 1
    end
    return unary_rule
end

function load_sentence_list(path_to_raw_sentence::String)
    f = open(path_to_raw_sentence)
    sentence_list = readlines(f)
    return sentence_list
end