from utils import Condition_Setter, load


def convert_line(line, vocab):
    converted = []
    idx = 0
    while True:
        char = line[idx]
        if char in ['(', ')']:
            converted.append(char)
            idx += 1
        elif char == '<':
            k = 1
            while True:
                if line[idx + k] == '>':
                    break
                else:
                    k += 1
            temp = line[idx + 1:idx + k].split()
            if temp[0] == 'T':
                converted.append(str(vocab[temp[1]]))
            else:
                converted.append(str(vocab[temp[1]]))
                converted.append(' ')
                converted.append(temp[4])
            idx += k + 1
        else:
            idx += 1
        if idx == len(line):
            break
    return "".join(converted)


def convert(lines, converted_list, vocab):
    for line in lines:
        if line == "(<L NP NN NN fail N>)\n":
            converted_list.append('\n')
        elif line[0] == '(':
            converted_list.append(convert_line(line, vocab) + '\n')
    return converted_list


condition = Condition_Setter(set_embedding_type=False)
vocab = load(condition.path_to_evalb_category_vocab)
path_to_file_table = condition.PATH_TO_DIR + 'CCGbank/ccgbank_1_1/doc/file.tbl'
f = open(path_to_file_table, 'r')
path_to_data = f.readlines()

predict = []
dev_gold = []
test_gold = []

i = 0
for path in path_to_data:
    if '.auto' in path:
        idx = int(path[-10:-6])
        path = condition.PATH_TO_DIR + 'CCGbank/ccgbank_1_1/' + path.replace('\n', '')
        if idx < 100:
            with open(path) as f:
                lines = f.readlines()
                dev_gold = convert(lines, dev_gold, vocab)
        elif 2300 <= idx and idx < 2400:
            with open(path) as f:
                lines = f.readlines()
                test_gold = convert(lines, test_gold, vocab)

with open('parse_result1.txt') as f:
    parse_result = f.readlines()
predict = convert(parse_result, predict, vocab)

f = open("dev_gold.txt", "w")
f.writelines(dev_gold)
f.close()
f = open("test_gold.txt", "w")
f.writelines(test_gold)
f.close()
f = open("predict.txt", "w")
f.writelines(predict)
f.close()
