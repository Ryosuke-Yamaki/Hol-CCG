import sys

path_to_ccgbank_deps = sys.argv[1]
path_to_autos = sys.argv[2]

with open(path_to_ccgbank_deps, 'r') as f:
    ccgbank_deps = f.readlines()
with open(path_to_autos, 'r') as f:
    autos = f.readlines()
with open(path_to_ccgbank_deps + '1', 'w') as f:
    f.writelines(ccgbank_deps[:3])

    ccgbank_deps = ccgbank_deps[3:]
    sentence_id = 1
    for deps in ccgbank_deps:
        if deps == '\n':
            auto = autos[sentence_id * 2 - 1]
            auto = auto.split()
            tag_info = ['<c>']
            for i in range(len(auto)):
                token = auto[i]
                if token == '(<L':
                    word = auto[i + 4]
                    pos = auto[i + 2]
                    cat = auto[i + 1]
                    tag_info.append('|'.join([word, pos, cat]))
            f.write(' '.join(tag_info) + '\n')
            sentence_id += 1
        f.write(deps)
