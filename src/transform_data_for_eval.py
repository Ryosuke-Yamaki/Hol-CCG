import sys

path_to_ccgbank_deps = sys.argv[1]
path_to_autos = sys.argv[2]

with open(path_to_ccgbank_deps, 'r') as f:
    ccgbank_deps = f.readlines()
with open(path_to_autos, 'r') as f:
    autos = f.readlines()

    transformed = ccgbank_deps[:3]
    ccgbank_deps = ccgbank_deps[3:]

    i = 0

    for auto in autos:
        if auto.startswith('ID'):
            continue
        else:
            if auto == '(<L N POS POS fail N>)\n':
                i += 1
                transformed.append('\n')
            else:
                if ccgbank_deps[i] != '\n':
                    while True:
                        deps = ccgbank_deps[i]
                        i += 1
                        if deps == '\n':
                            break
                        else:
                            transformed.append(deps)

                    auto = auto.split()
                    tag_info = ['<c>']
                    for j in range(len(auto)):
                        token = auto[j]
                        if token == '(<L':
                            word = auto[j + 4]
                            if word == '-LCB-':
                                word = '{'
                            elif word == '-RCB-':
                                word = '}'
                            elif word == '-LRB-':
                                word = '('
                            elif word == '-RRB-':
                                word = ')'
                            pos = auto[j + 2]
                            cat = auto[j + 1]
                            tag_info.append('|'.join([word, pos, cat]))
                    transformed.append(' '.join(tag_info) + '\n')
                else:
                    i += 1
                transformed.append('\n')


with open(path_to_ccgbank_deps, 'w') as f:
    f.writelines(transformed)
