import sys


def add_tag_info(auto, tag_info):
    auto = auto.split()
    for i in range(len(auto)):
        token = auto[i]
        if token == '(<L':
            word = auto[i + 4]
            if word == '-LCB-':
                word = '{'
            elif word == '-RCB-':
                word = '}'
            elif word == '-LRB-':
                word = '('
            elif word == '-RRB-':
                word = ')'
            pos = auto[i + 2]
            cat = auto[i + 1]
            tag_info.append('|'.join([word, pos, cat]))
    return tag_info


def add_deps_info(deps, start_idx, temp_deps):
    deps = deps.split()
    info = deps[0].split('_')
    info[1] = str(int(info[1]) + start_idx)
    deps[0] = '_'.join(info)
    info = deps[3].split('_')
    info[1] = str(int(info[1]) + start_idx)
    deps[3] = '_'.join(info)
    deps = ' '.join(deps) + '\n'
    temp_deps.append(deps)
    return temp_deps

args = sys.argv


# args = [
#     '',
#     "/home/yamaki-ryosuke/span_parsing/CCGBANK_DEPS/roberta-large_phrase(b)_dev_0.075_0.01_0.01_10.ccgbank_deps",
#     "/home/yamaki-ryosuke/span_parsing/AUTO/roberta-large_phrase(b)_dev_0.075_0.01_0.01_10.auto"]
path_to_ccgbank_deps = args[1]
path_to_autos = args[2]


with open(path_to_ccgbank_deps, 'r') as f:
    ccgbank_deps = f.readlines()
with open(path_to_autos, 'r') as f:
    autos = f.readlines()

transformed = ccgbank_deps[:3]
ccgbank_deps = ccgbank_deps[3:]

deps_idx = 0
sentence_id = 0

for auto_idx in range(len(autos)):
    auto = autos[auto_idx]
    if auto.startswith('ID'):
        previous_sentence_id = sentence_id
        parse_info = auto.split()
        sentence_id = int(parse_info[0].split('.')[0].split('=')[1])
        apply_skimmer = parse_info[2]
        if 'True' in apply_skimmer:
            apply_skimmer = True
            scope = [int(i) for i in parse_info[3].split('=')[1][1:-1].split(',')]
            next_parse_info = autos[auto_idx + 2].split()
            next_sentence_id = int(next_parse_info[0].split('.')[0].split('=')[1])
            next_apply_skimmer = next_parse_info[2]
            if 'True' in next_apply_skimmer:
                next_apply_skimmer = True
            else:
                next_apply_skimmer = False
        else:
            apply_skimmer = False

    else:
        # when move to next sentence
        if previous_sentence_id != sentence_id:
            if not apply_skimmer:
                temp_deps = []
                deps = ccgbank_deps[deps_idx]
                while deps != '\n':
                    temp_deps.append(deps)
                    deps_idx += 1
                    deps = ccgbank_deps[deps_idx]
                tag_info = ['<c>']
                tag_info = add_tag_info(auto, tag_info)
                # when there is no deps
                if temp_deps == []:
                    transformed.append('\n')
                else:
                    transformed.extend(temp_deps)
                    transformed.append(' '.join(tag_info) + '\n')
                    transformed.append('\n')
                deps_idx += 1
            # when apply skimmer
            else:
                temp_deps = []
                deps = ccgbank_deps[deps_idx]
                while deps != '\n':
                    temp_deps = add_deps_info(deps, scope[0], temp_deps)
                    deps_idx += 1
                    deps = ccgbank_deps[deps_idx]
                tag_info = ['<c>']
                tag_info = add_tag_info(auto, tag_info)
                if not next_apply_skimmer:
                    if temp_deps == []:
                        transformed.append('\n')
                    else:
                        transformed.extend(temp_deps)
                        transformed.append(' '.join(tag_info) + '\n')
                        transformed.append('\n')
                deps_idx += 1
        # when add another deps information to the same sentence
        else:
            deps = ccgbank_deps[deps_idx]
            while deps != '\n':
                temp_deps = add_deps_info(deps, scope[0], temp_deps)
                deps_idx += 1
                deps = ccgbank_deps[deps_idx]
            tag_info = add_tag_info(auto, tag_info)
            if sentence_id != next_sentence_id or not next_apply_skimmer:
                if temp_deps == []:
                    transformed.append('\n')
                else:
                    transformed.extend(temp_deps)
                    transformed.append(' '.join(tag_info) + '\n')
                    transformed.append('\n')
            deps_idx += 1
    auto_idx += 1


with open(path_to_ccgbank_deps, 'w') as f:
    f.writelines(transformed)
