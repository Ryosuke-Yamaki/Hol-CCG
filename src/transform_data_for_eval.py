import sys


def extract_tags(auto):
    tags = []
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
            tags.append('|'.join([word, pos, cat]))
    return ' '.join(tags)


def extract_deps(deps_list, start=None):
    extracted_deps = []
    while True:
        deps = deps_list.pop(0)
        if deps == '\n':
            if len(extracted_deps) > 0:
                if extracted_deps[-1].startswith('<c>'):
                    extracted_deps.pop(-1)
                if start is not None:
                    fix_deps_idx(extracted_deps, start)
            return extracted_deps
        else:
            extracted_deps.append(deps)


def fix_deps_idx(extracted_deps, start):
    for i in range(len(extracted_deps)):
        deps = extracted_deps[i].split()
        info = deps[0].split('_')
        info[1] = str(int(info[1]) + start)
        deps[0] = '_'.join(info)
        info = deps[3].split('_')
        info[1] = str(int(info[1]) + start)
        deps[3] = '_'.join(info)
        deps = ' '.join(deps) + '\n'
        extracted_deps[i] = deps


args = sys.argv

path_to_autos = args[1]
path_to_ccgbank_deps = args[2]
path_to_failure_deps = args[3]
path_to_error_log = args[4]
path_to_completed_ccgbank_deps = path_to_ccgbank_deps.replace(
    ".ccgbank_deps", ".completed_ccgbank_deps")

with open(path_to_autos, 'r') as f:
    autos = f.readlines()
with open(path_to_ccgbank_deps, 'r') as f:
    ccgbank_deps = f.readlines()
with open(path_to_failure_deps, 'r') as f:
    failure_deps = f.readlines()
with open(path_to_error_log, 'r') as f:
    error_log = f.readlines()

transformed_deps = ccgbank_deps[:3]
ccgbank_deps = ccgbank_deps[3:]
failure_deps = failure_deps[2:]
count = 0
for line in failure_deps:
    if line.startswith('<c>'):
        count += 1

auto_info = []
auto_content = []
for line in autos:
    if line.startswith('ID='):
        auto_info.append(line)
    else:
        auto_content.append(line)

extracted_tags = []
extracted_deps = []
for info, auto, log in zip(auto_info, auto_content, error_log):
    if len(info.split()) > 3:
        scope = info.split()[3].split('=')[1][1:-1].split(',')
        start = int(scope[0])
    else:
        start = None
    if 'parse successful' in log:
        extracted_deps.append(extract_deps(ccgbank_deps, start))
    else:
        extract_deps(ccgbank_deps, start)
        extracted_deps.append(extract_deps(failure_deps, start))
    extracted_tags.append(extract_tags(auto))

previous_sentence_id = 1
deps = extracted_deps[0]
tags = extracted_tags[0]
for idx in range(1, len(auto_info)):
    info = auto_info[idx]
    sentence_id = int(info.split()[0].split('=')[1].split('.')[0])
    if sentence_id != previous_sentence_id:
        transformed_deps.extend(deps)
        transformed_deps.append('<c> ' + tags + '\n')
        transformed_deps.append('\n')
        deps = extracted_deps[idx]
        tags = extracted_tags[idx]
    else:
        deps.extend(extracted_deps[idx])
        tags += ' ' + extracted_tags[idx]
    previous_sentence_id = sentence_id
transformed_deps.extend(deps)
transformed_deps.append('<c> ' + tags + '\n')
transformed_deps.append('\n')

with open(path_to_completed_ccgbank_deps, 'w') as f:
    f.writelines(transformed_deps)
