import argparse
import os
import subprocess


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


parser = argparse.ArgumentParser()
parser.add_argument('-m', '--model', type=str, required=True)
parser.add_argument('-t', '--target', type=str, choices=['dev', 'test'], required=True)
parser.add_argument('--stag_threshold', type=float, default=0.1)

# args = parser.parse_args()
# model = args.model
# target = args.target
# stag_threshold = args.stag_threshold

model = 'roberta-large_w_p_s_2022-04-07_02:11:15.pth'
target = 'dev'

model_target = model.replace(".pth", "_" + target)
path_to_auto = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'../span_parsing/AUTO/{model_target}.auto')
path_to_error_log = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    '../../candc-1.00/errors.log')
path_to_failed_sentence = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"../span_parsing/FAILURE/{model_target}.raw")
path_to_auto_pos = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"../span_parsing/FAILURE/{model_target}.auto_pos")
path_to_stagged = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), f"../span_parsing/FAILURE/{model_target}.stagged")
path_to_ccgbank_deps = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f"../span_parsing/CCGBANK_DEPS/{model_target}.ccgbank_deps")
path_to_alt_deps = os.path.join(os.path.dirname(
    __file__), f"../span_parsing/FAILURE/{model_target}.out")
path_to_replaced_deps = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'../span_parsing/CCGBANK_DEPS/{model_target}.replaced_ccgbank_deps')
path_to_log = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    f'../span_parsing/FAILURE/{model_target}.log')
if target == 'dev':
    path_to_sentence = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        "../span_parsing/GOLD/wsj00.raw")
elif target == 'test':
    path_to_sentence = os.path.join(
        os.path.dirname(
            os.path.abspath(__file__)),
        "../span_parsing/GOLD/wsj23.raw")

path_to_candc = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../candc-1.00/')
path_to_java_candc = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../../java-candc/')

with open(path_to_auto, 'r') as f:
    autos = f.readlines()
with open(path_to_error_log, 'r') as f:
    error_log = f.readlines()
with open(path_to_sentence, 'r') as f:
    sentence_list = f.readlines()[3:]
count = 0
failed_sentence_list = []
failed_sentence_id = []
error_log_idx = 0
for line in autos:
    if line.startswith('ID='):
        sentence_info = line.split()
        sentence_id = sentence_info[0].split('=')[1]
        apply_skimmer = sentence_info[2].split('=')[1]
        if apply_skimmer == 'True':
            apply_skimmer = True
            sentence_id = int(sentence_id.split('.')[0])
            scope = sentence_info[3].split('=')[1]
            scope = scope[1:-1].split(',')
            scope = (int(scope[0]), int(scope[1]))
        else:
            sentence_id = int(sentence_id)
            apply_skimmer = False
    else:
        message = error_log[error_log_idx]
        error_log_idx += 1
        sentence = sentence_list[sentence_id - 1]
        if 'parse successful' not in message:
            count += 1
            if apply_skimmer:
                sentence = sentence.split()
                scoped_sentence = ' '.join(sentence[scope[0]:scope[1]])
                if not scoped_sentence.endswith('\n'):
                    scoped_sentence += '\n'
                failed_sentence_list.append(scoped_sentence)
            else:
                failed_sentence_list.append(sentence)
                failed_sentence_id.append(sentence_id)

with open(path_to_failed_sentence, 'w') as f:
    f.writelines(failed_sentence_list)

pos_command = f"{os.path.join(path_to_candc,'bin/pos')} --model {os.path.join(path_to_candc,'models/pos')} --input {path_to_failed_sentence} --output {path_to_auto_pos}"
stag_command = f"python supertagging.py -m {model} -t {target} --failure"
candc_command = f"java -Xmx6g -classpath bin ParserBeam {path_to_stagged} {path_to_alt_deps} {path_to_log} model/weights params"
subprocess.run(pos_command, shell=True, text=True)
subprocess.run(stag_command, shell=True, text=True)
os.chdir(path_to_java_candc)
subprocess.run(candc_command, shell=True, text=True)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

with open(path_to_auto, 'r') as f:
    autos = f.readlines()
with open(path_to_ccgbank_deps, 'r') as f:
    ccgbank_deps = f.readlines()
with open(path_to_alt_deps, 'r') as f:
    alt_deps = f.readlines()
with open(path_to_error_log, 'r') as f:
    error_log = f.readlines()

replaced_deps = ccgbank_deps[:3]
ccgbank_deps = ccgbank_deps[3:]
alt_deps = alt_deps[2:]
count = 0
for line in alt_deps:
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
        extracted_deps.append(extract_deps(alt_deps, start))
    extracted_tags.append(extract_tags(auto))

previous_sentence_id = 1
deps = extracted_deps[0]
tags = extracted_tags[0]
for idx in range(1, len(auto_info)):
    info = auto_info[idx]
    sentence_id = int(info.split()[0].split('=')[1].split('.')[0])
    if sentence_id != previous_sentence_id:
        replaced_deps.extend(deps)
        replaced_deps.append('<c> ' + tags + '\n')
        replaced_deps.append('\n')
        deps = extracted_deps[idx]
        tags = extracted_tags[idx]
    else:
        deps.extend(extracted_deps[idx])
        tags += ' ' + extracted_tags[idx]
    previous_sentence_id = sentence_id
replaced_deps.extend(deps)
replaced_deps.append('<c> ' + tags + '\n')
replaced_deps.append('\n')

with open(path_to_replaced_deps, 'w') as f:
    f.writelines(replaced_deps)
