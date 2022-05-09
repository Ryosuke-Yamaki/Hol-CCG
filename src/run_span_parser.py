from dataclasses import replace
import torch
import time
import subprocess
import os

model_list = ["roberta-large_w_p_s_2022-04-06_10:46:58.pth"]

stat = []

for target in ['dev', 'test']:
    if target == "dev":
        gold = "wsj00"
    elif target == "test":
        gold = "wsj23"
    for model in model_list:
        model_target = model.replace(".pth", "_" + target)
        path_to_auto = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'../span_parsing/AUTO/{model_target}.auto')
        path_to_pipe = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'../span_parsing/PIPE/{model_target}.pipe')
        path_to_deps = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'../span_parsing/CCGBANK_DEPS/{model_target}.ccgbank_deps')
        path_to_replaced_deps = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'../span_parsing/CCGBANK_DEPS/{model_target}.replaced_ccgbank_deps')
        path_to_gold_stag = os.path.dirname(os.path.abspath(
            __file__)), f'../span_parsing/GOLD/{gold}.stagged'
        path_to_gold_deps = os.path.dirname(os.path.abspath(
            __file__)), f'../span_parsing/GOLD/{gold}.ccgbank_deps'
        path_to_eval = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'../span_parsing/EVAL/{model_target}.eval')
        path_to_stat = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            f'../span_parsing/EVAL/span_parsing.stat')
        path_to_candc = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../candc-1.00/')
        path_to_java_candc = os.path.join(
            os.path.dirname(
                os.path.abspath(__file__)),
            '../../java-candc/')

        span_parse_command = f"python span_parser.py -m {model} -t {target}| tee {path_to_auto}"
        convert_command = f"{os.path.join(path_to_candc,'src/scripts/ccg/convert_auto')} {path_to_auto} | {os.path.join(path_to_candc,'src/scripts/ccg/convert_brackets')} > {path_to_pipe}"
        generate_command = f"{os.path.join(path_to_candc,'bin/generate')} -j {os.path.join(path_to_java_candc,'grammar')} {os.path.join(path_to_java_candc,'grammar/markedup.new.modified')} {path_to_pipe} > {path_to_deps}"
        replacement_command = f"python replace_conversion_failure.py -m {model} -t {target}|tee temp.log"
        eval_command = f"python2 {os.path.join(path_to_java_candc,'scripts/evaluate_new')} {path_to_gold_stag} {path_to_gold_deps} {path_to_replaced_deps}|tee {path_to_eval}"
        subprocess.run(span_parse_command, shell=True, text=True)
        subprocess.run(convert_command, shell=True, text=True)
        subprocess.run(generate_command, shell=True, text=True)
        subprocess.run(replacement_command, shell=True, text=True)
        print(replacement_command)
        subprocess.run(eval_command, shell=True, text=True)
        with open(path_to_eval, 'r') as f:
            eval = f.readlines()
        for line in eval:
            if line.startswith('lp:'):
                lp = float(line.split()[1].replace('%', ''))
            elif line.startswith('lr:'):
                lr = float(line.split()[1].replace('%', ''))
            elif line.startswith('lf:'):
                lf = float(line.split()[1].replace('%', ''))
            elif line.startswith('lsent:'):
                lsent = float(line.split()[1].replace('%', ''))
        stat.append(f"{model} {target} {lp} {lr} {lf} {lsent}\n")

for line in stat:
    print(line)

with open(path_to_stat, "w") as f:
    f.writelines(stat)
