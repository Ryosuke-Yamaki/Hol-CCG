import subprocess
import os


MODEL_LIST = ["roberta-large_w_p_s_2022-04-07_02:11:15.pth"]

stat = []
path_to_hol_src = os.path.dirname(os.path.abspath(__file__))
path_to_java_candc = os.path.join(
    os.path.dirname(
        os.path.abspath(__file__)),
    '../../java-candc/')
for TARGET in ["dev", "test"]:
    for MODEL in MODEL_LIST:
        MODEL_TARGET = MODEL.replace('.pth', '_') + TARGET
        if TARGET == 'dev':
            GOLD = 'wsj00'
        elif TARGET == 'test':
            GOLD = "wsj23"
        path_to_stagged = os.path.join(
            path_to_java_candc,
            f'data/auto-stagged/{MODEL_TARGET}.stagged')
        path_to_deps = os.path.join(path_to_java_candc, f'data/output/{MODEL_TARGET}.out')
        path_to_log = os.path.join(path_to_java_candc, f'data/output/{MODEL_TARGET}.log')
        path_to_weight = os.path.join(path_to_java_candc, 'model/weights')
        path_to_params = os.path.join(path_to_java_candc, 'params')
        path_to_gold_stag = os.path.join(path_to_java_candc, f'data/gold/{GOLD}.stagged')
        path_to_gold_deps = os.path.join(path_to_java_candc, f'data/gold/{GOLD}.ccgbank_deps')
        path_to_eval = os.path.join(path_to_java_candc, f'data/eval/{MODEL_TARGET}.eval')
        path_to_eval = os.path.join(path_to_java_candc, 'data/eval/candc.stat')

        stag_command = f"python supertagging.py -m {MODEL} -t {TARGET}"
        parse_command = f"java -Xmx6g -classpath bin ParserBeam {path_to_stagged} {path_to_deps} {path_to_log} {path_to_weight} {path_to_params}"
        eval_command = f"python2 {os.path.join(path_to_java_candc,'scripts/evaluate_new')} {path_to_gold_stag} {path_to_gold_deps} {path_to_deps}|tee {path_to_eval}"

        os.chdir(path_to_hol_src)
        subprocess.run(stag_command, shell=True, text=True)
        os.chdir(path_to_java_candc)
        subprocess.run(parse_command, shell=True, text=True)
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
        stat.append(f"{MODEL} {TARGET} {lp} {lr} {lf} {lsent}\n")
for line in stat:
    print(line)

with open(path_to_eval, "w") as f:
    f.writelines(stat)
