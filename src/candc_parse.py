import subprocess
import os
from utils import Condition_Setter

condition = Condition_Setter(set_embedding_type=False)

MODEL_LIST = ["roberta-large_phrase_span_2022-01-26_11:08:46.pth",
              "roberta-large_phrase_span_2022-01-26_21:18:30.pth",
              "roberta-large_phrase_span_2022-01-26_04:56:33.pth",
              "roberta-large_phrase_span_2022-01-26_09:02:21.pth",
              "roberta-large_phrase_span_2022-01-26_06:57:59.pth",
              "roberta-large_phrase_span_2022-01-26_17:14:21.pth",
              "roberta-large_phrase_span_2022-01-26_15:14:18.pth",
              "roberta-large_phrase_span_2022-01-26_13:14:00.pth",
              "roberta-large_phrase_span_2022-01-26_23:21:23.pth",
              "roberta-large_phrase_span_2022-01-26_19:14:30.pth",
              "roberta-large_phrase_2022-01-28_05:39:36.pth",
              "roberta-large_phrase_2022-01-28_13:59:11.pth",
              "roberta-large_phrase_2022-01-31_04:57:52.pth",
              "roberta-large_phrase_2022-01-28_07:47:07.pth",
              "roberta-large_phrase_2022-01-28_18:12:40.pth",
              "roberta-large_phrase_2022-01-28_11:55:39.pth",
              "roberta-large_phrase_2022-01-28_22:22:56.pth",
              "roberta-large_phrase_2022-01-29_00:29:51.pth",
              "roberta-large_phrase_2022-01-28_20:19:51.pth",
              "roberta-large_phrase_2022-01-28_16:05:16.pth",
              "roberta-large_span_2022-01-28_18:04:45.pth",
              "roberta-large_span_2022-01-28_20:09:54.pth",
              "roberta-large_span_2022-01-28_22:15:29.pth",
              "roberta-large_span_2022-01-28_07:43:15.pth",
              "roberta-large_span_2022-01-28_15:58:06.pth",
              "roberta-large_span_2022-01-28_13:53:18.pth",
              "roberta-large_span_2022-01-29_00:17:53.pth",
              "roberta-large_span_2022-01-28_11:49:43.pth",
              "roberta-large_span_2022-01-28_05:39:45.pth",
              "roberta-large_span_2022-01-28_09:48:13.pth",
              "roberta-large_2022-01-29_18:04:38.pth",
              "roberta-large_2022-01-29_16:05:35.pth",
              "roberta-large_2022-01-29_22:03:58.pth",
              "roberta-large_2022-01-30_06:10:43.pth",
              "roberta-large_2022-01-30_04:10:59.pth",
              "roberta-large_2022-01-30_02:11:33.pth",
              "roberta-large_2022-01-29_20:04:18.pth",
              "roberta-large_2022-01-30_08:12:44.pth",
              "roberta-large_2022-01-29_14:06:06.pth",
              "roberta-large_2022-01-30_00:06:48.pth",
              "roberta-large_phrase_span_2022-01-26_20:16:06.pth",
              "roberta-large_phrase_span_2022-01-26_08:45:19.pth",
              "roberta-large_phrase_span_2022-01-26_12:39:01.pth",
              "roberta-large_phrase_span_2022-01-30_13:43:39.pth",
              "roberta-large_phrase_span_2022-01-26_16:28:05.pth",
              "roberta-large_phrase_span_2022-01-30_23:17:40.pth",
              "roberta-large_phrase_span_2022-01-26_06:48:23.pth",
              "roberta-large_phrase_span_2022-01-30_15:40:51.pth",
              "roberta-large_phrase_span_2022-01-26_22:09:11.pth",
              "roberta-large_phrase_span_2022-01-30_19:30:23.pth",
              "roberta-large_phrase_span_2022-01-30_11:36:05.pth",
              "roberta-large_phrase_span_2022-01-30_11:36:28.pth"]

stat = []
THRESHOLD = 0.1
for DEV_TEST in ["dev", "test"]:
    for MODEL in MODEL_LIST:
        print(MODEL)
        if DEV_TEST == 'dev':
            stagged_file = condition.PATH_TO_DIR + "java-candc/data/auto-stagged/" + \
                MODEL.replace('.pth', '_') + DEV_TEST + '.stagged'
            gold = 'wsj00'
        elif DEV_TEST == 'test':
            stagged_file = condition.PATH_TO_DIR + "java-candc/data/auto-stagged/" + \
                MODEL.replace('.pth', '_') + DEV_TEST + '.stagged'
            gold = "wsj23"

        os.chdir(condition.PATH_TO_DIR + 'Hol-CCG/src')
        stag_command = "python supertagging.py {} {} {}".format(MODEL, DEV_TEST, THRESHOLD)
        subprocess.run(stag_command, shell=True, text=True)
        os.chdir(condition.PATH_TO_DIR + 'java-candc')
        CANDC_TARGET = MODEL.replace('.pth', '_') + DEV_TEST
        parse_command = "java -Xmx6g -classpath bin ParserBeam {} data/output/{}.out data/output/{}.log model/weights params".format(
            stagged_file, CANDC_TARGET, CANDC_TARGET)
        eval_command = "python2 scripts/evaluate_new data/gold/{}.stagged data/gold/{}.ccgbank_deps data/output/{}.out|tee data/eval/{}.eval".format(
            gold, gold, CANDC_TARGET, CANDC_TARGET)
        subprocess.run(parse_command, shell=True, text=True)
        subprocess.run(eval_command, shell=True, text=True)
        with open("data/eval/{}.eval".format(CANDC_TARGET), 'r') as f:
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
        stat.append(f"{MODEL} {DEV_TEST} {lp} {lr} {lf} {lsent}\n")
for line in stat:
    print(line)

with open("data/eval/candc.stat", "w") as f:
    f.writelines(stat)
