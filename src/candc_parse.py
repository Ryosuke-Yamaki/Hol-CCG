import subprocess
import os
from utils import Condition_Setter

condition = Condition_Setter(set_embedding_type=False)

MODEL_LIST = ["roberta-large_2022-01-09_14_52_28.pth",
              "roberta-large_2022-01-09_12_17_27.pth",
              "roberta-large_2022-01-09_09_38_20.pth",
              "roberta-large_2022-01-09_07_06_03.pth",
              "roberta-large_2022-01-09_04_33_40.pth",
              "roberta-large_2022-01-09_01_53_57.pth",
              "roberta-large_2022-01-08_20_38_20.pth",
              "roberta-large_2022-01-08_23_15_36.pth",
              "roberta-large_2022-01-08_15_15_41.pth",
              "roberta-large_2022-01-08_17_55_37.pth",
              "roberta-large_phrase_2022-01-09_14_47_30.pth",
              "roberta-large_phrase_2022-01-09_12_16_41.pth",
              "roberta-large_phrase_2022-01-09_09_41_24.pth",
              "roberta-large_phrase_2022-01-09_07_03_50.pth",
              "roberta-large_phrase_2022-01-09_04_31_51.pth",
              "roberta-large_phrase_2022-01-09_01_48_54.pth",
              "roberta-large_phrase_2022-01-08_23_10_27.pth",
              "roberta-large_phrase_2022-01-08_20_34_09.pth",
              "roberta-large_phrase_2022-01-08_17_56_22.pth",
              "roberta-large_phrase_2022-01-08_15_15_45.pth",
              "roberta-large_span_2022-01-09_13_43_24.pth",
              "roberta-large_span_2022-01-09_11_15_10.pth",
              "roberta-large_span_2022-01-09_08_49_05.pth",
              "roberta-large_span_2022-01-09_06_23_49.pth",
              "roberta-large_span_2022-01-09_03_57_35.pth",
              "roberta-large_span_2022-01-09_01_24_54.pth",
              "roberta-large_span_2022-01-08_22_52_48.pth",
              "roberta-large_span_2022-01-08_20_23_07.pth",
              "roberta-large_span_2022-01-08_17_51_04.pth",
              "roberta-large_span_2022-01-08_15_15_48.pth",
              "roberta-large_phrase_span_2022-01-09_14_41_36.pth",
              "roberta-large_phrase_span_2022-01-09_12_26_53.pth",
              "roberta-large_phrase_span_2022-01-09_12_12_24.pth",
              "roberta-large_phrase_span_2022-01-09_07_04_32.pth",
              "roberta-large_phrase_span_2022-01-09_04_28_30.pth",
              "roberta-large_phrase_span_2022-01-09_01_52_35.pth",
              "roberta-large_phrase_span_2022-01-08_23_15_40.pth",
              "roberta-large_phrase_span_2022-01-08_20_34_15.pth",
              "roberta-large_phrase_span_2022-01-08_17_57_10.pth",
              "roberta-large_phrase_span_2022-01-08_15_15_54.pth"
              ]

stat = []
for MODEL in MODEL_LIST:
    print(MODEL)
    for DEV_TEST in ["dev"]:
        THRESHOLD = 0.1
        if DEV_TEST == 'dev':
            stagged_file = condition.PATH_TO_DIR + "java-candc/data/auto-stagged/" + \
                MODEL.replace('.pth', '_') + DEV_TEST + '.stagged'
            gold = 'wsj00'
        elif DEV_TEST == 'test':
            stagged_file = condition.PATH_TO_DIR + "java-candc/data/auto-stagged/" + \
                MODEL.replace('.pth', '_') + DEV_TEST + '.stagged'
            gold = "wsj23"

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
        stat.append("{} {} {} {} {} {}\n".format(MODEL, DEV_TEST, lp, lr, lf, lsent))
with open("data/eval/candc.stat", "w") as f:
    f.writelines(stat)
