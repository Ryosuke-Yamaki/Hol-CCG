import subprocess
import os
import pathlib
from utils import DIR
from os.path import join

MODEL_LIST = list(pathlib.Path(join(DIR, 'data/model')).glob('*.pth'))

stat = []
for MODEL in MODEL_LIST:
    print(MODEL)
    for TYPE in ["dev", "test"]:
        TARGET = str(MODEL.name).replace(".pth", "_") + TYPE
        if TYPE == "dev":
            gold = "wsj00"
        elif TYPE == "test":
            gold = "wsj23"
        PATH_TO_AUTO = join(DIR, f'span_parsing/AUTO/{TARGET}.auto')
        PATH_TO_PIPE = join(DIR, f'span_parsing/PIPE/{TARGET}.pipe')
        PATH_TO_DEPS = join(DIR, f'span_parsing/CCGBANK_DEPS/{TARGET}.ccgbank_deps')
        PATH_TO_REPLACED_DEPS = join(
            DIR, f'span_parsing/CCGBANK_DEPS/{TARGET}.replaced_ccgbank_deps')
        PATH_TO_GOLD_STAGGED = join(DIR, f'span_parsing/GOLD/{gold}.stagged')
        PATH_TO_GOLD_DEPS = join(DIR, f'span_parsing/GOLD/{gold}.ccgbank_deps')
        PATH_TO_EVAL = join(DIR, f'span_parsing/EVAL/{TARGET}.eval')
        span_parse_command = f"python span_parser.py -m {MODEL} -t {TYPE}| tee {PATH_TO_AUTO}"
        convert_command = f"src/scripts/ccg/convert_auto {PATH_TO_AUTO} | src/scripts/ccg/convert_brackets > {PATH_TO_PIPE}"
        generate_command = f"bin/generate -j {join(DIR,'../java-candc/grammar')} {join(DIR,'../java-candc/grammar/markedup.new.modified')} {PATH_TO_PIPE} > {PATH_TO_DEPS}"
        replacement_command = f"python replace_conversion_failure.py -m {MODEL} -t {TYPE}|tee temp.log"
        eval_command = f"python2 scripts/evaluate_new {PATH_TO_GOLD_STAGGED} {PATH_TO_GOLD_DEPS} {PATH_TO_REPLACED_DEPS}|tee {PATH_TO_EVAL}"
        os.chdir(join(DIR, 'src'))
        subprocess.run(span_parse_command, shell=True, text=True)
        os.chdir(join(DIR + "../candc-1.00"))
        subprocess.run(convert_command, shell=True, text=True)
        subprocess.run(generate_command, shell=True, text=True)
        os.chdir(join(DIR, 'src'))
        subprocess.run(replacement_command, shell=True, text=True)
        # os.chdir(join(DIR, '../java-candc'))
        # subprocess.run(eval_command, shell=True, text=True)
        # with open(PATH_TO_EVAL, 'r') as f:
        #     eval = f.readlines()
        # for line in eval:
        #     if line.startswith('lp:'):
        #         lp = float(line.split()[1].replace('%', ''))
        #     elif line.startswith('lr:'):
        #         lr = float(line.split()[1].replace('%', ''))
        #     elif line.startswith('lf:'):
        #         lf = float(line.split()[1].replace('%', ''))
        #     elif line.startswith('lsent:'):
        #         lsent = float(line.split()[1].replace('%', ''))
        # stat.append("{} {} {} {} {} {}\n".format(MODEL, TYPE, lp, lr, lf, lsent))
# with open(f"{DIR}span_parsing/EVAL/span.stat", "w") as f:
#     f.writelines(stat)
