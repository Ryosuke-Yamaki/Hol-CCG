import os
import subprocess
import sys
from utils import Condition_Setter

condition = Condition_Setter(set_embedding_type=False)

args = sys.argv
path_to_autos = args[1]
path_to_error_log = args[2]
dev_test = args[3]

if dev_test == 'dev':
    path_to_sentence = condition.PATH_TO_DIR + "span_parsing/GOLD/wsj00.raw"
elif dev_test == 'test':
    path_to_sentence = condition.PATH_TO_DIR + "span_parsing/GOLD/wsj23.raw"

result = path_to_autos.split('/')[-1].replace(".auto", "")
model = '_'.join(result.split('_')[:-5])
stag_threthold = result.split('_')[-4]
path_to_failed_sentence = condition.PATH_TO_DIR + "span_parsing/FAILURE/{}.raw".format(result)
path_to_auto_pos = condition.PATH_TO_DIR + "span_parsing/FAILURE/{}.auto_pos".format(result)

with open(path_to_autos, 'r') as f:
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

os.chdir(condition.PATH_TO_DIR + "candc-1.00")
pos_command = """bin/pos --model models/pos --input {} --output {}""".format(
    path_to_failed_sentence, path_to_auto_pos)
subprocess.run(pos_command, shell=True, text=True)

os.chdir(condition.PATH_TO_DIR + "Hol-CCG/src")
super_command = """python stag.py {}.pth failure {} None {}""".format(
    model, str(stag_threthold), result)
subprocess.run(super_command, shell=True, text=True)

os.chdir(condition.PATH_TO_DIR + "java-candc")
candc_command = """java -Xmx6g -classpath bin ParserBeam ~/span_parsing/FAILURE/{}.stagged  ~/span_parsing/FAILURE/{}.out ~/span_parsing/FAILURE/{}.log model/weights params""".format(
    result,
    result,
    result)
subprocess.run(candc_command, shell=True, text=True)
