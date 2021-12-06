#! /bin/sh

model=$1
dev_test=$2
stag_threshold=$3
label_threshold=$4
span_threshold=$5
min_freq=$6
result=${model/.pth/_}
result+=${dev_test}
result+="_"
result+=${stag_threshold}
result+="_"
result+=${label_threshold}
result+="_"
result+=${span_threshold}
result+="_"
result+=${min_freq}
if [ ${dev_test} = "dev" ];then
    gold="wsj00"
else
    gold="wsj23"
fi

python parser.py ${model} ${dev_test} ${stag_threshold} ${label_threshold} ${span_threshold} ${min_freq}| tee ~/span_parsing/AUTO/${result}.auto

cd ~/candc-1.00
src/scripts/ccg/convert_auto ~/span_parsing/AUTO/${result}.auto | src/scripts/ccg/convert_brackets > ~/span_parsing/PIPE/${result}.pipe
bin/generate -j ~/java-candc/grammar ~/java-candc/grammar/markedup.new.modified ~/span_parsing/PIPE/${result}.pipe > ~/span_parsing/CCGBANK_DEPS/${result}.ccgbank_deps

cd ~/Hol-CCG/src
python candc_completement.py ~/span_parsing/AUTO/${result}.auto ~/candc-1.00/errors.log ${dev_test}
python transform_data_for_eval.py ~/span_parsing/AUTO/${result}.auto ~/span_parsing/CCGBANK_DEPS/${result}.ccgbank_deps ~/span_parsing/FAILURE/${result}.out ~/candc-1.00/errors.log

cd ~/java-candc
python2 scripts/evaluate_new ~/span_parsing/GOLD/${gold}.stagged ~/span_parsing/GOLD/${gold}.ccgbank_deps ~/span_parsing/CCGBANK_DEPS/${result}.completed_ccgbank_deps|tee ~/span_parsing/EVAL/${result}.eval 