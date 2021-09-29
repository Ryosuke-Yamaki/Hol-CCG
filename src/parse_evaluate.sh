#! /bin/sh

model=$1
dev_test=$2
candc_target=${model/.pth/_}
candc_target+=${dev_test}
if [ ${dev_test} = "dev" ];then
    gold="wsj00"
else
    gold="wsj23"
fi
python pos_super.py ${model} ${dev_test}
cd ~/java-candc
java -Xmx6g -classpath bin ParserBeam data/auto-stagged/${candc_target}.stagged  data/output/${candc_target}.out data/output/${candc_target}.log model/weights
python2 scripts/evaluate_new data/gold/${gold}.stagged data/gold/${gold}.ccgbank_deps data/output/${candc_target}.out|tee data/eval/${candc_target}.eval