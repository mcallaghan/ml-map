MODEL="prajjwal1/bert-tiny"
MODEL="distilroberta-base"
for VAR in "INCLUDE" "4 -" "8 -"
#for VAR in "4 -" "8 -"
do
    python mlmap/pipeline_train.py -m "$MODEL" -y "$VAR" -t 25
    python mlmap/pipeline_predict.py -m "$MODEL" -y "$VAR"

done
