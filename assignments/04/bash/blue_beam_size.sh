timebleu=output_stats/output_time_bleu.txt
for i in {1..30}
do 

    echo 'beam size' $i
    echo "$i" >> "$timebleu"

    { time python translate_beam.py --beam-size $i --checkpoint-path assignments/03/baseline/checkpoints/checkpoint_best.pt --output assignments/04/translations/translation_beam$i.txt --data data/en-fr/prepared --dicts data/en-fr/prepared ; } 2> output.txt

    cat output.txt | grep -Po 'real\s*\K[0-9.ms]*' >> $timebleu

    bash scripts/postprocess.sh assignments/04/translations/translation_beam$i.txt assignments/04/translations/translation_beam$i.p.txt en

    cat assignments/04/translations/translation_beam$i.p.txt | sacrebleu data/en-fr/raw/test.en > output.txt
    
    cat output.txt | grep -Po '"score": \K[0-9.]*' >> $timebleu
    cat output.txt | grep -Po 'BP = \K[0-9.]*' >> $timebleu

done