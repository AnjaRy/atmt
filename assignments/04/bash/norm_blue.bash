alphableu=assignments/04/bash/output_stats/output_norm_bleu.txt
beamsize=13
for i in 0.0 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0
do 

    echo 'alpha' $i
    echo "$i" >> "$alphableu"

    { time python translate_beam.py --beam-size $beamsize --alpha $i --checkpoint-path assignments/03/baseline/checkpoints/checkpoint_best.pt --output assignments/04/translations/translation_alpha$i.txt --data data/en-fr/prepared --dicts data/en-fr/prepared ; } 2> output.txt

    cat output.txt | grep -Po 'real\s*\K[0-9.ms]*' >> $alphableu

    bash scripts/postprocess.sh assignments/04/translations/translation_alpha$i.txt assignments/04/translations/translation_alpha$i.p.txt en

    cat assignments/04/translations/translation_alpha$i.p.txt | sacrebleu data/en-fr/raw/test.en > output.txt
    
    cat output.txt | grep -Po '"score": \K[0-9.]*' >> $alphableu
    cat output.txt | grep -Po 'BP = \K[0-9.]*' >> $alphableu

done