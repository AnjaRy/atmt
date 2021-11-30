import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("test_output_time_bleu.txt", 'r', encoding='utf-8') as infile:
    counter = 0
    rows = []
    row = {'beam_size': 0, 'bleu': 0, 'time': 0, 'bp': 0}
    for line in infile:
        line = line.strip()
        
        # beam size
        if counter == 0:
            row['beam_size'] = line
            counter += 1
        
        # time
        elif counter == 1:
            row['time'] = line
            counter += 1
        
        # bleu
        elif counter == 2:
            row['bleu'] = line
            counter += 1
            
            
        elif counter == 3:
            row['bp'] = line    
            rows.append(row)
            counter = 0
            row = {'beam_size': 0, 'bleu': 0, 'time': 0, 'bp': 0}

df = pd.DataFrame.from_dict(rows, orient='columns')
print(df)

sns.lineplot(data=df, x="beam_size", y="bleu")

plt.show()