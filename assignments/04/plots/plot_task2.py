import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("assignments/04/bash/output_stats/output_norm_bleu.txt", 'r', encoding='utf-8') as infile:
    counter = 0
    rows = []
    row = {'alpha': 0, 'bleu': 0, 'time': 0, 'bp': 0}
    for line in infile:
        line = line.strip()

        # beam size
        if counter == 0:
            row['alpha'] = float(line)
            counter += 1

        elif counter == 1:
            counter += 1
        # bleu
        elif counter == 2:
            row['bleu'] = float(line)
            counter += 1

        elif counter == 3:
            rows.append(row)
            counter = 0
            row = {'alpha': 0, 'bleu': 0, 'time': 0, 'bp': 0}

df = pd.DataFrame.from_dict(rows, orient='columns')
print(df)

plt.figure() 
karl = sns.lineplot(data=df, x="alpha", y="bleu", marker='o')
karl.set(xlabel='Alpha', ylabel='Bleu Score')

plt.savefig('assignments/04/plots/alpha_bleu.png')


