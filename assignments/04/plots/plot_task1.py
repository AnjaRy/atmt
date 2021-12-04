import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

with open("assignments/04/bash/output_stats/output_time_bleu.txt", 'r', encoding='utf-8') as infile:
    counter = 0
    rows = []
    row = {'beam_size': 0, 'bleu': 0, 'time': 0, 'bp': 0}
    for line in infile:
        line = line.strip()

        # beam size
        if counter == 0:
            row['beam_size'] = int(line)
            counter += 1

        # time
        elif counter == 1:

            split = line.split('m')
            sec = float(split[0])*60 + float(split[1].rstrip('s'))
            row['time'] = sec
            counter += 1

        # bleu
        elif counter == 2:
            row['bleu'] = float(line)
            counter += 1

        elif counter == 3:
            row['bp'] = float(line)
            rows.append(row)
            counter = 0
            row = {'beam_size': 0, 'bleu': 0, 'time': 0, 'bp': 0}

df = pd.DataFrame.from_dict(rows, orient='columns')
print(df)

fig, ax1 = plt.subplots()
ax2 = ax1.twinx()


plot1 = sns.lineplot(data=df, x="beam_size", y="bleu", marker='o', ax=ax1, color='b')
plot2 = sns.lineplot(data=df, x="beam_size", y="bp", ax=ax2, marker='o', color='r')
ax1.set(xlabel='Beam Size', ylabel='BLEU Score')
ax2.set(ylabel="Brevity Penality")
ax2.yaxis.label.set_color('red')
ax1.yaxis.label.set_color('blue')

plt.savefig('assignments/04/plots/bleu_bp_beam_size.png')

plt.figure() 
karl = sns.lineplot(data=df, x="beam_size", y="time", marker='o')
karl.set(xlabel='Beam Size', ylabel='Time in s')

plt.savefig('assignments/04/plots/time_beam_size.png')


