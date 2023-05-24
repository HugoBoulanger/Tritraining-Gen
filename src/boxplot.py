import pandas as pd
import argparse
import numpy as np
import matplotlib.pyplot as plt
import re

from pathlib import Path

def get_f1_from_csv(path):
    dp = pd.read_csv(path)
    print(dp)
    print(dp.loc[dp['Unnamed: 0'] == 'micro avg', 'f1-score'])
    return float(dp.loc[dp['Unnamed: 0'] == 'micro avg', 'f1-score'])

def make_latex_table(ar, title, axes, std_ar=None, sign=False):

    tab = f"\n\n{title}\n\n"
    for w in axes[1]:
        tab += f" & {w}"
    tab += " \\\\ \n"
    for i in range(len(ar)):
        tab += f"{axes[0][i]}"
        for j in range(len(ar[i])):
            tab += f" & {'+' if ar[i][j] > 0 and sign else ''}{100 * ar[i][j]:.2f}"
            if std_ar is not None:
                tab += f"$\pm{100* std_ar[i][j]:.2f}$"
        tab += " \\\\ \n"
    return tab

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dir', type=str, required=True, help="Experiment directory where config.yml is located")
    args = parser.parse_args()


    lines = []
    l_c = sorted([p.parts[-1] for p in (Path(args.dir + '_conll') / 'logs').glob('*/')], key= lambda x: int(x[5:]))
    c_c = sorted(list(set([p.parts[-1] for p in (Path(args.dir + '_conll') / 'logs').glob('*/*')])), key= lambda x: int(x[6:]))
    print(l_c)
    print(c_c)

    l_i = sorted([p.parts[-1] for p in (Path(args.dir + '_i2b2') / 'logs').glob('*/')], key=lambda x: int(x[5:]))
    c_i = sorted(list(set([p.parts[-1] for p in (Path(args.dir + '_i2b2') / 'logs').glob('*/*')])),
                 key=lambda x: int(x[6:]))
    print(l_i)
    print(c_i)

    baseline_c = np.ma.masked_all((len(l_c), len(c_c)), dtype=float)
    tritrain_c = np.ma.masked_all((len(l_c), len(c_c)), dtype=float)

    baseline_i = np.ma.masked_all((len(l_i), len(c_i)), dtype=float)
    tritrain_i = np.ma.masked_all((len(l_i), len(c_i)), dtype=float)



    for il, s in enumerate(l_c):
        for ic, sp in enumerate(c_c):
            p = Path(args.dir + '_conll') / 'logs' / s / sp
            print(p)
            try:
                baseline_c_f1 = get_f1_from_csv(p / 'baseline/logs/slot_filling_report_test_EN.csv')
                baseline_c[il][ic] = baseline_c_f1
            except Exception as ex:
                print(ex)
            try:
                tri_c_f1 = get_f1_from_csv(p / 'results_results_tri_training.csv')
                tritrain_c[il][ic] = tri_c_f1
            except Exception as ex:
                print(ex)

    for il, s in enumerate(l_i):
        for ic, sp in enumerate(c_i):
            p = Path(args.dir + '_i2b2') / 'logs' / s / sp
            print(p)
            try:
                baseline_i_f1 = get_f1_from_csv(p / 'baseline/logs/slot_filling_report_test_EN.csv')
                baseline_i[il][ic] = baseline_i_f1
            except Exception as ex:
                print(ex)
            try:
                tri_i_f1 = get_f1_from_csv(p / 'results_results_tri_training.csv')
                tritrain_i[il][ic] = tri_i_f1
            except Exception as ex:
                print(ex)

    delta_c = np.ma.transpose(tritrain_c - baseline_c, axes=(1, 0))
    delta_i = np.ma.transpose(tritrain_i - baseline_i, axes=(1, 0))

    print(f'delta i :{delta_i}')

    delta = []

    for i in range(5):
        delta.append(delta_c[i])
        delta.append(delta_i[i])
    delta = np.ma.array(delta, dtype=float)

    print(delta)

    fig = plt.figure(figsize=(6, 4.5))
    bp1 = plt.boxplot(np.ma.transpose(delta_c*100, axes=(1,0)), sym='', positions=[1, 4, 7, 10, 13], patch_artist=True)
    print(bp1['boxes'])
    for patch in bp1['boxes']:
        patch.set_facecolor('lightblue')

    bp2 = plt.boxplot(np.ma.transpose(delta_i*100, axes=(1,0)), sym='', positions=[2, 5, 8, 11, 14], patch_artist=True)
    print(bp2['boxes'])
    for patch in bp2['boxes']:
        patch.set_facecolor('lightgreen')
    #plt.boxplot(np.ma.transpose(delta*100, axes=(1, 0)), sym='', positions=[1, 2, 4, 5, 7, 8, 10, 11, 13, 14])
    #plt.plot([1, 2, 3, 4, 5], avg_delta[0]*100, color='red')
    plt.plot([0.25, 15.75], [0, 0], color='black')
    plt.fill_between([20, 21], [1, 2], [3, 4], color='lightblue', label='CoNLL')
    plt.fill_between([20, 21], [1, 2], [3, 4], color='lightgreen', label='I2B2')
    plt.legend(loc='best')
    plt.xticks([1.5, 4.5, 7.5, 10.5, 13.5], [50, 100, 250, 500, 1000])
    plt.xlabel('Size of subset')
    plt.ylabel('F1')
    plt.title('F1 delta between tritrained ensemble and baseline')
    plt.xlim((0.5, 14.5))
    plt.savefig(Path(args.dir + '_conll') / 'delta_boxplot.svg')