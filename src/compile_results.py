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
    l = sorted([p.parts[-1] for p in (Path(args.dir) / 'logs').glob('*/')], key= lambda x: int(x[5:]))
    c = sorted(list(set([p.parts[-1] for p in (Path(args.dir) / 'logs').glob('*/*')])), key= lambda x: int(x[6:]))
    print(l)
    print(c)

    baseline = np.ma.masked_all((len(l), len(c)), dtype=float)
    tritrain = np.ma.masked_all((len(l), len(c)), dtype=float)
    uniques = np.ma.masked_all((len(l)*3, len(c)), dtype=float)


    for il, s in enumerate(l):
        for ic, sp in enumerate(c):
            p = Path(args.dir) / 'logs' / s / sp
            print(p)
            try:
                baseline_f1 = get_f1_from_csv(p / 'baseline/logs/slot_filling_report_test_EN.csv')
                baseline[il][ic] = baseline_f1
            except Exception as ex:
                print(ex)
            try:
                tri_f1 = get_f1_from_csv(p / 'results_results_tri_training.csv')
                tritrain[il][ic] = tri_f1
            except Exception as ex:
                print(ex)
            try:
                un1 = get_f1_from_csv(p / 'tri-1/logs/slot_filling_report_test_EN.csv')
                un2 = get_f1_from_csv(p / 'tri-2/logs/slot_filling_report_test_EN.csv')
                un3 = get_f1_from_csv(p / 'tri-3/logs/slot_filling_report_test_EN.csv')
                uniques[3*il][ic] = un1
                uniques[3*il+1][ic] = un2
                uniques[3*il+2][ic] = un3
            except Exception as ex:
                print(ex)

    delta = tritrain - baseline

    delta_u = uniques - np.ma.array([baseline[i//3] for i in range(len(l)*3)], dtype=float)
    std_delta_u = [np.ma.std(delta_u, axis=0)]
    avg_delta_u = [np.ma.average(delta_u, axis=0)]
    avg_uniques = [np.ma.average(uniques, axis=0)]

    std_delta = [np.ma.std(delta, axis=0)]
    avg_delta = [np.ma.average(delta, axis=0)]
    print(avg_delta)
    avg_baseline = [np.ma.average(baseline, axis=0)]
    std_baseline = [np.ma.std(baseline, axis=0)]
    med_baseline = [np.ma.median(baseline, axis=0)]
    print(avg_baseline)

    avg_tritrain = [np.ma.average(tritrain, axis=0)]
    std_tritrain = [np.ma.std(tritrain, axis=0)]
    med_tritrain = [np.ma.median(tritrain, axis=0)]

    d_table = make_latex_table(delta, 'delta', [l, c])
    b_table = make_latex_table(baseline, 'baseline', [l, c])
    t_table = make_latex_table(tritrain, 'tritrain', [l, c])

    d_avg_table = make_latex_table(avg_delta, 'delta average', [["avg"], c], std_ar=std_delta, sign=True)
    b_avg_table = make_latex_table(avg_baseline, 'baseline average', [["avg"], c], std_ar=std_baseline)

    u_avg_table = make_latex_table(avg_uniques, 'tritrain unique models average', [['avg'], c])
    du_avg_table = make_latex_table(avg_delta_u, 'tritrain unique models delta with baseline average', [['avg'], c], std_ar=std_delta_u, sign=True)

    with open(Path(args.dir) / 'results.txt', 'w') as f:
        f.write(b_table + b_avg_table + t_table + d_table + d_avg_table + u_avg_table + du_avg_table)

    fig = plt.figure(figsize=(6, 4.5))

    plt.scatter(baseline*100, tritrain*100)
    plt.plot([-10, 150], [0, 0], color='black')
    plt.plot([-10, 150], [-10, 150], color='black')
    plt.xlabel('Baseline F1')
    plt.ylabel('Tritrained Ensemble F1')
    plt.ylim((0, 100))
    plt.xlim((0, 100))
    plt.savefig(Path(args.dir) / 'scatter.svg')

    fig = plt.figure(figsize=(6, 4.5))
    plt.boxplot(np.ma.transpose(delta*100, axes=(0, 1)), showmeans=True, meanline=True)
    #plt.plot([1, 2, 3, 4, 5], avg_delta[0]*100, color='red')
    plt.plot([0.25, 5.75], [0, 0], color='black')
    plt.xticks([1, 2, 3, 4, 5], [50, 100, 250, 500, 1000])
    plt.xlabel('Size of subset')
    plt.ylabel('F1')
    plt.title('F1 delta between tritrained ensemble and baseline')
    plt.xlim((0.5, 5.5))
    plt.savefig(Path(args.dir) / 'delta_boxplot.svg')

    fig = plt.figure(figsize=(6, 4.5))
    #plt.fill_between([1, 2, 3, 4, 5], [np.ma.max(cl) * 100 for cl in np.ma.transpose(tritrain)], [np.ma.min(cl)*100 for cl in np.ma.transpose(tritrain)],
    #                 color='red', label='Tritrained ensemble min and max', alpha=0.1)
    #plt.fill_between([1, 2, 3, 4, 5], [np.ma.max(cl)*100 for cl in np.ma.transpose(baseline)], [np.ma.min(cl)*100 for cl in np.ma.transpose(baseline)],
    #                 color='blue', label='Baseline min and max', alpha=0.1)

    plt.fill_between([1, 2, 3, 4, 5], [(avg_tritrain[0][i] + std_tritrain[0][i])*100 for i in range(5)],
                     [(avg_tritrain[0][i] - std_tritrain[0][i])*100 for i in range(5)],
                     color='red', label='Tritrained std', alpha=0.2)
    plt.fill_between([1, 2, 3, 4, 5], [(avg_baseline[0][i] + std_baseline[0][i])*100 for i in range(5)],
                     [(avg_baseline[0][i] - std_baseline[0][i])*100 for i in range(5)],
                     color='blue', label='Baseline std', alpha=0.2)

    plt.xticks([1, 2, 3, 4, 5], [50, 100, 250, 500, 1000])
    plt.plot([1, 2, 3, 4, 5], avg_tritrain[0] * 100, color='red', label='Tritrained ensemble avg')
    plt.plot([1, 2, 3, 4, 5], avg_baseline[0] * 100, color='blue', label='Baseline avg')
    #plt.plot([1, 2, 3, 4, 5], med_tritrain[0] * 100, '--', color='red', label='Tritrained ensemble med')
    #plt.plot([1, 2, 3, 4, 5], med_baseline[0] * 100, '--', color='blue', label='Baseline med')
    plt.xlabel('Size of subset')
    plt.ylabel('F1')
    plt.legend(loc='best')
    plt.ylim((np.ma.min(baseline)*100 - 2 , 100))
    plt.savefig(Path(args.dir) / 'plot.svg')

    fig = plt.figure(figsize=(6, 4.5))
    plt.plot([1, 2, 3, 4, 5], avg_baseline[0] * 100, color='blue', label='Baseline avg')
    plt.plot([1, 2, 3, 4, 5], avg_tritrain[0] * 100, color='red', label='Tritraining avg')
    plt.xticks([1, 2, 3, 4, 5], [50, 100, 250, 500, 1000])
    plt.ylim((0, 100))
    plt.xlabel('Size of subset')
    plt.ylabel('F1')
    plt.legend(loc='best')
    plt.xlim((0.5, 5.5))
    plt.savefig(Path(args.dir) / 'avgplot.svg')