import os, shutil
from pathlib import Path
import glob

seeds = [1, 2, 3, 4, 5]
split = [50]

origin = Path("../20220318_conll")

#dest = "../20220507_conll"
#dest = "../20220614_conll"
dest = "../20220802_toplines/conll"

#dlist = ['_small']
dlist = ['_small']
#dlist = ['_complete', '_follow', "_combine", "_mention", "_context"]
#dlist = ["_context", '_completion', '_mention', '_entailment']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")



origin = Path("../20220318_i2b2")

#dest = "../20220507_i2b2"

dest = "../20220802_toplines/i2b2"

#dlist = ['_follow']
dlist = ['_small']
#dlist = ['_complete', '_follow', "_combine", "_mention", "_context"]
#dlist = ["_context", '_completion', '_mention', '_entailment']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")


"""

origin = Path("../20220318_i2b2")

dest = "../202205"

dlist = ["10_i2b2_semi_supervised_natural", '12_i2b2_combined']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")

origin = Path("../20220318_conll")

dest = "../202205"

dlist = ["10_conll_semi_supervised_natural", '12_conll_combined']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")



split = [100]

origin = Path("../20220318_conll")

dest = "../20220515_conll"

dlist = ["_context", '_completion', '_mention', '_entailment']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")



origin = Path("../20220318_i2b2")

dest = "../20220515_i2b2"

dlist = ["_context", '_completion', '_mention', '_entailment']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")



origin = Path("../20220318_i2b2")

dest = "../202205"

dlist = ["14_i2b2_semisuper", '14_i2b2_combined']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")

origin = Path("../20220318_conll")

dest = "../202205"

dlist = ["14_conll_semisuper", '14_conll_combined']

for s in seeds:
    for si in split:
        for e in dlist:
            if not os.path.exists(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs"):
                os.mkdir(Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs")
            shutil.copyfile(origin / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv", Path(dest + e) / f"logs/seed={s}/split={si}/baseline/logs/slot_filling_report_test_EN.csv")

"""