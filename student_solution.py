import os
import math
import subprocess
from pathlib import Path
from sklearn.metrics import auc

TRAIN_FILE = "negsel_src/tests/english.train"

TEST_FILES = [
    "negsel_src/tests/hiligaynon.test",
    "negsel_src/tests/middle-english.test",
    "negsel_src/tests/plautdietsch.test",
    "negsel_src/tests/xhosa.test"
]

LABEL_FILE = "negsel_src/tests/labels.test"

NGA_FILE = "negsel_src/negsel.jar"

def anom_scores(fp):
    args = ["java", "-jar", NGA_FILE, "-n", "10", "-r", "4", "-self", TRAIN_FILE,]
    with open(fp, "r", encoding="utf-8") as f:
        res = subprocess.run(args=args, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,check=False)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    score_tup = tuple(float(value) for value in res.stdout.split())

    return score_tup

def final_anom(fp):
    print(f"Yay final_anom({fp})!")
    a_sum = 0.0
    for val in anom_scores(fp):
        if val == 0:
            val = 1e-6
        a_sum += math.log(val)
    return a_sum

def get_labels():
    label_tup = tuple()
    with open(LABEL_FILE, "r", encoding="utf-8") as f:
        for l in f:
            label_tup += (int(l.strip()),)

    return label_tup

def fp_tp_calc(fp, theta):
    print(f"Yay fp_tp_calc({fp}, {theta})!")
    s_vals = anom_scores(fp)
    l_vals = get_labels()

    # How to determine normal vs anomaly? Get labels
    # FPR = normal instances with score > theta / total normal instances
    # TPR = anomalous instances with score > theta / total anomalous instances
    n_scores: List[float] = []
    a_scores: List[float] = []
    for s, l in zip(s_vals, l_vals):
        if l == 0:
            n_scores.append(s)
        else:
            a_scores.append(s)

    n_gt = sum(s > theta for s in n_scores)
    a_gt = sum(s > theta for s in a_scores)

    fpr = n_gt / len(n_scores)
    tpr = a_gt / len(a_scores)
    # RETURN: (fpr, tpr)
    return (fpr, tpr)

def auc_calc(fp):
    print(f"Yay auc_calc({fp})!")
    # Int threshold vals range(0, 11)
    # call fptpcalc to get roc point in (FPR, TPR) form
    # Use the trapezoidal rule to calculate the area under the ROC curve. You may implement the trapezoidal rule manually or use sklearn.metrics.auc
    points = [fp_tp_calc(fp, theta) for theta in range(0, 11)]

    # ascending order
    points.sort(key=lambda point: point[0])

    # separate x, y
    fpr = [point[0] for point in points]
    tpr = [point[1] for point in points]

    return auc(fpr, tpr)

def main():
    print("Hello world!")
    for tf in TEST_FILES:
        tfp = Path(tf)
        fa_res = round(final_anom(tfp), 2)
        fpr, tpr = fp_tp_calc(tfp, 2)
        print(f"FPR: {fpr}, TPR: {tpr}")
        auc_ = final_anom(tfp)
        print(f"AUC: {auc_}")

if __name__ == "__main__":
    main()
