from __future__ import annotations

import os
import math
import subprocess
from functools import lru_cache
from typing import List, Tuple



def _negsel_root() -> str:
    candidates = [
        "/usercode/negsel_src",
        "/autograder/source/negsel_src",
        os.path.join(os.getcwd(), "negsel_src"),
        os.path.join(os.path.dirname(__file__), "negsel_src"),
    ]
    for candidate in candidates:
        if os.path.exists(os.path.join(candidate, "negsel.jar")):
            return candidate
    # Fall back to the zyBooks path so any resulting error is easy to interpret.
    return "/usercode/negsel_src"

def _english_train_path() -> str:
    return os.path.join(_negsel_root(), "tests", "english.train")

def _labels_path() -> str:
    return os.path.join(_negsel_root(), "tests", "labels.test")

@lru_cache(maxsize=None)
def anom_scores(file: str) -> Tuple[float, ...]:
    args = [
        "java",
        "-jar",
        os.path.join(_negsel_root(), "negsel.jar"),
        "-n",
        "10",
        "-r",
        "4",
        "-self",
        _english_train_path(),
    ]
    with open(file, "r", encoding="utf-8") as input_file:
        result = subprocess.run(args, stdin=input_file, stdout=subprocess.PIPE, stderr=subprocess.PIPE, universal_newlines=True,check=False)
    if result.returncode != 0:
        raise RuntimeError(result.stderr)
    return tuple(float(value) for value in result.stdout.split())


@lru_cache(maxsize=None)
def _labels() -> Tuple[int, ...]:
    with open(_labels_path(), "r", encoding="utf-8") as handle:
        return tuple(int(line.strip()) for line in handle if line.strip())


def final_anom(file):
    score = 0.0
    for value in anom_scores(file):
        if value == 0:
            value = 1e-6
        score += math.log(value)
    return score


def fp_tp_calc(file, theta):
    scores = anom_scores(file)
    labels = _labels()

    normal_scores: List[float] = []
    anomaly_scores: List[float] = []
    for score, label in zip(scores, labels):
        if label == 0:
            normal_scores.append(score)
        else:
            anomaly_scores.append(score)

    false_positive_rate = sum(score > theta for score in normal_scores) / len(normal_scores)
    true_positive_rate = sum(score > theta for score in anomaly_scores) / len(anomaly_scores)
    return false_positive_rate, true_positive_rate


def auc_calc(file):
    points = [fp_tp_calc(file, theta) for theta in range(0, 11)]
    # The theta sweep gives FPR values in decreasing order. Sort by FPR so the
    # trapezoidal rule returns a positive area.
    points = sorted(points, key=lambda point: point[0])

    auc = 0.0
    for (x0, y0), (x1, y1) in zip(points, points[1:]):
        auc += (x1 - x0) * (y0 + y1) / 2
    return auc
