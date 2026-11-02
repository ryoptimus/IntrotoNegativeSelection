#!/usr/bin/env python3
"""Gradescope grader for CSE 568 Project 03.

The grader imports /autograder/submission/student_solution.py and tests:
  - final_anom(file)
  - fp_tp_calc(file, theta)
  - auc_calc(file)

It writes /autograder/results/results.json using Gradescope's JSON format.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import signal
import sys
import traceback
from dataclasses import dataclass
from typing import Callable, List, Sequence, Tuple

RESULTS_DIR = os.environ.get("GRADESCOPE_RESULTS_DIR", "/autograder/results")
SOURCE_DIR = os.environ.get("GRADESCOPE_SOURCE_DIR", "/autograder/source")
SUBMISSION_DIR = os.environ.get("GRADESCOPE_SUBMISSION_DIR", "/autograder/submission")

# Make local execution easy while preserving Gradescope behavior.
if not os.path.isdir(SOURCE_DIR):
    SOURCE_DIR = os.getcwd()
if not os.path.isdir(SUBMISSION_DIR):
    SUBMISSION_DIR = os.getcwd()

for path in [SUBMISSION_DIR, SOURCE_DIR, os.path.join(SOURCE_DIR, "tests")]:
    if path not in sys.path:
        sys.path.insert(0, path)


def ensure_usercode_paths() -> None:
    """Create /usercode/negsel_src when permissions allow, matching zyBooks paths."""
    source_negsel = os.path.join(SOURCE_DIR, "negsel_src")
    usercode = "/usercode"
    target = os.path.join(usercode, "negsel_src")
    try:
        os.makedirs(usercode, exist_ok=True)
        if os.path.islink(target) or os.path.exists(target):
            if os.path.islink(target):
                os.unlink(target)
            elif os.path.abspath(target) != os.path.abspath(source_negsel):
                return
        os.symlink(source_negsel, target)
    except Exception:
        # The grader can still pass absolute source paths if /usercode is unavailable.
        pass


ensure_usercode_paths()

TEST_DIR = "/usercode/negsel_src/tests"
if not os.path.isdir(TEST_DIR):
    TEST_DIR = os.path.join(SOURCE_DIR, "negsel_src", "tests")


EXPECTED_FINAL_ANOM = {
    "hiligaynon.test": 1410.5422463062357,
    "middle-english.test": 1351.0524246036819,
    "plautdietsch.test": 1405.503801625438,
    "xhosa.test": 1411.5676805858473,
}

EXPECTED_FP_TP = {
    ("hiligaynon.test", 7): (0.7983870967741935, 1.0),
    ("middle-english.test", 8): (0.6129032258064516, 0.666),
    ("plautdietsch.test", 6): (0.9112903225806451, 0.998),
    ("xhosa.test", 9): (0.3870967741935484, 0.934),
    ("hiligaynon.test", 10): (0.0, 0.0),
}

EXPECTED_AUC = {
    "hiligaynon.test": 0.7770161290322581,
    "middle-english.test": 0.5365,
    "plautdietsch.test": 0.7528064516129032,
    "xhosa.test": 0.7834596774193548,
}


def test_path(filename: str) -> str:
    return os.path.join(TEST_DIR, filename)


class TimeoutError(Exception):
    pass


def _timeout_handler(signum, frame):  # type: ignore[no-untyped-def]
    raise TimeoutError("Timed out while running this check")


def run_with_timeout(func: Callable[[], None], seconds: int = 20) -> None:
    old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
    signal.alarm(seconds)
    try:
        func()
    finally:
        signal.alarm(0)
        signal.signal(signal.SIGALRM, old_handler)


def numeric_value(value, *, label: str = "return value") -> float:
    """Accept DEAP-style (x,), one-item lists, numpy scalars, or plain floats."""
    if isinstance(value, (tuple, list)):
        if not value:
            raise AssertionError(f"{label} was an empty tuple/list")
        value = value[0]
    try:
        return float(value)
    except Exception as exc:
        raise AssertionError(f"{label} {value!r} is not numeric") from exc


def assert_close(actual, expected: float, *, label: str, tol: float = 1e-6) -> None:
    actual_float = numeric_value(actual, label=label)
    if not math.isclose(actual_float, expected, rel_tol=tol, abs_tol=tol):
        raise AssertionError(f"{label}: expected {expected}, got {actual_float}")


def assert_close_allow_rounded(actual, expected: float, *, label: str, decimals: int = 2) -> None:
    """Accept exact values or values rounded to the requested number of decimals."""
    actual_float = numeric_value(actual, label=label)
    rounded_expected = round(expected, decimals)
    if math.isclose(actual_float, expected, rel_tol=1e-6, abs_tol=1e-6):
        return
    if math.isclose(actual_float, rounded_expected, rel_tol=1e-9, abs_tol=0.5 * 10 ** (-decimals)):
        return
    raise AssertionError(
        f"{label}: expected {expected} or approximately {rounded_expected}, got {actual_float}"
    )


def assert_pair_close(actual, expected: Tuple[float, float], *, label: str, tol: float = 1e-6) -> None:
    if not isinstance(actual, (tuple, list)) or len(actual) != 2:
        raise AssertionError(f"{label}: expected a tuple/list of length 2, got {actual!r}")
    assert_close(actual[0], expected[0], label=f"{label} FPR", tol=tol)
    assert_close(actual[1], expected[1], label=f"{label} TPR", tol=tol)


@dataclass
class GradedCheck:
    name: str
    max_score: float
    func: Callable[[object], None]


def check_final_anom_hiligaynon(student) -> None:
    assert_close_allow_rounded(
        student.final_anom(test_path("hiligaynon.test")),
        EXPECTED_FINAL_ANOM["hiligaynon.test"],
        label="final_anom hiligaynon.test",
    )


def check_final_anom_plautdietsch(student) -> None:
    assert_close_allow_rounded(
        student.final_anom(test_path("plautdietsch.test")),
        EXPECTED_FINAL_ANOM["plautdietsch.test"],
        label="final_anom plautdietsch.test",
    )


def check_final_anom_xhosa(student) -> None:
    assert_close_allow_rounded(
        student.final_anom(test_path("xhosa.test")),
        EXPECTED_FINAL_ANOM["xhosa.test"],
        label="final_anom xhosa.test",
    )


def check_fp_tp_hiligaynon_theta7(student) -> None:
    expected = EXPECTED_FP_TP[("hiligaynon.test", 7)]
    assert_pair_close(
        student.fp_tp_calc(test_path("hiligaynon.test"), 7),
        expected,
        label="fp_tp_calc hiligaynon.test theta=7",
    )


def check_fp_tp_middle_english_theta8(student) -> None:
    expected = EXPECTED_FP_TP[("middle-english.test", 8)]
    assert_pair_close(
        student.fp_tp_calc(test_path("middle-english.test"), 8),
        expected,
        label="fp_tp_calc middle-english.test theta=8",
    )


def check_fp_tp_plautdietsch_theta6(student) -> None:
    expected = EXPECTED_FP_TP[("plautdietsch.test", 6)]
    assert_pair_close(
        student.fp_tp_calc(test_path("plautdietsch.test"), 6),
        expected,
        label="fp_tp_calc plautdietsch.test theta=6",
    )


def check_fp_tp_xhosa_theta9(student) -> None:
    expected = EXPECTED_FP_TP[("xhosa.test", 9)]
    assert_pair_close(
        student.fp_tp_calc(test_path("xhosa.test"), 9),
        expected,
        label="fp_tp_calc xhosa.test theta=9",
    )


def check_fp_tp_boundary_theta10(student) -> None:
    expected = EXPECTED_FP_TP[("hiligaynon.test", 10)]
    assert_pair_close(
        student.fp_tp_calc(test_path("hiligaynon.test"), 10),
        expected,
        label="fp_tp_calc hiligaynon.test theta=10 boundary",
    )


def check_auc_hiligaynon(student) -> None:
    assert_close_allow_rounded(
        student.auc_calc(test_path("hiligaynon.test")),
        EXPECTED_AUC["hiligaynon.test"],
        label="auc_calc hiligaynon.test",
    )


def check_auc_middle_english(student) -> None:
    assert_close_allow_rounded(
        student.auc_calc(test_path("middle-english.test")),
        EXPECTED_AUC["middle-english.test"],
        label="auc_calc middle-english.test",
    )


def check_auc_plautdietsch(student) -> None:
    assert_close_allow_rounded(
        student.auc_calc(test_path("plautdietsch.test")),
        EXPECTED_AUC["plautdietsch.test"],
        label="auc_calc plautdietsch.test",
    )


def check_auc_xhosa(student) -> None:
    assert_close_allow_rounded(
        student.auc_calc(test_path("xhosa.test")),
        EXPECTED_AUC["xhosa.test"],
        label="auc_calc xhosa.test",
    )


def build_checks() -> List[GradedCheck]:
    return [
        GradedCheck("Q1 final_anom hiligaynon sample", 10, check_final_anom_hiligaynon),
        GradedCheck("Q1 final_anom plautdietsch", 10, check_final_anom_plautdietsch),
        GradedCheck("Q1 final_anom xhosa", 10, check_final_anom_xhosa),
        GradedCheck("Q2 fp_tp_calc hiligaynon theta=7 sample", 10, check_fp_tp_hiligaynon_theta7),
        GradedCheck("Q2 fp_tp_calc middle-english theta=8", 7, check_fp_tp_middle_english_theta8),
        GradedCheck("Q2 fp_tp_calc plautdietsch theta=6", 7, check_fp_tp_plautdietsch_theta6),
        GradedCheck("Q2 fp_tp_calc xhosa theta=9", 7, check_fp_tp_xhosa_theta9),
        GradedCheck("Q2 fp_tp_calc theta=10 boundary", 4, check_fp_tp_boundary_theta10),
        GradedCheck("Q3 auc_calc hiligaynon sample", 10, check_auc_hiligaynon),
        GradedCheck("Q3 auc_calc middle-english", 8, check_auc_middle_english),
        GradedCheck("Q3 auc_calc plautdietsch", 8, check_auc_plautdietsch),
        GradedCheck("Q3 auc_calc xhosa", 9, check_auc_xhosa),
    ]


def result_entry(name: str, score: float, max_score: float, output: str) -> dict:
    return {
        "name": name,
        "score": score,
        "max_score": max_score,
        "output": output,
        "visibility": "visible",
    }


def write_results(payload: dict) -> None:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(os.path.join(RESULTS_DIR, "results.json"), "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def main() -> int:
    tests = []
    total_score = 0.0
    max_score = 100.0

    try:
        student = importlib.import_module("student_solution")
    except Exception:
        output = (
            "Could not import student_solution.py. Remove unguarded notebook/demo code that fails "
            "during import and make sure the file is named student_solution.py.\n\n"
        )
        output += traceback.format_exc(limit=20)
        payload = {
            "score": 0,
            "visibility": "visible",
            "stdout_visibility": "visible",
            "output": output,
            "tests": [result_entry("Import student_solution.py", 0, max_score, output)],
        }
        write_results(payload)
        print(output)
        return 0

    for check in build_checks():
        try:
            run_with_timeout(lambda check=check: check.func(student), seconds=30)
        except Exception:
            output = traceback.format_exc(limit=8)
            tests.append(result_entry(check.name, 0, check.max_score, output))
            print(f"FAIL: {check.name}\n{output}")
        else:
            tests.append(result_entry(check.name, check.max_score, check.max_score, "Passed"))
            total_score += check.max_score
            print(f"PASS: {check.name}")

    payload = {
        "score": total_score,
        "visibility": "visible",
        "stdout_visibility": "visible",
        "output": f"Project 03 autograder score: {total_score:.1f} / {max_score:.1f}",
        "tests": tests,
    }
    write_results(payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
