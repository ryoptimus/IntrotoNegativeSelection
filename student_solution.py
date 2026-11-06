import os
import math
import subprocess
from pathlib import Path

# train file:
# /usercode/negsel_src/tests/english.train

# ling test files:
# /usercode/negsel_src/tests/hiligaynon.test
# /usercode/negsel_src/tests/middle-english.test
# /usercode/negsel_src/tests/plautdietsch.test
# /usercode/negsel_src/tests/xhosa.test

# label file:
# /usercode/negsel_src/tests/labels.test

# negative selection algorithm:
# /usercode/negsel_src/negsel.jar

# generate anomaly scores for test files:
# java -jar /usercode/negsel_src/negsel.jar -n 10 -r 4 -self /usercode/negsel_src/tests/english.train < TEST_FILE
# In Python, run this command using subprocess.run(...). The test file should be passed as standard input to the Java process

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
    print(f"Yay anom_scores({fp})!")
    args = ["java", "-jar", NGA_FILE, "-n", "10", "-r", "4", "-self", TRAIN_FILE,]
    with open(fp, "r", encoding="utf-8") as f:
        res = subprocess.run(args=args, stdin=f, stdout=subprocess.PIPE, stderr=subprocess.PIPE,)
    if res.returncode != 0:
        raise RuntimeError(res.stderr)
    score_tup = tuple()
    for val in res.stdout.strip():
        score_tup += (val,)

    return score_tup

def final_anom(fp):
    print(f"Yay final_anom({fp})!")

def main():
    print("Hello world!")
    for tf in TEST_FILES:
        tfp = Path(tf)
        print(f"tfp: {tfp}")
        anom_scores(tfp)

if __name__ == "__main__":
    main()
