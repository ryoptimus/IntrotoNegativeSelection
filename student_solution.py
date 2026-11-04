import os
import math
import subprocess


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


def main():
    print("Hello world!")

if __name__ == "__main__":
    main()
