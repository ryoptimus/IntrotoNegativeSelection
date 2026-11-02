# CSE 568 Project 03: Negative Selection

Submit a single file named `student_solution.py`.

Your file must define these functions:

```python
def final_anom(file):
    ...

def fp_tp_calc(file, theta):
    ...

def auc_calc(file):
    ...
```

## Assignment summary

This project uses the negative selection implementation provided in `negsel_src/negsel.jar` to classify English and non-English strings.

Training data:

```text
/usercode/negsel_src/tests/english.train
```

Language test files:

```text
/usercode/negsel_src/tests/hiligaynon.test
/usercode/negsel_src/tests/middle-english.test
/usercode/negsel_src/tests/plautdietsch.test
/usercode/negsel_src/tests/xhosa.test
```

Labels:

```text
/usercode/negsel_src/tests/labels.test
```

`labels.test` contains 124 normal English strings labeled `0` followed by 500 anomalous non-English strings labeled `1`.

## Java command

Use this command pattern to generate anomaly scores:

```bash
java -jar /usercode/negsel_src/negsel.jar -n 10 -r 4 -self /usercode/negsel_src/tests/english.train < TEST_FILE
```

Do not use the `-c` flag for anomaly scoring in this project.

## Function requirements

### Question 1: `final_anom(file)`

Return the final anomaly score for the input test file. The score is:

```text
sum(log(alpha_i))
```

where each `alpha_i` is an anomaly score produced by `negsel.jar`. If any `alpha_i` is `0`, use `1e-6` before taking the log.

The hiligaynon sample should round to:

```text
1410.54
```

### Question 2: `fp_tp_calc(file, theta)`

Return a tuple:

```python
(FPR, TPR)
```

where:

```text
FPR = normal instances with anomaly score > theta / total normal instances
TPR = anomalous instances with anomaly score > theta / total anomalous instances
```

Use the labels from `/usercode/negsel_src/tests/labels.test`.

For `hiligaynon.test` and `theta = 7`, the values should round to:

```text
FPR = 0.80
TPR = 1.00
```

### Question 3: `auc_calc(file)`

Calculate the AUC of the ROC curve using integer theta values from `0` through `10`, inclusive.

Use the trapezoidal rule. The hiligaynon sample should round to:

```text
0.78
```

## Notes

The autograder creates a `/usercode/negsel_src` path so code written for the zyBooks notebook should continue to work.
