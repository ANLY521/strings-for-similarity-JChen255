Semantic textual similarity using string similarity
---------------------------------------------------

This project examines string similarity metrics for semantic textual similarity.
Though semantics go beyond the surface representations seen in strings, some of these
metrics constitute a good benchmark system for detecting STS.

Data is from the [STS benchmark](http://ixa2.si.ehu.es/stswiki/index.php/STSbenchmark).

**TODO:**
Describe each metric in ~ 1 sentence
1. BLEU: A metric to evaluate the machine-translation quality 
         by calculating n-gram precision adding equal weight to each one.
2. NIST: Based on BLEU metric, NIST also evaluates the machine-translation quality, which not only 
         calculates n-gram precision adding equal weight to each one, but also how informative each n-gram is.
3. Word Error Rate: A metric to evaluate the machine-translation or speech recognition quality through calculating
                    the ratio between the number of substitutions, deletions, insertions, and the total number
                    of words.
4. Longest Common Substring: A metric that can evaluate semantic textual similarity by calculating the longest
                             string which is a substring of two or more strings.
5. Edit Distance: A metric that can evaluate semantic textual similarity by calculating the smallest steps will be taken
                  to transform a string to another.

**TODO:** Fill in the correlations. Expected output for DEV is provided; it is ok if your actual result
varies slightly due to preprocessing/system difference, but the difference should be quite small.

**Correlations:**

Metric | Train | Dev | Test 
------ | -- | --- | ----
NIST | (0.496) | 0.593 | (0.475)
BLEU | (0.371) | 0.433 | (0.353)
WER | (-0.266) | -0.452| (-0.284)
LCS | (0.365) | 0.468| (0.347)
Edit Dist | (0.033) | -0.175| (-0.039)

**TODO:**
Show usage of the homework script with command line flags (see example under lab, week 1).
1.`python sts_pearson.py --sts_data stsbenchmark/sts-dev.csv`
2.`python sts_pearson.py --sts_data stsbenchmark/sts-train.csv`
3.`python sts_pearson.py --sts_data stsbenchmark/sts-test.csv`

## lab, week 1: sts_nist.py

Calculates NIST machine translation metric for sentence pairs in an STS dataset.

Example usage:

`python sts_nist.py --sts_data stsbenchmark/sts-dev.csv`

## lab, week 2: sts_tfidf.py

Calculate pearson's correlation of semantic similarity with TFIDF vectors for text.

## homework, week 1: sts_pearson.py

Calculate pearson's correlation of semantic similarity with the metrics specified in the starter code.
Calculate the metrics between lowercased inputs and ensure that the metric is the same for either order of the 
sentences (i.e. sim(A,B) == sim(B,A)). If not, use the strategy from the lab.
Use SmoothingFunction method0 for BLEU, as described in the nltk documentation.

Run this code on the three partitions of STSBenchmark to fill in the correlations table above.
Use the --sts_data flag and edit PyCharm run configurations to run against different inputs,
 instead of altering your code for each file.