from nltk import word_tokenize
from nltk.translate.nist_score import sentence_nist
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from nltk import edit_distance
from difflib import SequenceMatcher
from Levenshtein import distance
from util import parse_sts
import argparse
import numpy as np
from scipy.stats import pearsonr

def symmetrical_nist(text_pair):
    t1,t2 = text_pair
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())
    try:
        nist_1 = sentence_nist([t1_toks, ], t2_toks)
    except ZeroDivisionError:
        nist_1 = 0.0
    try:
        nist_2 = sentence_nist([t2_toks, ], t1_toks)
    except ZeroDivisionError:
        nist_2 = 0.0
    return nist_1 + nist_2

def symmetrical_bleu(text_pair):
    t1,t2 = text_pair
    t1_toks = word_tokenize(t1.lower())
    t2_toks = word_tokenize(t2.lower())
    try:
        bleu_1 = sentence_bleu([t1_toks, ], t2_toks,  smoothing_function=SmoothingFunction().method0)
    except ZeroDivisionError:
        bleu_1 = 0.0
    try:
        bleu_2 = sentence_bleu([t2_toks, ], t1_toks, smoothing_function=SmoothingFunction().method0)
    except ZeroDivisionError:
        bleu_2 = 0.0
    return bleu_1 + bleu_2

def symmetrical_ed(text_pair):
    t1,t2 = text_pair
    t1_toks = t1.lower()
    t2_toks = t2.lower()
    try:
        ed_1 = distance(t1_toks, t2_toks)
    except ZeroDivisionError:
        ed_1 = 0.0
    try:
        ed_2 = distance(t2_toks, t1_toks)
    except ZeroDivisionError:
        ed_2 = 0.0
    return ed_1 + ed_2

def symmetrical_wer(text_pair):
    t1,t2 = text_pair
    t1_lower = t1.lower()
    t2_lower = t2.lower()
    t1_toks = word_tokenize(t1_lower)
    t2_toks = word_tokenize(t2_lower)
    try:
        wer_1 = (edit_distance(t1_lower, t2_lower))/max(len(t2_toks),len(t1_toks))
    except ZeroDivisionError:
        wer_1 = 0.0
    try:
        wer_2 = (edit_distance(t2_lower, t1_lower))/min(len(t2_toks),len(t1_toks))
    except ZeroDivisionError:
        wer_2 = 0.0
    return wer_1 + wer_2

def symmetrical_lcs(text_pair):
    t1,t2 = text_pair
    t1_toks = t1.lower()
    t2_toks = t2.lower()
    try:
        seqMatch1 = SequenceMatcher(None, t1_toks, t2_toks)
        match1 = seqMatch1.find_longest_match(0, len(t1_toks), 0, len(t2_toks))
        lcs_1 = match1.size
    except ZeroDivisionError:
        lcs_1 = 0.0
    try:
        seqMatch2 = SequenceMatcher(None, t2_toks, t1_toks)
        match2 = seqMatch2.find_longest_match(0, len(t2_toks), 0, len(t1_toks))
        lcs_2 = match2.size
    except ZeroDivisionError:
        lcs_2 = 0.0
    return lcs_1 + lcs_2


def main(sts_data):
    """Calculate pearson correlation between semantic similarity scores and string similarity metrics.
    Data is formatted as in the STS benchmark"""

    # TODO 1: read the dataset; implement in util.py
    texts, labels = parse_sts(sts_data)

    print(f"Found {len(texts)} STS pairs")

    # TODO 2: Calculate each of the the metrics here for each text pair in the dataset
    # HINT: Longest common substring can be complicated. Investigate difflib.SequenceMatcher for a good option.
    score_types = ["NIST", "BLEU", "Word Error Rate", "Longest common substring", "Edit Distance"]
    sample_text1 = texts
    sample_labels1 = labels
    sample_data1= zip(sample_labels1, sample_text1)

    sample_text2 = texts
    sample_labels2 = labels
    sample_data2= zip(sample_labels2, sample_text2)

    sample_text3 = texts
    sample_labels3 = labels
    sample_data3= zip(sample_labels3, sample_text3)

    sample_text4 = texts
    sample_labels4 = labels
    sample_data4= zip(sample_labels4, sample_text4)

    sample_text5 = texts
    sample_labels5 = labels
    sample_data5= zip(sample_labels5, sample_text5)


    nist_scores = []
    for label,text_pair in sample_data1:
        nist_total = symmetrical_nist(text_pair)
        nist_scores.append(nist_total)

    bleu_scores = []
    for label,text_pair in sample_data2:
        bleu_total = symmetrical_bleu(text_pair)
        bleu_scores.append(bleu_total)

    ed_scores = []
    for label,text_pair in sample_data3:
        ed_total = symmetrical_ed(text_pair)
        ed_scores.append(ed_total)

    wer_scores = []
    for label,text_pair in sample_data4:
        wer_total = symmetrical_wer(text_pair)
        wer_scores.append(wer_total)

    lcs_scores = []
    for label,text_pair in sample_data5:
        lcs_total = symmetrical_lcs(text_pair)
        lcs_scores.append(lcs_total)

    nist_corr, _ = pearsonr(nist_scores, sample_labels1)
    bleu_corr, _ = pearsonr(bleu_scores, sample_labels2)
    ed_corr, _ = pearsonr(ed_scores, sample_labels3)
    wer_corr, _ = pearsonr(wer_scores, sample_labels4)
    lcs_corr, _ = pearsonr(lcs_scores, sample_labels5)

    #TODO 3: Calculate pearson r between each metric and the STS labels and report in the README.
    # Sample code to print results. You can alter the printing as you see fit. It is most important to put the results
    # in a table in the README
    print(f"\nSemantic textual similarity for {sts_data}\n")
    print(f"Nist correlation: {nist_corr:.03f}")
    print(f"BLEU correlation: {bleu_corr:.03f}")
    print(f"Word Error Rate correlation: {wer_corr:.03f}")
    print(f"Longest common substring correlation: {lcs_corr:.03f}")
    print(f"Edit Distance correlation: {ed_corr:.03f}")

    # TODO 4: Complete writeup as specified by TODOs in README (describe metrics; show usage)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--sts_data", type=str, default="stsbenchmark/sts-test.csv",
                        help="tab separated sts data in benchmark format")
    args = parser.parse_args()

    main(args.sts_data)

