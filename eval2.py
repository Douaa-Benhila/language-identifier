#!/usr/bin/env python3

import sys
import argparse

# Traitement des arguments en ligne de commande
parser = argparse.ArgumentParser(description="Evaluate language detection accuracy.")
parser.add_argument("predfile", type=argparse.FileType("r", encoding="UTF-8"),
                    help="Prediction text file, with one sentence per line, UTF-8")
parser.add_argument("goldfile", type=argparse.FileType("r", encoding="UTF-8"),
                    help="Gold/reference text file, with one sentence per line, UTF-8")
args = parser.parse_args()

total = tp = 0

for pred_line, gold_line in zip(args.predfile, args.goldfile):
    try:
        pred_text, pred_lang = pred_line.strip().split("\t", 1)
        gold_text, gold_lang = gold_line.strip().split("\t", 1)
    except ValueError:
        print(f"Error line {total+1}: file not well formatted", file=sys.stderr)
        print(f"Problematic line {total+1} in pred: '{pred_line.strip()}'", file=sys.stderr)
        print(f"Problematic line {total+1} in gold: '{gold_line.strip()}'", file=sys.stderr)
        sys.exit(-1)

    if pred_text != gold_text:
        print(f"Error line {total+1}: pred and gold files not aligned!", file=sys.stderr)
        sys.exit(-1)

    if pred_lang == gold_lang:
        tp += 1

    total += 1

accuracy = 100.0 * (tp / total)
print(f"Predictions file: {args.predfile.name}")
print(f"Gold/reference file: {args.goldfile.name}")
print(f"Accuracy: {accuracy:.2f}% ({tp}/{total})")
