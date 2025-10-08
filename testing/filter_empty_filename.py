#!/usr/bin/env python3
"""
Script to filter questions from questions.jsonl where file_name is empty.
"""

import json
import sys

def filter_questions(input_file, output_file):
    """
    Filter questions from input JSONL file where file_name is empty string.

    Args:
        input_file (str): Path to input JSONL file
        output_file (str): Path to output JSONL file
    """
    filtered_count = 0
    total_count = 0

    with open(input_file, 'r', encoding='utf-8') as infile, \
         open(output_file, 'w', encoding='utf-8') as outfile:

        for line in infile:
            line = line.strip()
            if not line:
                continue

            total_count += 1
            try:
                data = json.loads(line)

                # Check if file_name is empty string
                if data.get('file_name') == '':
                    outfile.write(line + '\n')
                    filtered_count += 1

            except json.JSONDecodeError as e:
                print(f"Error parsing line {total_count}: {e}", file=sys.stderr)
                continue

    print(f"Processed {total_count} lines")
    print(f"Filtered {filtered_count} questions with empty file_name")
    print(f"Saved to {output_file}")

if __name__ == "__main__":
    input_file = "testing/questions/metadata.jsonl" # https://huggingface.co/datasets/gaia-benchmark/GAIA/blob/main/2023/validation/metadata.jsonl
    output_file = "testing/questions/questions.jsonl"

    filter_questions(input_file, output_file)