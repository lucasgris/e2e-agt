import sys
import csv
import ast

input_csv = sys.argv[1]
# s0;s1;s2;s3;s4;s5;
count_by_strings = {"s0": 0, "s1": 0, "s2": 0, "s3": 0, "s4": 0, "s5": 0}

with open(input_csv, 'r') as f:
    data = csv.DictReader(f, delimiter=';')
    for row in data:
        for s in count_by_strings.keys():
            count_by_strings[s] += len(list(ast.literal_eval(row[s])))
    
# compute weights
total = sum(count_by_strings.values())
print(count_by_strings)
weights = {k: (total-v)/total for k, v in count_by_strings.items()}
print(weights)
print("Add these weights to the config file:")
print(list(weights.values()))