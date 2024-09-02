import argparse
import csv
import random


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('files', nargs='+')
    parser.add_argument('--balance', '-b', action='store_true')
    parser.add_argument('--shuffle', '-s', action='store_true')
    parser.add_argument('--output', '-o', default='train.csv')
    args = parser.parse_args()

    data = {}
    header = None
    for file in args.files:
        data[file] = []
        with open(file) as f:
            if header is None:
                header = next(f)
            else:
                next(f)
            for line in f:
                data[file].append(line)
        print(f'{file}: {len(data[file])} rows')
    
    if args.balance:
        print('Balancing datasets')
        # balance by repeating the smallest datasets
        target_len = max(len(d) for d in data.values())
        for file in data:
            while len(data[file]) < target_len:
                data[file].append(random.choice(data[file]))
            print(f'{file}: {len(data[file])} rows')
    
    data = [row for file in data for row in data[file]]
    print(f'Total: {len(data)} rows')
    
    if args.shuffle:
        random.shuffle(data)
    
    with open(args.output, 'w') as f:
        f.write(header)
        for row in data:
            f.write(row)