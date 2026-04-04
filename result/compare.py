import argparse
import sys

def parse_file_to_dict(file_path):
    """Parse a file into a dictionary of name: value pairs."""
    data = {}
    try:
        with open(file_path, 'r') as f:
            for line in f:
                if ',' not in line:
                    continue
                line = line.strip()
                parts = line.split(',', 1)
                if len(parts) != 2:
                    continue
                name = parts[0].strip()
                num_str = parts[1].strip()
                try:
                    num = float(num_str)
                    data[name] = num
                except ValueError:
                    print(f"Warning: Invalid number '{num_str}' in {file_path}, skipping.")
    except FileNotFoundError:
        print(f"Error: File {file_path} not found.")
        sys.exit(1)
    return data

def compute_percentage_change(a_dict, b_dict, output_file):
    """Compute percentage change and write to output file."""
    with open(output_file, 'w') as f:
        for key, a_v in a_dict.items():
            if key not in b_dict:
                print(f"Warning: Key '{key}' not found in second file, skipping.")
                continue
            b_v = b_dict[key]
            if a_v == 0:
                print(f"Warning: Division by zero for key '{key}', skipping.")
                continue
            ratio = (b_v - a_v) / a_v * 100
            f.write(f'{key},\t{ratio:.3f}%\n')

def main():
    parser = argparse.ArgumentParser(description="Compare two result files and compute percentage changes.")
    parser.add_argument('--file_a', help="Path to the first result file", default="/home/hsiaochuan/slam/faster-lio/result/test_results/mcd_viral_2026-04-04-10-23-55.txt")
    parser.add_argument('--file_b', help="Path to the second result file", default="/home/hsiaochuan/slam/faster-lio/result/test_results/mcd_viral_2026-04-04-15-24-58.txt")
    parser.add_argument('--output', default='./compare.txt', help="Output file path (default: /tmp/compare.txt)")
    args = parser.parse_args()

    a_dict = parse_file_to_dict(args.file_a)
    b_dict = parse_file_to_dict(args.file_b)
    compute_percentage_change(a_dict, b_dict, args.output)
    print(f"Comparison completed. Results saved to {args.output}")

if __name__ == "__main__":
    main()
