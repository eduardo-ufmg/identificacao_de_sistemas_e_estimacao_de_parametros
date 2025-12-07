import argparse
import numpy as np


def main():
	parser = argparse.ArgumentParser(description='Compute row-wise differences of a CSV file')
	parser.add_argument('name', help='Base name of the CSV file (without .csv)')
	args = parser.parse_args()

	src = f'{args.name}.csv'
	dst = f'{args.name}_diff.csv'

	data = np.loadtxt(src, delimiter=',')
	if data.ndim == 1:
		data = data.reshape(-1, 1)

	diff = np.zeros_like(data)
	diff[1:] = data[1:] - data[:-1]

	np.savetxt(dst, diff, fmt='%.6f', delimiter=',')
	print(f'Wrote differences to {dst}')


if __name__ == '__main__':
	main()