import argparse
import json
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from PIL import Image


def main():
	parser = argparse.ArgumentParser(
		description='Decimate trajectory (and optionally odo/laser) and preview on the map'
	)
	parser.add_argument('factor', type=int, help='Decimation factor (> 0)')
	args = parser.parse_args()

	factor = args.factor
	if factor <= 0:
		sys.exit('Decimation factor must be positive')

	# Load map info and image
	with open('map_info.json', 'r') as f:
		map_info = json.load(f)
	map_img = Image.open(map_info['image'])

	# Load and decimate reference trajectory
	ref = np.loadtxt('ref.csv', delimiter=',')
	ref_dec = ref[::factor]
	if ref_dec.size == 0:
		sys.exit('Decimation factor too large; no samples remain')

	x_ref = ref_dec[:, 0]
	y_ref = ref_dec[:, 1]

	fig, ax = plt.subplots(figsize=(12, 10))

	extent = (
		map_info['xlimits'][0],
		map_info['xlimits'][1],
		map_info['ylimits'][0],
		map_info['ylimits'][1],
	)
	ax.imshow(map_img, extent=extent)

	ax.plot(x_ref, y_ref, '.-', label=f'Ref decimated (/{factor})')
	ax.plot(x_ref[0], y_ref[0], 'o', label='Start')
	ax.plot(x_ref[-1], y_ref[-1], 'x', label='End')

	ax.set_xlabel('X Position (m)')
	ax.set_ylabel('Y Position (m)')
	ax.set_title(f'Decimated reference preview (factor {factor})')
	ax.set_xlim(map_info['xlimits'])
	ax.set_ylim(map_info['ylimits'])
	ax.set_aspect('equal')
	ax.grid(True, alpha=0.3)
	ax.legend(loc='best')

	# Buttons for user choice
	btn_ax_save = plt.axes((0.7, 0.02, 0.12, 0.05))
	btn_ax_cancel = plt.axes((0.84, 0.02, 0.12, 0.05))
	btn_save = Button(btn_ax_save, 'Save decimated')
	btn_cancel = Button(btn_ax_cancel, 'Cancel')

	def save_decimated(event):
		odo = np.loadtxt('odo.csv', delimiter=',')
		laser = np.loadtxt('laser.csv', delimiter=',')

		min_len = min(len(ref), len(odo), len(laser))
		if min_len == 0:
			print('No samples available to save')
			plt.close(fig)
			return

		if min_len < len(ref):
			print(f'Warning: trimming to {min_len} samples to match shortest series')

		ref_trim = ref[:min_len]
		odo_trim = odo[:min_len]
		laser_trim = laser[:min_len]

		ref_dec_save = ref_trim[::factor]
		odo_dec = odo_trim[::factor]
		laser_dec = laser_trim[::factor]

		np.savetxt('ref_dec.csv', ref_dec_save, fmt='%.6f', delimiter=',')
		np.savetxt('odo_dec.csv', odo_dec, fmt='%.6f', delimiter=',')
		np.savetxt('laser_dec.csv', laser_dec, fmt='%.6f', delimiter=',')

		print(f'Decimated data saved with factor {factor} -> ref_dec.csv, odo_dec.csv, laser_dec.csv')
		plt.close(fig)

	def discard(event):
		print('Decimation cancelled; no files written')
		plt.close(fig)

	btn_save.on_clicked(save_decimated)
	btn_cancel.on_clicked(discard)

	plt.show()


if __name__ == '__main__':
	main()
