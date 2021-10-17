import vehicle_classifier_generator
from feature_extraction_functions import *

import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import matplotlib as mpl
import glob
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip
import sys

global svc
global svc_params
try:
	# Load saved svc and parameters if available
	svc, svc_params = pickle.load(open('svc_and_params.p', 'rb'))

	# svc_dict = pickle.load(open('svc_and_params_archive.p', 'rb'))
	# svc = svc_dict['svc']
	# del svc_dict['svc']
	# svc_params = svc_dict
	# svc_params['hog_feat_on'] = True
	# svc_params['spatial_feat_on'] = True
	# svc_params['hist_feat_on'] = True

	print('SVC and parameters loaded from saved file.')
except(OSError, IOError, FileNotFoundError):
	# Generate new classifier and save it
	svc, svc_params = vehicle_classifier_generator.generate_svc_and_params(percent_samples=100)
	pickle.dump((svc, svc_params), open('svc_and_params.p', 'wb'))
	print('SVC and parameters saved to svc.p.')


def find_cars(img, ystart, ystop, scale, svc, svc_params):
	"""
	single function that can extract features using hog sub-sampling and make predictions
	:param img: input image
	:param ystart:
	:param ystop:
	:param scale:
	:param svc: support vector classifier
	:param svc_params: dictionary of svc parameters
	:return:
	"""

	pix_per_cell = svc_params['pix_per_cell']
	orient = svc_params['orient']
	color_space = svc_params['color_space']
	cell_per_block = svc_params['cell_per_block']
	spatial_size = svc_params['spatial_size']
	hist_bins = svc_params['hist_bins']
	X_scaler = svc_params['X_scaler']

	draw_img = np.copy(img)
	
	img_tosearch = img[ystart:ystop,:,:]
	if color_space != 'RGB':
		color_space_str = 'RGB2'+color_space
		ctrans_tosearch = convert_color(img_tosearch, conv=color_space_str)
	else:
		ctrans_tosearch = img_tosearch
	if scale != 1:
		imshape = ctrans_tosearch.shape
		ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
		
	ch1 = ctrans_tosearch[:,:,0]
	ch2 = ctrans_tosearch[:,:,1]
	ch3 = ctrans_tosearch[:,:,2]


	# Define blocks and steps as above
	nxblocks = (ch1.shape[1] // pix_per_cell) - cell_per_block + 1
	nyblocks = (ch1.shape[0] // pix_per_cell) - cell_per_block + 1
	nfeat_per_block = orient * cell_per_block**2
	
	# 64 was the original sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2  # Instead of overlap, define how many cells to step
	nxsteps = (nxblocks - nblocks_per_window) // cells_per_step
	nysteps = (nyblocks - nblocks_per_window) // cells_per_step
	
	# Compute individual channel HOG features for the entire image
	hog1 = get_hog_features(ch1, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog2 = get_hog_features(ch2, orient, pix_per_cell, cell_per_block, feature_vec=False)
	hog3 = get_hog_features(ch3, orient, pix_per_cell, cell_per_block, feature_vec=False)
	box_inds = []
	count_windows=0
	for xb in range(nxsteps):
		for yb in range(nysteps):
			count_windows += 1
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			features = []

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64, 64))

			# Get color features
			if svc_params['spatial_feat_on']:
				spatial_features = bin_spatial(subimg, size=spatial_size)
				features.append(spatial_features)

			if svc_params['hist_feat_on']:
				hist_features = color_hist(subimg, nbins=hist_bins)
				features.append(hist_features)

			if svc_params['hog_feat_on']:
				# Extract HOG for this patch
				hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
				hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
				hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel()
				hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
				features.append(hog_features)

			# Scale features and make a prediction
			test_features = X_scaler.transform(\
							np.hstack(tuple(features)).reshape(1, -1))
			test_prediction = svc.predict(test_features)

			if test_prediction == 1:
				xbox_left = np.int(xleft*scale)
				ytop_draw = np.int(ytop*scale)
				win_draw = np.int(window*scale)
				box_inds.append(((xbox_left, ytop_draw+ystart),(xbox_left+win_draw,ytop_draw+win_draw+ystart)))
	return box_inds


def add_heat(heatmap, bbox_list):
	# Iterate through list of bboxes
	for box in bbox_list:
		# Add += 1 for all pixels inside each bbox
		# Assuming each "box" takes the form ((x1, y1), (x2, y2))
		heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

	# Return updated heatmap
	return heatmap


def apply_threshold(heatmap, threshold):
	# Zero out pixels below the threshold
	heatmap[heatmap <= threshold] = 0
	# Return thresholded map
	return heatmap


def draw_labeled_bboxes(img, labels):
	# Iterate through all detected cars
	bboxes = []
	for car_number in range(1, labels[1]+1):
		# Find pixels with each car_number label value
		nonzero = (labels[0] == car_number).nonzero()
		# Identify x and y values of those pixels
		nonzeroy = np.array(nonzero[0])
		nonzerox = np.array(nonzero[1])
		# Define a bounding box based on min/max x and y
		bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
		# Only add the box to the list if the width is sufficient
		if (bbox[1][0] - bbox[0][0]) >= 64:
			bboxes.append(bbox)
		# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0, 0, 255), 6)
	# Return the image
	return img, bboxes

test_img = glob.glob('test_images/*.jpg')[0]
test_img_ch0 = mpimg.imread(test_img)[:,:,0]
last_frames_heatmaps = np.array([np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0), \
								np.zeros_like(test_img_ch0)])
	
# scale_min = 1.5
# scale_max = 2.5
# scale_step = .5
# scales = np.linspace(scale_min, scale_max, 1+(scale_max-scale_min)/scale_step)
# ystarts = [400, 400, 400]
# ystops = [592, 624, 680]


def process_image(img):
	global svc
	global svc_params
	global last_frames_heatmaps
	
	scale_min = 1.5
	scale_max = 2.5
	scale_step = .5
	scales = np.linspace(scale_min, scale_max, int(1+(scale_max-scale_min)/scale_step))
	ystarts = [400, 400, 400]
	ystops = [592, 624, 680]

	all_box_inds = []
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	draw_img = np.copy(img)
	for scale_ys in zip(scales, ystarts, ystops):
		scale = scale_ys[0]
		ystart = scale_ys[1]
		ystop = scale_ys[2]
		all_box_inds += find_cars(img, ystart, ystop, scale, svc, X_scaler, svc_params)
	heat = add_heat(heat, all_box_inds)
	heat = apply_threshold(heat, 1)
	heatmap = np.clip(heat, 0, 1)
	last_frames_heatmaps = np.roll(last_frames_heatmaps, -1, axis=0)
	last_frames_heatmaps[-1] = heatmap
	sum_heatmap = np.zeros_like(heat)
	for lfh in last_frames_heatmaps:
		sum_heatmap += lfh
	sum_heatmap = apply_threshold(sum_heatmap,6)
	labels = label(sum_heatmap)
	draw_img, labeled_boxes = draw_labeled_bboxes(np.copy(img), labels)
	return draw_img


if __name__ == '__main__':
	args = sys.argv
	mode = sys.argv[1]

	if mode in ('-i', '-image'):
		image_file = sys.argv[2]
	elif mode in ('-v', '-video'):
		video_file = sys.argv[2]
		project_video = VideoFileClip('project_video.mp4')
		clip = project_video.fl_image(process_image)
		clip.write_videofile('annotated.mp4', audio=False)
	else:
		"""
		Only used for processing test images
		"""
		def plot_figures(figures, nrows=1, ncols=1):
			fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
			for ind, title in zip(range(len(figures)), figures):
				axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
				axeslist.ravel()[ind].set_title(title)
				# axeslist.ravel()[ind].set_axis_off()
			plt.tight_layout() # optional

		figs = {}
		scale_min = 1.5
		scale_max = 2.5
		scale_step = .5
		scales = np.linspace(scale_min, scale_max, int(1 + (scale_max - scale_min) / scale_step))
		ystarts = [400, 400, 400]
		ystops = [592, 624, 680]

		test_imgs = glob.glob('test_images/*.jpg')
		print('{} test images found'.format(len(test_imgs)))
		for test_img in test_imgs:
			title = test_img
			img = mpimg.imread(test_img)
			draw_img = np.copy(img)
			all_box_inds = []
			# ystarts = [int(img.shape[0]/3)]
			# ystops = [int(img.shape[0]*7/8)]
			heat = np.zeros_like(img[:,:,0]).astype(np.float)
			for scale_ys in zip(scales, ystarts, ystops):
				scale = scale_ys[0]
				ystart = scale_ys[1]
				ystop = scale_ys[2]
				all_box_inds += find_cars(img, ystart, ystop, scale, svc, svc_params)

			for box_inds in all_box_inds:
				cv2.rectangle(draw_img,box_inds[0],box_inds[1],(0,0,255),6)
				# draw_img = np.copy(img)
			print('boxes created')
			figs.update({title:draw_img})
			heat = add_heat(heat, all_box_inds)
			heat = apply_threshold(heat, 1)
			heatmap = np.clip(heat, 0, 255)
			print('heatmap generated')
			mpl.use('tkagg')
			plt.imshow(heatmap, cmap='hot')
			plt.show()
			labels = label(heatmap)
			draw_img, boxes= draw_labeled_bboxes(np.copy(img), labels)
			print(labels[1], 'cars found')

			plt.imshow(draw_img)
			plt.show()

		plot_figures(figs, 3, 2)
		plt.show()
