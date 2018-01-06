import vehicle_classifier_generator
import pickle
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.preprocessing import StandardScaler
from feature_extraction_functions import *
import glob
from scipy.ndimage.measurements import label
from moviepy.editor import VideoFileClip

try:
	# Load saved svc and parameters if available
	dict = pickle.load(open('svc_and_params.p', 'rb'))
	print('SVC and parameters loaded from saved file.')
except(OSError, IOError):
	# Generate new classifier and save it
	dict = vehicle_classifier_generator.generate_svc_and_params(percent_samples=100)
	pickle.dump(dict, open('svc_and_params.p', 'wb'))
	print('SVC and parameters saved to svc.p.')

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space):
	
	draw_img = np.copy(img)
	
	img_tosearch = img[ystart:ystop,:,:]
	if color_space != 'RGB':
		color_space = 'RGB2'+color_space
		ctrans_tosearch = convert_color(img_tosearch, conv=color_space)
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
	nfeat_per_block = orient*cell_per_block**2
	
	# 64 was the original sampling rate, with 8 cells and 8 pix per cell
	window = 64
	nblocks_per_window = (window // pix_per_cell) - cell_per_block + 1
	cells_per_step = 2	# Instead of overlap, define how many cells to step
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
			count_windows+=1
			ypos = yb*cells_per_step
			xpos = xb*cells_per_step
			# Extract HOG for this patch
			hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
			hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))

			xleft = xpos*pix_per_cell
			ytop = ypos*pix_per_cell

			# Extract the image patch
			subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
		  
			# Get color features
			spatial_features = bin_spatial(subimg, size=spatial_size)
			hist_features = color_hist(subimg, nbins=hist_bins)

			# Scale features and make a prediction
			test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))	
	  
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
		if ((bbox[1][0] - bbox[0][0]) >= 64):
			bboxes.append(bbox)
		# Draw the box on the image
			cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
	# Return the image
	return img, bboxes
			
global svc
global pix_per_cell
global orient
global spatial_size
global cell_per_block
global hist_bins
global X_scaler
global color_space
global last_frames_heatmaps

svc = dict['svc']
pix_per_cell = dict['pix_per_cell']
orient = dict['orient']
spatial_size = dict['spatial_size']
cell_per_block = dict['cell_per_block']
hist_bins = dict['hist_bins']
X_scaler = dict['X_scaler']
color_space = dict['color_space']

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
	
scale_min = 1.5
scale_max = 2.5
scale_step = .5
scales = np.linspace(scale_min, scale_max, 1+(scale_max-scale_min)/scale_step)
ystarts = [400, 400, 400]
ystops = [592, 624, 680]

def process_image(img):
	global svc
	global pix_per_cell
	global orient
	global spatial_size
	global cell_per_block
	global hist_bins
	global X_scaler
	global color_space
	global last_frames_heatmaps
	
	scale_min = 1.5
	scale_max = 2.5
	scale_step = .5
	scales = np.linspace(scale_min, scale_max, 1+(scale_max-scale_min)/scale_step)
	ystarts = [400, 400, 400]
	ystops = [592, 624, 680]
	
	all_box_inds = []
	heat = np.zeros_like(img[:,:,0]).astype(np.float)
	draw_img = np.copy(img)
	for scale_ys in zip(scales, ystarts, ystops):
		scale = scale_ys[0]
		ystart = scale_ys[1]
		ystop = scale_ys[2]
		all_box_inds += find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)
	heat = add_heat(heat,all_box_inds)
	heat = apply_threshold(heat,1)
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

project_video = VideoFileClip('project_video.mp4')
clip = project_video.fl_image(process_image)
clip.write_videofile('annotated.mp4', audio=False)


"""
Only used for processing test images
"""
# def plot_figures(figures, nrows = 1, ncols=1):
	# fig, axeslist = plt.subplots(ncols=ncols, nrows=nrows)
	# for ind,title in zip(range(len(figures)), figures):
		# axeslist.ravel()[ind].imshow(figures[title], cmap=plt.gray())
		# axeslist.ravel()[ind].set_title(title)
		# # axeslist.ravel()[ind].set_axis_off()
	# plt.tight_layout() # optional
	
# figs = {}
# scales = np.linspace(scale_min, scale_max, 1+(scale_max-scale_min)/scale_step)
# ystarts = [400, 400, 400]
# ystops = [592, 624, 680]

# test_imgs = glob.glob('test_images/*.jpg')
# for test_img in test_imgs:
	# title = test_img
	# img = mpimg.imread(test_img)
	# draw_img = np.copy(img)
	# all_box_inds = []
	# heat = np.zeros_like(img[:,:,0]).astype(np.float)
	# for scale_ys in zip(scales, ystarts, ystops):#np.linspace(scale_min, scale_max, (scale_max - scale_min)/scale_step):
		# scale = scale_ys[0]
		# ystart = scale_ys[1]
		# ystop = scale_ys[2]
		# all_box_inds += find_cars(img, ystart, ystop, scale, svc, X_scaler, orient, pix_per_cell, cell_per_block, spatial_size, hist_bins, color_space)

	# for box_inds in all_box_inds:
		# cv2.rectangle(draw_img,box_inds[0],box_inds[1],(0,0,255),6) 
		# # draw_img = np.copy(img)
	# figs.update({title:draw_img})
	# heat = add_heat(heat,all_box_inds)
	# heat = apply_threshold(heat,1)
	# heatmap = np.clip(heat, 0, 255)
	# plt.imshow(heatmap, cmap='hot')
	# plt.show()
	# labels = label(heatmap)
	# draw_img, boxes= draw_labeled_bboxes(np.copy(img), labels)
	# print(labels[1], 'cars found')

	

	# plt.imshow(draw_img)
	# plt.show()

# plot_figures(figs, 3, 2)
# plt.show()
