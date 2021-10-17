import cv2
import numpy as np
from skimage.feature import hog


def convert_color(img, conv='RGB2YCrCb'):
	"""
	converts an image from one color space to another
	:param img: input image
	:param conv: color space conversion string
	:return: converted image
	"""
	if conv == 'RGB2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
	if conv == 'BGR2YCrCb':
		return cv2.cvtColor(img, cv2.COLOR_BGR2YCrCb)
	if conv == 'RGB2LUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2LUV)
	if conv == 'RGB2HSV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
	if conv == 'RGB2HLS':
		return cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
	if conv == 'RGB2YUV':
		return cv2.cvtColor(img, cv2.COLOR_RGB2YUV)


def get_hog_features(img, orient, pix_per_cell, cell_per_block, vis=False, feature_vec=True):
	"""
	function to get a HOG feature vector from an input image
	:param img: the input image
	:param orient:
	:param pix_per_cell:
	:param cell_per_block:
	:param vis: a boolean, whether or not a visualization of the HOG result is desired
	:param feature_vec:
	:return: feature vector and a HOG image if vis == True
	"""
	# Call with two outputs if vis==True
	if vis:
		features, hog_image = hog(img, orientations=orient, 
								  pixels_per_cell=(pix_per_cell, pix_per_cell),
								  cells_per_block=(cell_per_block, cell_per_block), 
								  transform_sqrt=False, 
								  visualise=vis, feature_vector=feature_vec)
		return features, hog_image
	# Otherwise call with one output
	else:	   
		features = hog(img, orientations=orient,
					   pixels_per_cell=(pix_per_cell, pix_per_cell),
					   cells_per_block=(cell_per_block, cell_per_block),
					   transform_sqrt=False,
					   visualize=vis, feature_vector=feature_vec)
		return features


def bin_spatial(img, size=(32, 32)):
	"""
	compute spatial binning for an input image
	the image is simply resized to the given input size, then 'ravelled' to a 1d array
	:param img: input image
	:param size: pixel height, pixel width for resizing
	:return: horizontal array of spatially binned image
	"""
	color1 = cv2.resize(img[:,:,0], size).ravel()
	color2 = cv2.resize(img[:,:,1], size).ravel()
	color3 = cv2.resize(img[:,:,2], size).ravel()
	return np.hstack((color1, color2, color3))


def color_hist(img, nbins=32):	  # bins_range=(0, 256)
	"""
	compute a histogram of each color channel
	the color range is from 0 to 255, and it will be split into nbins
	:param img: input image
	:param nbins: number of bins to be used for the histogram
	:return: histograms concatenated as a feature vector
	"""
	# Compute the histogram of the color channels separately
	channel1_hist = np.histogram(img[:,:,0], bins=nbins)
	channel2_hist = np.histogram(img[:,:,1], bins=nbins)
	channel3_hist = np.histogram(img[:,:,2], bins=nbins)
	# Concatenate the histograms into a single feature vector
	hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
	# Return the individual histograms, bin_centers and feature vector
	return hist_features


def extract_features(image_files, svc_params):
	"""
	Function to extract features from a list of images
	There are three feature extraction techniques: spatial binning, color histogram, and HOG; each one can be
	deactivated or activated as desired
	:param image_files: list of image file paths
	:param svc_params: parameters for the svc
	:return: list of feature vectors
	"""

	# Define an empty list for feature vectors
	features = []

	# Iterate through the list of images
	for image_file in image_files:
		file_features = []  # this will be the feature vector
		image = cv2.imread(image_file)

		# Apply color conversion if other than 'RGB'
		if svc_params['color_space'] != 'RGB':
			color_space_str = 'RGB2' + svc_params['color_space']
			feature_image = convert_color(image, color_space_str)
		else:
			feature_image = np.copy(image)

		# Compute spatial features if flag is set
		if svc_params['spatial_feat_on']:
			spatial_features = bin_spatial(feature_image, size=svc_params['spatial_size'])
			file_features.append(spatial_features)  # add the spatial features to the feature vector

		# Compute histogram features if flag is set
		if svc_params['hist_feat_on']:
			hist_features = color_hist(feature_image, nbins=svc_params['hist_bins'])
			file_features.append(hist_features)  # add the histogram features to the feature vector

		# Compute the HOG features if flag is set
		if svc_params['hog_feat_on']:
			# Call get_hog_features() with vis=False, feature_vec=True
			if svc_params['hog_channel'] == 'ALL':
				hog_features = []
				for channel in range(feature_image.shape[2]):
					hog_features.append(get_hog_features(feature_image[:,:,channel], 
										svc_params['orient'], svc_params['pix_per_cell'], svc_params['cell_per_block'],
										vis=False, feature_vec=True))
				hog_features = np.ravel(hog_features)		 
			else:
				hog_features = get_hog_features(feature_image[:,:,svc_params['hog_channel']], svc_params['orient'],
							svc_params['pix_per_cell'], svc_params['cell_per_block'], vis=False, feature_vec=True)
			file_features.append(hog_features)  # add the HOG features to the feature vector

		features.append(np.concatenate(file_features))  # Append the new feature vector to the features list
	# Return list of feature vectors
	return features


def single_img_features(img, svc_params):
	"""
	Function to extract features from a single image window
	This function is very similar to extract_features(), but it applies to a single image rather than list of images
	:param img: input image
	:param svc_params: svc parameters
	:return: feature list
	"""
	# 1) Define an empty list for features
	img_features = []

	# 2) Apply color conversion if other than 'RGB'
	if svc_params['color_space'] != 'RGB':
		color_space_str = 'RGB2' + svc_params['color_space']
		feature_image = convert_color(img, color_space_str)
	else:
		feature_image = np.copy(img)

	# 3) Compute spatial features if flag is set
	if svc_params['spatial_feat_on']:
		spatial_features = bin_spatial(feature_image, size=svc_params['spatial_size'])
		# 4) Append features to list
		img_features.append(spatial_features)

	# 5) Compute histogram features if flag is set
	if svc_params['hist_feat_on']:
		hist_features = color_hist(feature_image, nbins=svc_params['hist_bins'])
		# 6) Append features to list
		img_features.append(hist_features)

	# 7) Compute HOG features if flag is set
	if svc_params['hog_feat_on']:
		# Call get_hog_features() with vis=False, feature_vec=True
		if svc_params['hog_channel'] == 'ALL':
			hog_features = []
			for channel in range(feature_image.shape[2]):
				hog_features.append(get_hog_features(feature_image[:, :, channel],
													svc_params['orient'], svc_params['pix_per_cell'],
													svc_params['cell_per_block'],
													vis=False, feature_vec=True))
			hog_features = np.ravel(hog_features)
		else:
			hog_features = get_hog_features(feature_image[:, :, svc_params['hog_channel']], svc_params['orient'],
											svc_params['pix_per_cell'], svc_params['cell_per_block'], vis=False,
											feature_vec=True)
		# 8) add the HOG features to the feature vector
		img_features.append(hog_features)

	# 9) Return concatenated array of features
	return np.concatenate(img_features)