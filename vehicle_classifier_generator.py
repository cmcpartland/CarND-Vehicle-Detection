import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np
import cv2
import glob
import time
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from skimage.feature import hog
from feature_extraction_functions import *
from random import shuffle
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def generate_svc_and_params(percent_samples=100):
	print("Generating classifier...")
	# Divide up into vehicles and nonvehicles
	nonveh_images = glob.glob('non-vehicles/*/*.png')
	veh_images = glob.glob('vehicles/*/*.png')
	
	sample_size = int(percent_samples*len(veh_images)/100)
	rand_indxs_veh = np.random.randint(0, len(nonveh_images), sample_size)
	nonveh_images = np.array(nonveh_images)[rand_indxs_veh]
	rand_indxs_nonveh = np.random.randint(0, len(veh_images), sample_size)
	veh_images = np.array(veh_images)[rand_indxs_nonveh]

	color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9	# HOG orientations
	pix_per_cell = 8 # HOG pixels per cell
	cell_per_block = 2 # HOG cells per block
	hog_channel = 'ALL' # Can be 0, 1, 2, or "ALL"
	spatial_size = (32, 32) # Spatial binning dimensions
	hist_bins = 32	  # Number of histogram bins
	spatial_feat = True # Spatial features on or off
	hist_feat = True # Histogram features on or off
	hog_feat = True # HOG features on or off

	t=time.time()
	car_features = extract_features(veh_images, color_space=color_space, 
						spatial_size=spatial_size, hist_bins=hist_bins, 
						orient=orient, pix_per_cell=pix_per_cell, 
						cell_per_block=cell_per_block, 
						hog_channel=hog_channel, spatial_feat=spatial_feat, 
						hist_feat=hist_feat, hog_feat=hog_feat)
	notcar_features = extract_features(nonveh_images, color_space=color_space, 
						spatial_size=spatial_size, hist_bins=hist_bins, 
						orient=orient, pix_per_cell=pix_per_cell, 
						cell_per_block=cell_per_block, 
						hog_channel=hog_channel, spatial_feat=spatial_feat, 
						hist_feat=hist_feat, hog_feat=hog_feat)
	
	t2 = time.time()
	print(sample_size, 'samples used')
	print(round(t2-t, 2), 'Seconds to extract HOG features...')
	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)						 
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(veh_images)), np.zeros(len(nonveh_images))))

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.25, random_state=rand_state)

	print('Using:',orient,'orientations',pix_per_cell,
		'pixels per cell and', cell_per_block,'cells per block')
	print('Feature vector length:', len(X_train[0]))
	# Use a linear SVC 
	# svc = LinearSVC()
	svr = svm.SVC()
	parameters = {'kernel':('linear', 'rbf'), 'C':[1, 10]}
	svc = grid_search.GridSearchCV(svr, parameters)
	# Check the training time for the SVC
	t=time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print(round(t2-t, 2), 'Seconds to train SVC...')
	# Check the score of the SVC
	print('Test Accuracy of SVC = ', round(svc.score(X_test, y_test), 4))
	# Check the prediction time for a single sample
	t=time.time()
	n_predict = 100
	print('My SVC predicts: ', svc.predict(X_test[0:n_predict]))
	print('For these',n_predict, 'labels: ', y_test[0:n_predict])
	t2 = time.time()
	print(round(t2-t, 5), 'Seconds to predict', n_predict,'labels with SVC')
	
	dict = {'svc': svc, 'X_scaler': X_scaler, 'orient': orient, 'pix_per_cell': pix_per_cell, \
		'cell_per_block': cell_per_block, 'spatial_size': spatial_size, 'hist_bins': hist_bins, \
		'color_space': color_space}
	return dict