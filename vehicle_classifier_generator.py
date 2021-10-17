import glob
import time
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from feature_extraction_functions import *
from sklearn import svm
from sklearn.model_selection import GridSearchCV


def generate_svc_and_params(percent_samples=100):
	"""
	function to generate the support vector classifier (svc) and parameters used to generate the svc
	:param percent_samples: what percentage of samples to use for training
	:return: a dictionary containing the svc and parameters
	"""
	print("Generating classifier...")

	# Divide up into vehicles and nonvehicles
	nonveh_images = glob.glob('non-vehicles/*/*.png')
	veh_images = glob.glob('vehicles/*/*.png')

	print('{} vehicle images available'.format(len(veh_images)))
	print('{} non-vehicle images available'.format(len(nonveh_images)))
	
	sample_size = 2*int(percent_samples*len(veh_images)/100)
	rand_indxs_veh = np.random.randint(0, len(nonveh_images), sample_size)
	nonveh_images = np.array(nonveh_images)[rand_indxs_veh]
	rand_indxs_nonveh = np.random.randint(0, len(veh_images), sample_size)
	veh_images = np.array(veh_images)[rand_indxs_nonveh]

	svc_params = {}
	svc_params['color_space'] = 'YCrCb'  # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	svc_params['orient'] = 9  # HOG orientations
	svc_params['pix_per_cell'] = 8  # HOG pixels per cell
	svc_params['cell_per_block'] = 2  # HOG cells per block
	svc_params['hog_channel'] = 'ALL'  # Can be 0, 1, 2, or "ALL"
	svc_params['spatial_size'] = (32, 32)  # Spatial binning dimensions
	svc_params['hist_bins'] = 32   # Number of histogram bins
	svc_params['spatial_feat_on'] = True  # Spatial features on or off
	svc_params['hist_feat_on'] = True  # Histogram features on or off
	svc_params['hog_feat_on'] = True  # HOG features on or off

	color_space = 'YCrCb' # Can be RGB, HSV, LUV, HLS, YUV, YCrCb
	orient = 9	 # HOG orientations
	pix_per_cell = 8  # HOG pixels per cell
	cell_per_block = 2  # HOG cells per block
	hog_channel = 'ALL'  # Can be 0, 1, 2, or "ALL"
	spatial_size = (32, 32)  # Spatial binning dimensions
	hist_bins = 32	  # Number of histogram bins

	t = time.time()
	car_features = extract_features(veh_images, svc_params)
	notcar_features = extract_features(nonveh_images, svc_params)
	t2 = time.time()
	print('{} samples used'.format(sample_size))
	print('{} seconds to extract HOG features...'.format(round(t2-t, 2)))

	# Create an array stack of feature vectors
	X = np.vstack((car_features, notcar_features)).astype(np.float64)						 
	# Fit a per-column scaler
	X_scaler = StandardScaler().fit(X)
	svc_params['X_scaler'] = X_scaler
	# Apply the scaler to X
	scaled_X = X_scaler.transform(X)

	# Define the labels vector
	y = np.hstack((np.ones(len(veh_images)), np.zeros(len(nonveh_images))))

	# Split up data into randomized training and test sets
	rand_state = np.random.randint(0, 100)
	
	X_train, X_test, y_train, y_test = train_test_split(
		scaled_X, y, test_size=0.25, random_state=rand_state)

	print('Using: orient {}, pixels per cell {}, and cells per block {}'.format(orient, pix_per_cell, cell_per_block))
	print('Feature vector length: {}'.format(len(X_train[0])))


	# Use GridSearch to find the optimal parameters
	svr = svm.SVC()
	parameters = {'kernel': ('linear', 'rbf'), 'C':[1, 10]}
	svc = GridSearchCV(svr, parameters)

	# Check the training time for the SVC
	print('Training SVC...')
	t = time.time()
	svc.fit(X_train, y_train)
	t2 = time.time()
	print('{} seconds to train SVC...'.format(round(t2-t, 2)))
	# Check the score of the SVC
	print('Test Accuracy of SVC = {}'.format(round(svc.score(X_test, y_test), 4)))

	# Check the prediction time for a single sample
	t = time.time()
	n_predict = 100
	print('My SVC predicts: {}'.format(svc.predict(X_test[0:n_predict])))
	print('For these {} labels: {}'.format(n_predict,y_test[0:n_predict]))
	t2 = time.time()
	print('{} seconds to predict {} labels with SVC'.format(round(t2-t, 5), n_predict))
	
	return svc, svc_params