import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, convolve
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import pickle
import math

#sift object
sift = cv.SIFT_create()

# Load the data using np.load
data = np.load('resources/training_images_full.npz', allow_pickle=True)

# Extract the images
images = data['images']
# and the data points
pts = data['points']#

print(images.shape)
print(pts.shape)

#print(images.shape, pts.shape)

# Load the data that only has a subset of annotations using np.load
data = np.load('resources/training_images_subset.npz', allow_pickle=True)

# Extract the images
images_subset = data['images']
# and the data points
pts_subset = data['points']
print(pts_subset.shape)

#print(images_subset.shape, pts_subset.shape)

test_data = np.load('resources/test_images.npz', allow_pickle=True)
test_images = test_data['images']
print(test_images.shape)

example_data = np.load('resources/examples.npz', allow_pickle=True)
example_images = example_data['images']
print(example_images.shape)

#visualises points data
def visualise_pts(img, pts):
  plt.imshow(img)
  plt.plot(pts[:, 0], pts[:, 1], '+b', ms=7)
  plt.show()
#testing visualisation

for i in range(3):
  idx = np.random.randint(0, images.shape[0])
  visualise_pts(images[idx, ...], pts[idx, ...])

for i in range(3):
  idx = np.random.randint(0, images_subset.shape[0])
  visualise_pts(images_subset[idx, ...], pts_subset[idx, ...])
  

#extracts subset of points from full set
def extract_subset_of_points(pts):
  indices = (20, 29, 16, 32, 38)
  if len(pts.shape) == 3:
    return pts[:, indices, :]
  elif len(pts.shape) == 2:
    return pts[indices, :]
  
  #test above
'''
for i in range(3):
  idx = np.random.randint(0, images.shape[0])
  visualise_pts(images[idx, ...], extract_subset_of_points(pts[idx, ...]))
  '''

#calculate euclidean distance
def euclid_dist(pred_pts, gt_pts):
  """
  Calculate the euclidean distance between pairs of points
  :param pred_pts: The predicted points
  :param gt_pts: The ground truth points
  :return: An array of shape (no_points,) containing the distance of each predicted point from the ground truth
  """
  pred_pts = np.reshape(pred_pts, (-1, 2))
  gt_pts = np.reshape(gt_pts, (-1, 2))
  return np.sqrt(np.sum(np.square(pred_pts - gt_pts), axis=-1))

#gets the mean sqrd error of one image
def mean_sqrd_error(error):
  total = 0
  for num in error:
    total += num**2
  mean = math.sqrt(total)
  return mean

#save as csv
def save_as_csv(points, location = '.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==44*2, 'wrong number of points provided. There should be 44 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

def gauss_blur_imgs(imgs): #this function takes an array of images and returns them all with gaussian blur
  blurd_imgs = []
  for img in imgs:
    blurd = np.uint8(np.mean(img, axis=2)) #gray_scale
    blurd = cv.GaussianBlur(blurd, (35,35), 0) #blur
    blurd_imgs.append(blurd)
  
  return np.array(blurd_imgs)

#takes the image and dimension of grid OR coords of features as arguments
def get_kps(img, grid_size=None, coords=None):
  if grid_size is not None: #for generating kps on regular grid
    kps = [cv.KeyPoint(x, y, img.shape[0]//grid_size) for x in range((img.shape[0]//grid_size)//2, img.shape[0], img.shape[0]//grid_size) for y in range((img.shape[0]//grid_size)//2, img.shape[1], img.shape[0]//grid_size)]
  
  elif coords is not None and coords.shape[0]==5: #for generating kps based off of predetermined coords
    #the size of each kp is the distance between corner of left eye and right corner of mouth
    kps = []
    for i in range(coords.shape[0]):
      kps.append(cv.KeyPoint(coords[i, 0], coords[i, 1], np.sqrt(np.sum(np.square(coords[0, ...] - coords[-1, ...])))))
    #adds a sixth descriptor to be around the chin area
    kps.append(cv.KeyPoint(coords[2, 0], coords[2, 1]+np.sqrt(np.sum(np.square(coords[0, ...] - coords[-1, ...]))), np.sqrt(np.sum(np.square(coords[0, ...] - coords[-1, ...])))))
  
  return kps

def get_full_kps(imgs, grid=None, coords=None):
  kps = []
  if coords is not None: #generates kps with predetermined coordinates
    for i,img in enumerate(imgs):
      kps.append(get_kps(img, grid, coords[i, ...]))
  else:
    for i,img in enumerate(imgs): #generates kps on regular grid
      kps.append(get_kps(img, grid))
  return kps

def get_full_vector(des): #for getting the large concatenated vector
  vector = np.array(des).reshape(-1) #makes the array of descriptors 1D
  return vector

def get_input_feats(imgs, kps_set):
  vecs = [] #defines a list to hold all the descriptor vectors for dataset
  for i, kps in enumerate(kps_set): #computes the descriptors for every set of kps and adds to list
    kps_set[i], des = sift.compute(imgs[i], kps)
    vecs.append(get_full_vector(des))
  vecs = np.asarray(vecs) #converts to numpy array
  return vecs


blurd_train_imgs_full = gauss_blur_imgs(images)#blurs the training images
blurd_train_imgs, blurd_valid_imgs = np.split(blurd_train_imgs_full, [(blurd_train_imgs_full.shape[0]*8)//10])#separates validation set from the full training set
blurd_sub_imgs = gauss_blur_imgs(images_subset) #blurs the images from the dataset only marked with the subset of 5 points
blurd_test_imgs = gauss_blur_imgs(test_images) #blurs the test images
blurd_exam_imgs = gauss_blur_imgs(example_images) #blurs the example images

sub_sub_kps = get_full_kps(blurd_sub_imgs, grid=5) #gets the kps for the images from the subset dataset
sub_train_kps = get_full_kps(blurd_train_imgs, grid=5) #gets the kps for the training images
sub_exam_kps = get_full_kps(blurd_exam_imgs, grid=5) #gets the kps for the example images
sub_valid_kps = get_full_kps(blurd_valid_imgs, grid=5) #gets the kps for the validation images
sub_test_kps = get_full_kps(blurd_test_imgs, grid=5) #gets the kps for the test set


sub_train_vecs = get_input_feats(np.concatenate((blurd_train_imgs,blurd_sub_imgs), axis=0), sub_train_kps+sub_sub_kps) #gets the vectors for the subset and training images together
sub_valid_vecs = get_input_feats(blurd_valid_imgs, sub_valid_kps) #gets the vectors for the validation set
sub_exam_vecs = get_input_feats(blurd_exam_imgs, sub_exam_kps) #gets the vectors for the example set
sub_test_vecs = get_input_feats(blurd_test_imgs, sub_test_kps) #gets the vectors for the example set

#gathers subset of 5 ground truth points for training and validation data
sub_pts = np.concatenate((extract_subset_of_points(pts[:blurd_train_imgs.shape[0]]), pts_subset), axis=0)#gets the subset of 5 points from every image in the dataset and concatenates with the subset dataset
sub_pts = sub_pts.reshape((blurd_train_imgs.shape[0]+blurd_sub_imgs.shape[0], 10)) #reshapes to remove the 3rd axis to be used in the regressor
sub_valid_pts = extract_subset_of_points(pts[blurd_train_imgs.shape[0]:]) 

#calls and trains primary regressor
sub_regressor = RandomForestRegressor(max_depth=10)
sub_regressor.fit(sub_train_vecs, sub_pts)

#gets results for different sets of data
sub_train_res = sub_regressor.predict(sub_train_vecs)
sub_valid_res = sub_regressor.predict(sub_valid_vecs)
sub_exam_res = sub_regressor.predict(sub_exam_vecs)
sub_test_res = sub_regressor.predict(sub_test_vecs)

#separates the results of the images with the full set of GT points to train secondary regressor
sub_train_res, sub_sub_res = np.split(sub_train_res, [blurd_train_imgs.shape[0]]) 

#reshapes to work in kp generation
sub_train_res = sub_train_res.reshape(sub_train_res.shape[0], 5, 2)
sub_sub_res = sub_sub_res.reshape(sub_sub_res.shape[0], 5, 2)
sub_valid_res = sub_valid_res.reshape(sub_valid_res.shape[0], 5, 2)
sub_exam_res = sub_exam_res.reshape(sub_exam_res.shape[0], 5, 2)
sub_test_res = sub_test_res.reshape(sub_test_res.shape[0], 5, 2)

#gathers set of 6 kps for secondary regressor
ult_train_kps = get_full_kps(blurd_train_imgs, coords=sub_train_res)
ult_valid_kps = get_full_kps(blurd_valid_imgs, coords=sub_valid_res)
ult_exam_kps = get_full_kps(blurd_exam_imgs, coords=sub_exam_res)
ult_test_kps = get_full_kps(blurd_test_imgs, coords=sub_test_res)

#gets the vector descriptions for secondary regressor
ult_train_vecs = get_input_feats(blurd_train_imgs, ult_train_kps)
ult_valid_vecs = get_input_feats(blurd_valid_imgs, ult_valid_kps)
ult_exam_vecs = get_input_feats(blurd_exam_imgs, ult_exam_kps)
ult_test_vecs = get_input_feats(blurd_test_imgs, ult_test_kps)

#gathers GT points for training and validation set
train_pts, valid_pts = np.split(pts, [blurd_train_imgs.shape[0]]) 
train_pts = train_pts.reshape(train_pts.shape[0], 88) #to work in fitting function

#calls and trains secondary regressor
ult_regressor = RandomForestRegressor(max_depth=10)
ult_regressor.fit(ult_train_vecs, train_pts)

#gathers predictions from secondary regressor
ult_train_res = ult_regressor.predict(ult_train_vecs)
ult_valid_res = ult_regressor.predict(ult_valid_vecs)
ult_exam_res = ult_regressor.predict(ult_exam_vecs)
ult_test_res = ult_regressor.predict(ult_test_vecs)

#reshapes results for display and testing
ult_train_res = ult_train_res.reshape(ult_train_res.shape[0], 44, 2)
ult_valid_res = ult_valid_res.reshape(ult_valid_res.shape[0], 44, 2)
ult_exam_res = ult_exam_res.reshape(ult_exam_res.shape[0], 44, 2)
ult_test_res = ult_test_res.reshape(ult_test_res.shape[0], 44, 2)

#reshapes points arrays for display/testing
sub_pts = sub_pts.reshape(sub_pts.shape[0], 5, 2)
train_pts = train_pts.reshape(train_pts.shape[0], 44, 2)

#concats list of original images used to train primary regressor
sub_imgs = np.concatenate((images[:blurd_train_imgs.shape[0]], images_subset), axis=0)

#defines list to hold the error of training and validation data
sub_train_errors = []
sub_valid_errors = []
ult_train_errors = []
ult_valid_errors = []

#separates original training images into training and validation set
train_imgs, valid_imgs = np.split(images, [blurd_train_imgs.shape[0]])

#displays 3 examples of training images with GT and then predicted points from primary regressor
for i in range(3):
  idx = np.random.randint(0, sub_imgs.shape[0])
  visualise_pts(sub_imgs[idx, ...], sub_pts[idx, ...])
  visualise_pts(sub_imgs[idx, ...], np.concatenate((sub_train_res, sub_sub_res), axis=0)[idx, ...])

#displays 3 examples of valid images with GT and then predicted points
for i in range(3):
  idx = np.random.randint(0, valid_imgs.shape[0])
  visualise_pts(valid_imgs[idx, ...], sub_valid_pts[idx, ...])
  visualise_pts(valid_imgs[idx, ...], sub_valid_res[idx, ...])

#finds mean squared error for every set of predicted points for training set
for i in range(sub_train_res.shape[0]):
  euc = euclid_dist(np.concatenate((sub_train_res, sub_sub_res), axis=0)[i, ...], sub_pts[i, ...])
  mean = mean_sqrd_error(euc)
  sub_train_errors.append(mean)

#finds mean squared error for every set of predicted points for validation set
for i in range(sub_valid_res.shape[0]):
  euc = euclid_dist(sub_valid_res[i, ...], sub_valid_pts[i, ...])
  mean = mean_sqrd_error(euc)
  sub_valid_errors.append(mean)

#plots the error from the primary predicted points for each image in training and validation set in red and blue respectively
plt.scatter(range(sub_train_res.shape[0]), sub_train_errors, c='b')
plt.scatter(range(sub_train_res.shape[0], sub_train_res.shape[0]+sub_valid_res.shape[0]), sub_valid_errors, c='r')
plt.show()

#displays 3 examples of training images with GT and then secondary predicted points
for i in range(3):
  idx = np.random.randint(0, train_imgs.shape[0])
  visualise_pts(train_imgs[idx, ...], train_pts[idx, ...])
  visualise_pts(train_imgs[idx, ...], ult_train_res[idx, ...])

#displays 3 examples of valid images with GT and then secondary predicted points
for i in range(3):
  idx = np.random.randint(0, valid_imgs.shape[0])
  visualise_pts(valid_imgs[i, ...], valid_pts[idx, ...])
  visualise_pts(valid_imgs[idx, ...], ult_valid_res[idx, ...])

#finds mean squared error for every set of predicted points for training set
for i in range(ult_train_res.shape[0]):
  euc = euclid_dist(ult_train_res[i, ...], train_pts[i, ...])
  mean = mean_sqrd_error(euc)
  ult_train_errors.append(mean)

#finds mean squared error for every set of predicted points for validation set
for i in range(ult_valid_res.shape[0]):
  euc = euclid_dist(ult_valid_res[i, ...], valid_pts[i, ...])
  mean = mean_sqrd_error(euc)
  ult_valid_errors.append(mean)

#plots the error for each image in training and validation set in red and blue respectively
plt.scatter(range(ult_train_res.shape[0]), ult_train_errors, c='b')
plt.scatter(range(ult_train_res.shape[0], ult_train_res.shape[0]+ult_valid_res.shape[0]), ult_valid_errors, c='r')
plt.show()

save_as_csv(ult_test_res)