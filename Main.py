import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
from scipy.signal import convolve2d, convolve
from sklearn.datasets import make_regression
from sklearn.ensemble import RandomForestRegressor
import pickle

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

#print(images_subset.shape, pts_subset.shape)

test_data = np.load('resources/test_images.npz', allow_pickle=True)
test_images = test_data['images']
print(test_images.shape)

example_data = np.load('resources/examples.npz', allow_pickle=True)
example_images = example_data['images']
#print(example_images.shape)

#visualises points data
def visualise_pts(img, pts):
  plt.imshow(img)
  plt.plot(pts[:, 0], pts[:, 1], '+r', ms=7)
  plt.show()

#testing visualisation
'''
for i in range(3):
  idx = np.random.randint(0, images.shape[0])
  visualise_pts(images[idx, ...], pts[idx, ...])

for i in range(3):
  idx = np.random.randint(0, images.shape[0])
  visualise_pts(images_subset[idx, ...], pts_subset[idx, ...])
  '''

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

#calculate euclidean distance (incomplete)
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

#save as csv [for when finished :)))))))]
def save_as_csv(points, location = '.'):
    """
    Save the points out as a .csv file
    :param points: numpy array of shape (no_test_images, no_points, 2) to be saved
    :param location: Directory to save results.csv in. Default to current working directory
    """
    assert points.shape[0]==554, 'wrong number of image points, should be 554 test images'
    assert np.prod(points.shape[1:])==44*2, 'wrong number of points provided. There should be 34 points with 2 values (x,y) per point'
    np.savetxt(location + '/results.csv', np.reshape(points, (points.shape[0], -1)), delimiter=',')

'''
def gauss_blur_img(img): #this function applies the gaussian blur to an image
  gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
  smooth = cv.GaussianBlur(gray, (101,101), 0)
  division = cv.divide(gray, smooth, scale=255)
  plt.imshow(division, cmap='gray', vmin=0, vmax=255)
  plt.show()
  return division'''

#make new list of blurred images

def gauss_blur_imgs(imgs): #this function takes an array of images and returns them all with gaussian blur
  blurd_imgs = []
  for img in imgs:
    blurd = np.uint8(np.mean(img, axis=2)) #gray_scale
    blurd = cv.GaussianBlur(blurd, (15,15), 0) #blur
    blurd_imgs.append(blurd)
  
  return blurd_imgs

#takes the image and dimension of grid OR coords of features as arguments
def get_kps(img, grid_size=None, coords=None):
  if grid_size is not None: #for generating kps on regular grid
    kps = [cv.KeyPoint(x, y, img.shape[0]//grid_size) for x in range((img.shape[0]//grid_size)//2, img.shape[0], img.shape[0]//grid_size) for y in range((img.shape[0]//grid_size)//2, img.shape[1], img.shape[0]//grid_size)]
 # elif coords is not None: #for generating kps based off of predetermined coords
  # kps = [cv.KeyPoint(x, y), ]
  return kps

#def get_feature_coords(pts):

def get_full_vector(des): #for getting the large concatenated vector
  vector = np.array(des).reshape(-1)
  return vector

def get_input_feats(imgs, kps_set):
  vecs = []
  for i, kps in enumerate(kps_set):
    kps_set[i], des = sift.compute(imgs[i], kps_set[i])
    vecs.append(get_full_vector(des))
  vecs = np.asarray(vecs)
  return vecs


#blurd_train_imgs = gauss_blur_imgs(images)#blurs the training images
blurd_test_imgs = gauss_blur_imgs(test_images)


#train_kps = []
#for img in blurd_train_imgs:
  #train_kps.append(get_kps(img, 5))

test_kps = []
for img in blurd_test_imgs:
  test_kps.append(get_kps(img, 5))



#train_kps[0], des = sift.compute(blurd_train_imgs[0], train_kps[0])
#img_kp = np.zeros_like(data['images'][0])
#cv.drawKeypoints(data['images'][0], train_kps[0], img_kp, flags=4)


#train_vecs = get_input_feats(blurd_train_imgs, train_kps)
#print(train_vecs.shape)
test_vecs = get_input_feats(blurd_test_imgs, test_kps)

pts = pts.reshape((1425, 88)) #to work inside fitting function
print(pts.shape)
#regressor = RandomForestRegressor(max_depth=5, random_state=0)
#regressor.fit(train_vecs, pts)
with open('first_regressor.dictionary', 'rb') as first_regressor:
  regressor = pickle.load(first_regressor)
#result_points = regressor.predict(test_vecs)

#result_points = result_points.reshape(554, 44, 2)
with open('first_test_results.dictionary', 'rb') as first_results:
  result_points = pickle.load(first_results)
for i in range(3):
  idx = np.random.randint(0, test_images.shape[0])
  print(idx)
  visualise_pts(test_images[idx, ...], result_points[idx, ...])



def augment_vec(data, poly_order):#allows for the data to be augmented to a given polynomial order
  return np.concatenate([np.power(data, p) for p in range(poly_order+1)], axis=1)



