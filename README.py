
# coding: utf-8

# # Self-Driving Car Engineer Nanodegree 
# ## Project IV : Advanced Lane Lines
# 
# 
# [![Udacity - Self-Driving Car NanoDegree](https://s3.amazonaws.com/udacity-sdc/github/shield-carnd.svg)](http://www.udacity.com/drive)
# 
# ### Overview
# 
# In this project, your goal is to write a software pipeline to identify the lane boundaries in a video, but the main output or product we want you to create is a detailed writeup of the project. Check out the writeup template for this project and use it as a starting point for creating your own writeup.
# 
# 
# ### The Project
# The goals / steps of this project are the following:
# 
# - Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
# 
# - Apply a distortion correction to raw images.
# 
# - Use color transforms, gradients, etc., to create a thresholded binary image.
# 
# - Apply a perspective transform to rectify binary image ("birds-eye view").
# 
# - Detect lane pixels and fit to find the lane boundary.
# 
# - Determine the curvature of the lane and vehicle position with respect to center.
# 
# - Warp the detected lane boundaries back onto the original image.
# 
# - Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.
# 
# The images for camera calibration are stored in the folder called camera_cal. The images in test_images are for testing your pipeline on single frames. To help the reviewer examine your work, please save examples of the output from each stage of your pipeline in the folder called ouput_images, and include a description in your writeup for the project of what each image shows. The video called project_video.mp4 is the video your pipeline should work well on.
# 
# The challenge_video.mp4 video is an extra (and optional) challenge for you if you want to test your pipeline under somewhat trickier conditions. The harder_challenge.mp4 video is another optional challenge and is brutal!
# 
# If you're feeling ambitious (again, totally optional though), don't stop there! We encourage you to go out and take video of your own, calibrate your camera and show us how you would implement this project from scratch!

# In[1]:

import os
import scipy
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pickle
from collections import deque
import glob
import imageio
imageio.plugins.ffmpeg.download()
# Import everything needed to edit/save/watch video clips
from moviepy.editor import VideoFileClip
from IPython.display import HTML, YouTubeVideo
get_ipython().magic('matplotlib inline')


# In[2]:

camera_calibration_images = glob.glob('camera_cal/calibration*.jpg')

test_images = glob.glob('test_images/*.jpg')

output_images_dir = 'output_images/'

chessboard_images = [mpimg.imread(f) for f in glob.glob('camera_cal/calibration*.jpg')]

test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]


# # 1. Camera Calibration

# Camera calibration estimates the camera parameters (camera matrix and distortion coefficients) using the calibration chessboard images to correct for lens distortion, measure the size of an object in world units, or determine the location of the camera in the scene and undistort the test calibration images. The code for distortion correction is contained in IPython notebook.
# 
# We start by preparing "object points", which are (x, y, z) coordinates of the chessboard corners in the world (assuming coordinates such that z=0). Thus, objp is just a replicated array of coordinates, and objpoints will be appended with a copy of it every time all chessboard corners are successfully detected  in a test image. imgpoints will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.

# In[3]:

def find_corner(chessboard_images, nx=9, ny=6):
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for image in chessboard_images:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
            
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        return objpoints, imgpoints


# In[4]:

def calibrate_camera(chessboard_images):
    img = chessboard_images[0]
    h, w = img.shape[:2]
    image_size = (w, h)
        
    objpoints, imgpoints = find_corner(chessboard_images)
    # Do camera calibration given object points and image points
    ret, camera_matrix, distortion_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, 
                                                                             image_size, None,None)
    return camera_matrix, distortion_coeff 


# In[5]:

def display_corners(chessboard_images, nx=9, ny=6):
    corner_images = []
    for image in chessboard_images:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Find the chessboard corners
        ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
        img = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
        corner_images.append(img)
        
    fig, axes = plt.subplots(5,4,figsize=(20, 10))
    #fig.subplots_adjust(hspace=0.2, wspace=0.05)
    fig.tight_layout()
    
    for i, ax in enumerate(axes.flat):
        ax.imshow(corner_images[i])
        xlabel = "Finding corner of chessboard {0}".format(i)
        ax.set_xlabel(xlabel)
        ax.set_xticks([])
        ax.set_yticks([])   
        #plt.imshow(img)
        #plt.show()    


# In[6]:

#display_corners(chessboard_images)


# # 2. Pipeline (single images)
# 
# ### 2.1 Distortion-corrected image
# objpoints and imgpoints are then used to compute the camera matrix and distortion coefficients using the `cv2.calibrateCamera()` function. The distortion correction is applied to the distorted images in test_images folder using the `cv2.undistort()` function. The results of finding corners and undistortion of chessboard images are shown below.
# 
# The results of distortion correction are shown below. 

# In[7]:

camera_matrix, distortion_coeff = calibrate_camera(chessboard_images)


# In[8]:

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = 'camera_dist_pickle.p'
print('Saving data to pickle file...')
try:
    with open(dist_pickle, 'wb') as pfile:
        pickle.dump(
            {
                "camera_matrix" : camera_matrix,
                "distortion_coeff" : distortion_coeff
            },
            pfile, pickle.HIGHEST_PROTOCOL)
except Exception as e:
    print('Unable to save data to', dist_pickle, ':', e)
    raise

print('Data cached in pickle file.')


# In[9]:

# load pickled distortion matrix
with open('camera_dist_pickle.p', mode='rb') as f:
    dist_pickle = pickle.load(f)
    camera_matrix = dist_pickle["camera_matrix"]
    distortion_coeff = dist_pickle["distortion_coeff"]


# In[10]:

def undistort(image):
        """img: original RGB img to be undistorted
        Return undistorted image.
        """
        undistort_image = cv2.undistort(image, camera_matrix, distortion_coeff, None, camera_matrix)
        
        return undistort_image


# In[11]:

fig, axes = plt.subplots(len(test_images),2, figsize=(8*2, 4*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
#fig.tight_layout()

for i , (image, ax) in enumerate(zip(test_images, axes)):
    undistorted_image = undistort(image)

    ax[0].imshow(image)
    ax[0].set_axis_off()
    xlabel0 = "Original image{0}".format(i)
    ax[0].set_title(xlabel0 , fontsize=30)
    ax[1].imshow(undistorted_image)
    ax[1].set_axis_off()
    xlabel1 = "Undistorted image {0}".format(i)
    ax[1].set_title(xlabel1, fontsize=30)
    ax[1].set_xticks([])
    ax[1].set_yticks([]) 
    
plt.show()
#plt.savefig('./output_images/undistort_output.png')


# In[12]:

fig, axes = plt.subplots(len(chessboard_images),2, figsize=(20, 10*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.tight_layout()
fig.subplots_adjust(hspace=0.2, wspace=0.05)
#fig.tight_layout()

for i , (image, ax) in enumerate(zip(chessboard_images, axes)):
    undistorted_image = undistort(image)

    ax[0].imshow(image)
    ax[0].set_axis_off()
    xlabel0 = "Original image{0}".format(i)
    ax[0].set_title(xlabel0 , fontsize=10)
    ax[1].imshow(undistorted_image)
    ax[1].set_axis_off()
    xlabel1 = "Undistorted image {0}".format(i)
    ax[1].set_title(xlabel1, fontsize=10)
    ax[1].set_xticks([])
    ax[1].set_yticks([]) 
    
#plt.show()


# ### 2.2 Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image
# 
# There are two steps implemented for line detection:
# 
# - Detection of lines from undistorted images by a combination:
#    - sobel of x on a gray image from HLS channel - detecting lines with horizontal gradients
#    
#    - two sobel of directions, $arctan({\frac{sobely}{sobelx}})$, from hls channel - as left and right lines within a certain angel ranges
# 
#    - combination of the above three - lines_with_gradx AND (left_line OR right_line).
# 
# For color thresholding , the L and S channel of HLS images are specially good at detecting bright lines. The s channel for a gradient filter along x and saturation threshold, as well as the l channel for a luminosity threshold filter. A combination of these filters is used in the `binarize_pipeline()` function. This is implemented in `binarize_pipeline()` function 

# In[13]:

def binarize_pipeline(img, s_thresh=(120, 255), sx_thresh=(20, 255),  h_thresh=(200, 255), l_thresh=(40,255)):
    #s_thresh=(170, 255), sx_thresh=(80, 100),
    
    img = np.copy(img)
    
    # Convert to HLS color space and separate the V channel
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(np.float)
    h_channel = hls[:,:,0]
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    
    # Sobel x
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 0) # Take the derivative in x
    abs_sobelx = np.absolute(sobelx) # Absolute x derivative to accentuate lines away from horizontal
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
    # Threshold x gradient
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # Threshold color channel
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    '''
    # Threshold color channel
    h_binary = np.zeros_like(h_channel)
    h_binary[(s_channel >= h_thresh[0]) & (s_channel <= h_thresh[1])] = 1
    # Stack each channel
    # Note color_binary[:, :, 0] is all 0s, effectively an all black image. It might
    # be beneficial to replace this channel with something else.
    #color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, s_binary))
    color_binary = np.dstack(( np.zeros_like(sxbinary), sxbinary, h_binary))

    return color_binary
    '''

    # Threshold lightness
    l_binary = np.zeros_like(l_channel)
    l_binary[(l_channel >= l_thresh[0]) & (l_channel <= l_thresh[1])] = 1
    
    channels = 255*np.dstack(( l_binary, sxbinary, s_binary)).astype('uint8')        
    binary = np.zeros_like(sxbinary)
    binary[((l_binary == 1) & (s_binary == 1) | (sxbinary==1))] = 1
    binary = 255*np.dstack((binary,binary,binary)).astype('uint8')   
    
    return  binary,channels
    


# In[14]:

fig, axes = plt.subplots(len(test_images),2, figsize=(8*2, 4*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.tight_layout()

for i , (image, ax) in enumerate(zip(test_images, axes)):
    binary, channels = binarize_pipeline(image)
    ax[0].imshow(binary)
    ax[0].set_axis_off()
    xlabel0 = "Binary {0}".format(i)
    ax[0].set_title(xlabel0 , fontsize=30)
    ax[1].imshow(channels)
    ax[1].set_axis_off()
    xlabel1 = "Channel {0}".format(i)
    ax[1].set_title(xlabel1, fontsize=30)
    ax[1].set_xticks([])
    ax[1].set_yticks([]) 
    
plt.show()
#plt.savefig('./output_images/binarize_pipeline.png')


# ### 2.3 Perspective transform
# 
# The perspective transform to and from "bird's eye" perspective is implemented in `warp()` function. 
# 
# - The `warp()` function takes as input an color image (img), as well as the bird_view boolean paramter.
# 
# - After esimating the transform matrix, I can transform any new image (original RGB image or simply its binary line image) to a bird-eye view. 
# 
# - The parameters src and  dst of the transform are hardcoded in the function as follows:
# 
# 
#     'Source' = {[(190, 720), 
#                 (589, 457), 
#                 (698, 457), 
#                 (1145,720)]}
#                 
#     'Destination' = {[(340, 720), 
#                      (340, 0), 
#                      (995, 0), 
#                      (995, 720)]}	
# 
# 
# 	
# 	
# 	
# 	

# In[15]:

def warp(img, nx=9, ny=6, bird_view=True):
    # define 4 destination points dst = np.float32([[,],[,],[,],[,]])
    # use cv2.getPerspectiveTransform() to get M, the transform matrix
    # use cv2.warpPerspective() to warp your image to a top-down view
            
    img_size = (img.shape[1], img.shape[0])

    corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])
    new_top_left=np.array([corners[0,0],0])
    new_top_right=np.array([corners[3,0],0])
    offset=[150,0]
    
    
    src = np.float32([corners[0],corners[1],corners[2],corners[3]])
    dst = np.float32([corners[0]+offset,new_top_left+offset,new_top_right-offset ,corners[3]-offset])
    '''
    offset = 100 # offset for dst points
    img_size = (gray.shape[1], gray.shape[0])
    # For source points I'm grabbing the outer four detected corners
    src = np.float32([corners[0], corners[nx-1], corners[-1], corners[-nx]])
    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result 
    # again, not exact, but close enough for our purposes
    dst = np.float32([[offset, offset], [img_size[0]-offset, offset], 
                                     [img_size[0]-offset, img_size[1]-offset], 
                                     [offset, img_size[1]-offset]])
               
    '''
                
    if bird_view:
        M = cv2.getPerspectiveTransform(src, dst)
    else:
        #Compute the inverse perspective transform:
        M = cv2.getPerspectiveTransform(dst, src)
            
    #Warp an image using 
    warped = cv2.warpPerspective(img, M, img_size, flags=cv2.INTER_LINEAR)
 
    return warped, M


# In[16]:

test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]
corners = np.float32([[190,720],[589,457],[698,457],[1145,720]])

corner_tuples=[]
for ind,c in enumerate(corners):
    corner_tuples.append(tuple(corners[ind]))


fig, axes = plt.subplots(len(test_images),2, figsize=(8*2, 4*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.tight_layout()

for i , (image, ax) in enumerate(zip(test_images, axes)):
    image = undistort(image)
    cv2.line(image, corner_tuples[0], corner_tuples[1], color=[255,0,0], thickness=2)
    cv2.line(image, corner_tuples[1], corner_tuples[2], color=[255,0,0], thickness=2)
    cv2.line(image, corner_tuples[2], corner_tuples[3], color=[255,0,0], thickness=2)
    cv2.line(image, corner_tuples[3], corner_tuples[0], color=[255,0,0], thickness=2)    
    warped, _= warp(image, bird_view=True)
    ax[0].imshow(image)
    ax[0].set_axis_off()
    xlabel0 = "Original {0}".format(i)
    ax[0].set_title(xlabel0 , fontsize=30)
    ax[1].imshow(warped)
    ax[1].set_axis_off()
    xlabel1 = "Warped {0}".format(i)
    ax[1].set_title(xlabel1, fontsize=30)
    ax[1].set_xticks([])
    ax[1].set_yticks([]) 
    
plt.show()
#plt.savefig('./output_images/warped_original.png')


# In[17]:

def region_of_interest(img):
    """
    Applies an image mask.
    
    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """    
    shape = img.shape
    vertices = np.array([[(0,0),(shape[1],0),(shape[1],0),(6*shape[1]/7,shape[0]),
                      (shape[1]/7,shape[0]), (0,0)]],dtype=np.int32)

    mask = np.zeros_like(img)   
    
    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255
        
    #filling pixels inside the polygon defined by "vertices" with the fill color    
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    
    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image



def warp_pipeline(img):  
    img = undistort(img)
    warped,_ = warp(img)
    warp_roi = region_of_interest(warped)
    return warp_roi


def warp_binary_pipeline(img): 
    img = undistort(img)
    binary,_ = binarize_pipeline(img)
    binary_warped,_ = warp(binary)
    binary_warped_roi = region_of_interest(binary_warped)
    return binary_warped_roi
    


# To reduce artefacts at the bottom of the image, a region of interest is implemented to act on the warped image  This region is defined through the function `region_of_interest()` and is tested using the wrappers `warp_pipeline(img)` and `warp_binary_pipeline(img)`. The result is shown below.

# In[18]:

test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]

fig, axes = plt.subplots(len(test_images),2, figsize=(8*2, 4*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.tight_layout()

for i , (image, ax) in enumerate(zip(test_images, axes)):
    #cv2.line(image, corner_tuples[0], corner_tuples[1], color=[255,0,0], thickness=2)
    #cv2.line(image, corner_tuples[1], corner_tuples[2], color=[255,0,0], thickness=2)
    #cv2.line(image, corner_tuples[2], corner_tuples[3], color=[255,0,0], thickness=2)
    #cv2.line(image, corner_tuples[3], corner_tuples[0], color=[255,0,0], thickness=2)
    warped_roi = warp_pipeline(image)
    binary_warped_roi = warp_binary_pipeline(image)
    binary
    ax[0].imshow(warped_roi)
    ax[0].set_axis_off()
    xlabel0 = "Warp ROI {0}".format(i)
    ax[0].set_title(xlabel0 , fontsize=30)
    ax[1].imshow(binary_warped_roi)
    ax[1].set_axis_off()
    xlabel1 = "Warp binary ROI {0}".format(i)
    ax[1].set_title(xlabel1, fontsize=30)
    ax[1].set_xticks([])
    ax[1].set_yticks([]) 
    
plt.show()
#plt.savefig('./output_images/warp_roi.png')


# ### 2.4  Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?
# 
# The same techniques can be used to detect the lane pixels in the warped images. 
# 
#  - The function `find_peaks(img,thresh)` takes the bottom half of a binarized and warped lane image to compute a histogram of detected pixel values. The result is smoothened using a gaussian filter and peaks are subsequently detected using. The function returns the x values of the peaks larger than thresh as well as the smoothened curve.
# 
#  - The next function `get_next_window(img,center_point,width)` which takes an binary (3 channel) image and computes the average x value center of all detected pixels in a window centered at center_point of width width. It returns a masked copy of img a well as center.
# 
#  - The function `lane_from_window(binary,center_point,width)` slices a binary image horizontally in 6 zones and applies get_next_window to each of the zones. The center_point of each zone is chosen to be the center value of the previous zone. Thereby subsequent windows follow the lane line pixels if the road bends. The function returns a masked image of a single lane line seeded at center_point. Given a binary image left_binary of a lane line candidate all properties of the line are determined within an instance of a Line class.
# 
# - The Line.update(img) method takes a binary input image of a lane line candidate, fits a second order polynomial to the provided data and computes other metrics. Sanity checks are performed and successful detections are pushed into a FIFO que of max length n. Each time a new line is detected all metrics are updated. If no line is detected the oldest result is dropped until the queue is empty and peaks need to be searched for from scratch.
# 
# - A fit to the current lane candidate is saved in the Line.current_fit_xvals attribute, together with the corresponding coefficients. 
# 
# The results of the lane pixels in the bird-eye view are shown below. We can see there are some noises in the final lane images, which need to be removed before parameter estimation.

# In[19]:

def find_peaks(img,thresh):
    img_half=img[img.shape[0]/2:,:,0]
    data = np.sum(img_half, axis=0)
    filtered = scipy.ndimage.filters.gaussian_filter1d(data,20)
    xs = np.arange(len(filtered))
    peak_ind = signal.find_peaks_cwt(filtered, np.arange(20,300))
    peaks = np.array(peak_ind)
    peaks = peaks[filtered[peak_ind]>thresh]
    return peaks,filtered


def get_next_window(img,center_point,width):
    """
    input: img,center_point,width
        img: binary 3 channel image
        center_point: center of window
        width: width of window
    
    output: masked,center_point
        masked : a masked image of the same size. mask is a window centered at center_point
        center : the mean ofall pixels found within the window
    """
    
    ny,nx,_ = img.shape
    mask  = np.zeros_like(img)
    if (center_point <= width/2): center_point = width/2
    if (center_point >= nx-width/2): center_point = nx-width/2
    
    left  = center_point - width/2
    right = center_point + width/2
    
    vertices = np.array([[(left,0),(left,ny), (right,ny),(right,0)]], dtype=np.int32)
    ignore_mask_color=(255,255,255)
    cv2.fillPoly(mask, vertices, ignore_mask_color)
    masked = cv2.bitwise_and(mask,img)

    hist = np.sum(masked[:,:,0],axis=0)
    if max(hist>10000):
        center = np.argmax(hist)
    else:
        center = center_point
        
    return masked,center

def lane_from_window(binary,center_point,width):
    n_zones=6
    ny,nx,nc = binary.shape
    zones = binary.reshape(n_zones,-1,nx,nc)
    zones = zones[::-1] # start from the bottom slice
    window,center = get_next_window(zones[0],center_point,width)
    
    for zone in zones[1:]:
        next_window,center = get_next_window(zone,center,width)
        window = np.vstack((next_window,window))
    
    return window


# In[20]:

test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]

fig, axes = plt.subplots(len(test_images),2, figsize=(8*2, 4*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.tight_layout()

for i , (image, ax) in enumerate(zip(test_images, axes)):
    #warped_roi = warp_pipeline(image)
    binary_warped = warp_binary_pipeline(image)
    left_binary = lane_from_window(binary_warped,380,300)
    right_binary = lane_from_window(binary_warped,1000,300)
    binary
    ax[0].imshow(left_binary)
    ax[0].set_axis_off()
    xlabel0 = "Left line {0}".format(i)
    ax[0].set_title(xlabel0 , fontsize=30)
    ax[1].imshow(right_binary)
    ax[1].set_axis_off()
    xlabel1 = "Right line {0}".format(i)
    ax[1].set_title(xlabel1, fontsize=30)
    ax[1].set_xticks([])
    ax[1].set_yticks([]) 
    
plt.show()
#plt.savefig('./output_images/left_right_lines.png')


# In[21]:

'''
import numpy as np
import cv2
import matplotlib.pyplot as plt

binary_warped = binary_warped_roi
# Assuming you have created a warped binary image called "binary_warped"
# Take a histogram of the bottom half of the image
histogram = np.sum(binary_warped[binary_warped.shape[0]/2:,:], axis=0)
# Create an output image to draw on and  visualize the result
out_img = np.dstack((binary_warped, binary_warped, binary_warped))*255
# Find the peak of the left and right halves of the histogram
# These will be the starting point for the left and right lines
midpoint = np.int(histogram.shape[0]/2)
leftx_base = np.argmax(histogram[:midpoint])
rightx_base = np.argmax(histogram[midpoint:]) + midpoint

# Choose the number of sliding windows
nwindows = 9
# Set height of windows
window_height = np.int(binary_warped.shape[0]/nwindows)
# Identify the x and y positions of all nonzero pixels in the image
nonzero = binary_warped.nonzero()
nonzeroy = np.array(nonzero[0])
nonzerox = np.array(nonzero[1])
# Current positions to be updated for each window
leftx_current = leftx_base
rightx_current = rightx_base
# Set the width of the windows +/- margin
margin = 100
# Set minimum number of pixels found to recenter window
minpix = 50
# Create empty lists to receive left and right lane pixel indices
left_lane_inds = []
right_lane_inds = []

# Step through the windows one by one
for window in range(nwindows):
    # Identify window boundaries in x and y (and right and left)
    win_y_low = binary_warped.shape[0] - (window+1)*window_height
    win_y_high = binary_warped.shape[0] - window*window_height
    win_xleft_low = leftx_current - margin
    win_xleft_high = leftx_current + margin
    win_xright_low = rightx_current - margin
    win_xright_high = rightx_current + margin
    # Draw the windows on the visualization image
    cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),(0,255,0), 2) 
    cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),(0,255,0), 2) 
    # Identify the nonzero pixels in x and y within the window
    good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
    good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
    # Append these indices to the lists
    left_lane_inds.append(good_left_inds)
    right_lane_inds.append(good_right_inds)
    # If you found > minpix pixels, recenter next window on their mean position
    if len(good_left_inds) > minpix:
        leftx_current = np.int(np.mean(nonzerox[good_left_inds]))
    if len(good_right_inds) > minpix:        
        rightx_current = np.int(np.mean(nonzerox[good_right_inds]))

# Concatenate the arrays of indices
left_lane_inds = np.concatenate(left_lane_inds)
right_lane_inds = np.concatenate(right_lane_inds)

# Extract left and right line pixel positions
leftx = nonzerox[left_lane_inds]
lefty = nonzeroy[left_lane_inds] 
rightx = nonzerox[right_lane_inds]
righty = nonzeroy[right_lane_inds] 

# Fit a second order polynomial to each
left_fit = np.polyfit(lefty, leftx, 2)
right_fit = np.polyfit(righty, rightx, 2)
'''


# ### 2.5 Lane parameter estimation
# 
# we have the lane pixels in bird-eye view and meter-per-pixel for both x and y, estimating the curvature and the center offset is straightforward. The whole process is implemented in `Line.get_radius_of_curvature()` method:
# 
# 
# - After getting pixels for each lane, a 2nd order polynomial is fit for each lane, based on which the radius of curvature and center offset are caculated in `Line.get_radius_of_curvature()` method. The radius of curvature is computed upon calling the `Line.update()` method of Line Class. 
# 
# 
# -  For a second order polynomial $f(y)=A y^2 +B y + C$ the radius of curvature is given by $R = {{(1+(2 Ay +B)^2 )^{3/2}} \over {2|A|}}$. The mathematics involved is summarized in [this tutorial here](http://www.intmath.com/applications-differentiation/8-radius-curvature.php).
# 
# 
# - The distance from the center of the lane is computed in the Line.set_line_base_pos() method, which essentially measures the distance to each lane and computes the position assuming the lane has a given fixed width of 3.7m. 
# 
# 
# - The estimated 2nd polynoimal approximation of lanes are overlayed onto the image for furthure visual check.
# 
# 
# 
# 

# In[22]:

# Define a class to receive the characteristics of each line detection
class Line:
    def __init__(self,n=5):
        # length of queue to store data
        self.n = n
        #number of fits in buffer
        self.n_buffered = 0
        # was the line detected in the last iteration?
        self.detected = False  
        # x values of the last n fits of the line
        self.recent_xfitted = deque([],maxlen=n)
        #average x values of the fitted line over the last n iterations
        self.avgx = None
        # fit coeffs of the last n fits
        self.recent_fit_coeffs = deque([],maxlen=n)        
        #polynomial coefficients averaged over the last n iterations
        self.avg_fit_coeffs = None  
        # xvals of the most recent fit
        self.current_fit_xvals = [np.array([False])]  
        #polynomial coefficients for the most recent fit
        self.current_fit_coeffs = [np.array([False])]          
        #x values for detected line pixels
        self.allx = None  
        #y values for detected line pixels
        self.ally = None
        #y values for line fit
        self.fit_yvals = np.linspace(0, 100, num=101)*7.2  # always the same y-range as image
        #radius of curvature of the line in some units
        self.radius_of_curvature = None 
        # origin (pixels) of fitted line at the bottom of the image
        self.line_pos = None 
        #distance in meters of vehicle center from the line
        self.line_base_pos = None 
        #difference in fit coefficients between last and new fits
        self.diffs = np.array([0,0,0], dtype='float') 

    def set_current_fit_xvals(self):
        yvals = self.fit_yvals
        self.current_fit_xvals = self.current_fit_coeffs[0]*yvals**2 + self.current_fit_coeffs[1]*yvals + self.current_fit_coeffs[2]
        
    def add_data(self):
        self.recent_xfitted.appendleft(self.current_fit_xvals)
        self.recent_fit_coeffs.appendleft(self.current_fit_coeffs)
        assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
        self.n_buffered = len(self.recent_xfitted)
        
    def pop_data(self):        
        if self.n_buffered>0:
            self.recent_xfitted.pop()
            self.recent_fit_coeffs.pop()
            assert len(self.recent_xfitted)==len(self.recent_fit_coeffs)
            self.n_buffered = len(self.recent_xfitted)
        
        return self.n_buffered
        
    def set_avgx(self):
        fits = self.recent_xfitted
        if len(fits)>0:
            avg=0
            for fit in fits:
                avg +=np.array(fit)
            avg = avg / len(fits)
            self.avgx = avg
            
    def set_avgcoeffs(self):
        coeffs = self.recent_fit_coeffs
        if len(coeffs)>0:
            avg=0
            for coeff in coeffs:
                avg +=np.array(coeff)
            avg = avg / len(coeffs)
            self.avg_fit_coeffs = avg
    
    def set_allxy(self,lane_candidate):
        self.ally,self.allx = (lane_candidate[:,:,0]>254).nonzero()

    def set_current_fit_coeffs(self):
        self.current_fit_coeffs = np.polyfit(self.ally, self.allx, 2)
    
    def get_diffs(self):
        if self.n_buffered>0:
            self.diffs = self.current_fit_coeffs - self.avg_fit_coeffs
        else:
            self.diffs = np.array([0,0,0], dtype='float')                 
            
    def set_radius_of_curvature(self):
        # Define y-value where we want radius of curvature (choose bottom of the image)
        y_eval = max(self.fit_yvals)
        if self.avg_fit_coeffs is not None:
            self.radius_of_curvature = ((1 + (2*self.avg_fit_coeffs[0]*y_eval + self.avg_fit_coeffs[1])**2)**1.5)                              /np.absolute(2*self.avg_fit_coeffs[0])
                        
            
    def set_line_base_pos(self):
        y_eval = max(self.fit_yvals)
        self.line_pos = self.current_fit_coeffs[0]*y_eval**2                         +self.current_fit_coeffs[1]*y_eval                         + self.current_fit_coeffs[2]
        basepos = 640
        
        self.line_base_pos = (self.line_pos - basepos)*3.7/600.0 # 3.7 meters is about 600 pixels in the x direction

    # here come sanity checks of the computed metrics
    def accept_lane(self):
        flag = True
        maxdist = 2.8  # distance in meters from the lane
        if(abs(self.line_base_pos) > maxdist ):
            print('lane too far away')
            flag  = False        
        if(self.n_buffered > 0):
            relative_delta = self.diffs / self.avg_fit_coeffs
            # allow maximally this percentage of variation in the fit coefficients from frame to frame
            if not (abs(relative_delta)<np.array([0.7,0.5,0.15])).all():
                print('fit coeffs too far off [%]',relative_delta)
                flag=False
                
        return flag
    
    def update(self,lane):
        self.set_allxy(lane)
        self.set_current_fit_coeffs()
        self.set_current_fit_xvals()
        self.set_radius_of_curvature()
        self.set_line_base_pos()
        self.get_diffs()
        if self.accept_lane():
            self.detected=True
            self.add_data()
            self.set_avgx()
            self.set_avgcoeffs()            
        else:
            self.detected=False            
            self.pop_data()
            if self.n_buffered>0:
                self.set_avgx()
                self.set_avgcoeffs()
                    
        return self.detected,self.n_buffered
    
def get_binary_lane_image(img,line,window_center,width=300):
    if line.detected:
        window_center=line.line_pos
    else:
        peaks,filtered = find_peaks(img,thresh=3000)
        if len(peaks)!=2:
            print('Trouble ahead! '+ str(len(peaks)) +' lanes detected!')
            plt.imsave('troublesome_image.jpg',img)                        
            
        peak_ind = np.argmin(abs(peaks-window_center))
        peak  = peaks[peak_ind]
        window_center = peak
    
    lane_binary = lane_from_window(img,window_center,width)
    return lane_binary


# In[23]:

test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]

fig, axes = plt.subplots(len(test_images),1, figsize=(5, 3*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.tight_layout()

for i , (image, ax) in enumerate(zip(test_images, axes)):
    left=Line()
    right=Line()
    binary_warped = warp_binary_pipeline(image)
    left_binary = lane_from_window(binary_warped,380,300)
    right_binary = lane_from_window(binary_warped,1000,300)

    detected_l,n_buffered_left = left.update(left_binary)
    detected_r,n_buffered_right = right.update(right_binary)

    leftx = left.allx
    left_fitx = left.current_fit_xvals
    yvals_l = left.ally

    rightx = right.allx
    right_fitx = right.current_fit_xvals
    yvals_r = right.ally

    yvals = left.fit_yvals

    ax.plot(rightx, yvals_r, '.', color='red')
    ax.plot(right_fitx, yvals, color='green', linewidth=3)

    ax.plot(leftx, yvals_l, '.', color='red')
    ax.plot(left_fitx, yvals, color='green', linewidth=3)
    
plt.show()
#plt.savefig('./output_images/fitted_lines.png')


# ### 2.6 Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

# In[31]:

def project_lane_lines(img,left_fitx,right_fitx,yvals):
    
    # Create an image to draw the lines on
    color_warp = np.zeros_like(img).astype(np.uint8)

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
    pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(255,0,0), thickness=20)
    cv2.fillPoly(color_warp, np.int_([pts]), (0,255, 0))
    
    
        
    undist = undistort(img)  
    #sp = (550, 310) 
    #ep = (700, 460)
    #for i in range(4):
        #center = ((ep[0] + sp[0])/2 , )
        #cv2.rectangle(undist, (550, 310), (700, 460), (0,0,255), 4)
    unwarp,Minv = warp(img,bird_view=False)

    

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (img.shape[1], img.shape[0])) 
    # Combine the result with the original image
    result = cv2.addWeighted(undist, 1, newwarp, 0.3, 0)
    return result


# test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]
# img = test_images [0]
# 
# global left
# global right
# undist = undistort(img)
# binary,_  = binarize_pipeline(undist)
# warped,_  = warp(binary)
# warped_binary = region_of_interest(warped)
#     
# window_center_l = 340
# if left.detected:
#     window_center_l = left.line_pos        
# left_binary = get_binary_lane_image(warped_binary,left,window_center_l,width=300)
# 
# window_center_r = 940
# if right.detected:
#     window_center_r = right.line_pos        
# right_binary = get_binary_lane_image(warped_binary,right,window_center_r,width=300)
#     
# detected_l,n_buffered_left = left.update(left_binary)
# detected_r,n_buffered_right = right.update(right_binary)    
#     
# left_fitx = left.avgx
# right_fitx = right.avgx
# yvals = left.fit_yvals
# lane_width = 3.7
# 
# pts_left = np.array([np.transpose(np.vstack([left_fitx, yvals]))])
# pts_right = np.array([np.flipud(np.transpose(np.vstack([right_fitx, yvals])))])
# print(pts_left[0][:])
# print(pts_right[0][:])

# test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]
# img = test_images [0]
# x1 = int(pts_left[0][50][0])
# y1 = int(pts_left[0][50][1]) - 200
# x2 = int(pts_right[0][50][0])
# y2 = int(pts_right[0][50][1]) + 100
# 
# cv2.rectangle(img, (x1, y1), (x2, y2), (0,0,255), 4)
# plt.imshow(img)

# x, y = pts_left[0][40]
# x1, y1 = pts_right[0][40]
# print(x, y, x1, y1)

# In[37]:

def process_image(img):
    global left
    global right
    undist = undistort(img)
    binary,_  = binarize_pipeline(undist)
    warped,_  = warp(binary)
    warped_binary = region_of_interest(warped)
    
    window_center_l = 340
    if left.detected:
        window_center_l = left.line_pos        
    left_binary = get_binary_lane_image(warped_binary,left,window_center_l,width=300)
    
    window_center_r = 940
    if right.detected:
        window_center_r = right.line_pos        
    right_binary = get_binary_lane_image(warped_binary,right,window_center_r,width=300)
    
    detected_l,n_buffered_left = left.update(left_binary)
    detected_r,n_buffered_right = right.update(right_binary)    
    
    left_fitx = left.avgx
    right_fitx = right.avgx
    yvals = left.fit_yvals
    lane_width = 3.7
    #off_center = -100*round(0.5*(right.line_base_pos-lane_width/2) +  0.5*(abs(left.line_base_pos)-lane_width/2),2)
    
    #notice that line_pos is used instead of line_base_pos
    center_of_lane = (left.line_pos + right.line_pos)/2
    
    #Then the distance from the center in pixels
    distance_in_pixels = center_of_lane - 640

    #Then converting it to real space: 
    off_center = round(distance_in_pixels * 3.7 / 600.0, 2)
    
    result = project_lane_lines(img,left_fitx,right_fitx,yvals)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    str1 = str('distance from center: '+str(off_center)+'cm')
    cv2.putText(result,str1,(430,630), font, 1,(255,0,0),2,cv2.LINE_AA)
    if left.radius_of_curvature and right.radius_of_curvature:
        curvature = 0.5*(round(right.radius_of_curvature/1000,1) + round(left.radius_of_curvature/1000,1))
        str2 = str('radius of curvature: '+str(curvature)+'km')
        cv2.putText(result,str2,(430,670), font, 1,(255,0,0),2,cv2.LINE_AA)    
    
    return result


# In[38]:

test_images = [mpimg.imread(f) for f in glob.glob('./test_images/*.jpg')]

fig, axes = plt.subplots(len(test_images),2, figsize=(8*2, 4*len(test_images)))
fig.subplots_adjust(hspace=0.2, wspace=0.05)
fig.tight_layout()

for i , (image, ax) in enumerate(zip(test_images, axes)):
    left = Line()
    right = Line()
    result = process_image(image)
    result = process_image(image)

    ax[0].imshow(image)
    ax[0].set_axis_off()
    xlabel0 = "Original image {0}".format(i)
    ax[0].set_title(xlabel0 , fontsize=10)
    ax[1].imshow(result)
    ax[1].set_axis_off()
    xlabel1 = "Detected lane {0}".format(i)
    ax[1].set_title(xlabel1, fontsize=10)
    ax[1].set_xticks([])
    ax[1].set_yticks([]) 
    
plt.show()
#plt.savefig('./output_images/detected_lane.png')


# # 3. Pipeline for video
# 
# The processed project video can be found here.
# 
# The result on project_video.mp4 is shown below. The algorithm works did not work on the two challenge videos. I didn't go further to modify the code to work on these challenges. As mentioned above, I am not really convinced by the material in this project, so even it succeeds on the challenge videos, I have no confidence at all that it will work on new scenarios.

# In[39]:

output_dir= './output_images/'
clip_input_file = 'project_video.mp4'
clip_output_file = output_dir +'sample_' + clip_input_file
clip = VideoFileClip(clip_input_file).subclip(30, 40)
clip_output = clip.fl_image(process_image)
get_ipython().magic('time clip_output.write_videofile(clip_output_file, audio=False)')


# In[40]:

output_dir= './output_images/'
clip_input_file = 'project_video.mp4'
clip_output_file = output_dir +'processed_' + clip_input_file
clip = VideoFileClip(clip_input_file)
clip_output = clip.fl_image(process_image)
get_ipython().magic('time clip_output.write_videofile(clip_output_file, audio=False)')


# In[2]:

YouTubeVideo('Nuctmk4_eKE')


# In[ ]:

output_dir= './output_images/'
clip_input_file = 'harder_challenge_video.mp4'
clip_output_file = output_dir +'processed_' + clip_input_file
clip = VideoFileClip(clip_input_file).subclip(30, 40)
clip_output = clip.fl_image(process_image)
get_ipython().magic('time clip_output.write_videofile(clip_output_file, audio=False)')


# In[ ]:




# # 4. Discussion
# 
# ### 4.1 Problems/issues
# 
# This project is very challenging since I am new to computer vision. I think there are some difficulties I faced during this project:
# 
# - It's difficult to automatically detect the src and dst points for perspective transform as well as figure out the right size of sliding window to detect lane lines. Another problemis that how to effectively find the best combination of binary image and paired with its best threshold causes problematic.
# 
# 
# - It's hard to get the pipeline to be robust against shadows and at the same time capable of detecting yellow lane lines on white ground. Maybe building separate lane line detectors for yellow and white together with additional logic which line to choose will work.
# 
# 
# - The pipeline fails as soon as more (spurious) lines are on the same lane, as e.g. in the first challenge video.
# 
# 
# - I hope I can learn how to apply deep learning for lane lines detection. 

# In[ ]:



