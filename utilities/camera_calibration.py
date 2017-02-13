import numpy as np
import cv2
import matplotlib.pyplot as plt

from utilities import config

class CameraCalibration(object):
    """Calibrate camera by estimating the distortion matrix and coefficients."""
    def __int__(self):
        self.camera_matrix = None # Camera matrix
        self.distortion_coeff = None # Distortion coefficients
        
    def find_corner(self, chessboard_images, nx=9, ny=6):
        objp = np.zeros((ny*nx,3), np.float32)
        objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1,2)

        # Arrays to store object points and image points from all the images.
        objpoints = [] # 3d points in real world space
        imgpoints = [] # 2d points in image plane.

        # Step through the list and search for chessboard corners
        for image_file in chessboard_images:
            image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
            
            # If found, add object points, image points
            if ret == True:
                objpoints.append(objp)
                imgpoints.append(corners)
        return objpoints, imgpoints
      
        
    def calibrate_camera(self, chessboard_images):
        img = cv2.imread(chessboard_images[0])
        h, w = img.shape[:2]
        image_size = (w, h)
        
        objpoints, imgpoints = self.find_corner(chessboard_images)
        # Do camera calibration given object points and image points
        ret, self.camera_matrix, self.distortion_coeff, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, image_size, None,None)
        return self  

    
    def display_corners(self, chessboard_images, nx=9, ny=6):
        for image_file in chessboard_images:
            image = cv2.cvtColor(cv2.imread(image_file), cv2.COLOR_BGR2RGB)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            # Find the chessboard corners
            ret, corners = cv2.findChessboardCorners(gray, (nx,ny),None)
            img = cv2.drawChessboardCorners(image, (nx,ny), corners, ret)
            plt.imshow(img)
            plt.show()
        
        
    def save(self, model_file):
        model = {
            "camera_matrix": self.camera_matrix, 
            "distortion_coeff": self.distortion_coeff
        }
        pickle.dump(model, open(model_file, "wb"))
        print("calibration model saved at %s" % model_file)
        return self
    
    def restore(self, model_file):
        model = pickle.load(open(model_file, "rb"))
        self.camera_matrix = model["camera_matrix"]
        self.distortion_coeff = model["distortion_coeff"]
        print("calibration model restored from %s" % model_file)
        return self
    
    def undistort(self, image):
        """img: original RGB img to be undistorted
        Return undistorted image.
        """
        undistort_image = cv2.undistort(image, self.camera_matrix, self.distortion_coeff, None, self.camera_matrix)
        return undistort_image
        
def build_undistort_image():
    cc = CameraCalibration()
    cc.calibrate_camera(config.camera_calibration_images)
    return cc.undistort