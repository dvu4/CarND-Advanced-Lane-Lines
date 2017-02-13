"""
Self Driving Car Lane Detection Module. It 
- calibrate camera by chessboard images
- detects lane boundaries from images
- estimate lane curvature and distance-to-center by warp-transform
"""

__all__ = [
    'camera_calibration',
    'config',
    'lane_detection',
    'line_detection',
    'transform',
    'utility']

