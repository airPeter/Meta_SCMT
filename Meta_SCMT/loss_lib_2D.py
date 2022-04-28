import torch  
import matplotlib.pyplot as plt
import numpy as np
import cv2

mse = torch.nn.MSELoss(reduction = 'sum')

def max_center(If, center, max_length):
    intensity = torch.sum(torch.abs(If[center[0] - int(max_length//2): center[0] + int(max_length//2 + 1), center[1] - int(max_length//2): center[1] + int(max_length//2) + 1]))
    return - intensity

def max_center_circle(If, center, max_length):
    radius = max_length
    circle = np.zeros((s, s))
    circle = cv2.circle(circle, (center, center), radius, 1, -1)
def max_corner(If, Knn, res, max_length):
    #the field corner is not the true corner of the metasurface. there's some boundary of the field.
    #use start to skip the boundary to find the true corner.
    start = (Knn + 1) * res
    intensity = torch.sum(torch.abs(If[start:start + int(max_length//2) + 1,start:start + int(max_length//2) + 1]))
    return - intensity