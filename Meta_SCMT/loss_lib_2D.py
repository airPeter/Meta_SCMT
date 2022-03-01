import torch  
import matplotlib.pyplot as plt
import numpy as np

mse = torch.nn.MSELoss(reduction = 'sum')

def max_center(If, total_size, max_length):
    center = int(total_size//2)
    intensity = torch.sum(torch.abs(If[center - int(max_length//2): center + int(max_length//2), center - int(max_length//2): center + int(max_length//2)]))
    return - intensity

def max_corner(If, Knn, res, max_length):
    #the field corner is not the true corner of the metasurface. there's some boundary of the field.
    #use start to skip the boundary to find the true corner.
    start = (Knn + 1) * res
    intensity = torch.sum(torch.abs(If[start:start + int(max_length//2),start:start + int(max_length//2)]))
    return - intensity