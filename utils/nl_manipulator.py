# python3.7
"""Utility functions for latent codes manipulation."""

import numpy as np
from sklearn import svm
import torch
from utils.pose import viz_pose, calc_pose
from utils.noise import apply_manipulation, NNBoundary,PolyBoundary
import os
import time
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
import cv2
import numpy as np

from .logger import setup_logger

__all__ = ['train_net', 'predict_yaw', 'frontalize_latent', 'nonlinear_interpolate']

# ------------------------------------


# Load pose estimation models
cfg = yaml.load(open('configs/mb1_120x120.yml'), Loader=yaml.SafeLoader)
tddfa = TDDFA(gpu_mode=True, **cfg)
face_boxes = FaceBoxes()

def predict_yaw(pos_imgs,concat=True,to_np=True,move_ax=True):

    if to_np:
        pos_imgs=pos_imgs.cpu().numpy()             # cast pytorch tensor to numpy array
    if move_ax:
        pos_imgs=np.moveaxis(pos_imgs, 1,-1)        # change the (c,h,w) to (w,h,c)
    if concat:
        pos_imgs = np.concatenate(pos_imgs, axis=1) # Concate if input is multiple images in a batch
    # print(pos_imgs.shape)

    boxes = face_boxes(pos_imgs)
    # n = len(boxes)
    # print('Num of detected faces:',n)

    # TDDFA face pose estimation pipeline
    param_lst, roi_box_lst = tddfa(pos_imgs, boxes)

    ver_lst = tddfa.recon_vers(param_lst, roi_box_lst, dense_flag=False)

    yaw=list()
    for param, ver in zip(param_lst, ver_lst):
        P, pose = calc_pose(param)
        # img = plot_pose_box(img, P, ver)
        # print(P[:, :3])
        yaw.append(np.round(pose[0],3))

    return torch.Tensor(yaw).reshape(-1,1)


def frontalize_latent(start_latent, yaw_predict, yaw_bound, yaw_classifier, generator, synthesis_kwargs, max_iter=10, yaw_thresh=4):
    """
    Adjust the latent code to achieve a frontal view in the generated image.

    Args:
        start_latent (np.array): The starting latent code.
        yaw_predict (callable): Function to predict yaw angle from the image.
        yaw_bound (np.array): The linear boundary direction for yaw adjustment.
        yaw_classifier (nn.Module): The trained model for classification of latent codes.
        generator (object): The generator model to synthesize images.
        synthesis_kwargs (dict): Additional arguments for image synthesis.
        max_iter (int): Maximum number of iterations for adjustment. Default is 10.
        yaw_thresh (float): Threshold for yaw angle to consider the image as frontal. Default is 4.

    Returns:
        tuple: (final_latent, latent_list, yaw_list)
            final_latent (np.array): The final latent code after adjustment.
            latent_list (list): List of latent codes over iterations.
            yaw_list (list): List of yaw angles over iterations.
    """
    start_time = time.time()
    iter_c = 0
    cur_latent = start_latent
    latent_list = [cur_latent]


    # Generate the initial image from the start latent code
    start_img = generator.easy_synthesize(start_latent, **synthesis_kwargs)['image']

    # Predict the initial yaw angle
    cur_yaw = yaw_predict(np.array([start_img[0]]), True, False, False)
    yaw_list = [cur_yaw]

    while abs(cur_yaw) > yaw_thresh:
        if yaw_classifier is None:
            # Determine the direction of adjustment based on the current yaw
            if cur_yaw > 0:
                m = -1
            else:
                m = 1

            # Adjust the latent code in the direction to reduce the yaw
            cur_latent += yaw_bound * m
        
        else:
           cur_latent = apply_manipulation(cur_latent, yaw_classifier, multiplier=m, steps=1)


        # Generate a new image from the adjusted latent code
        cur_img = generator.easy_synthesize(cur_latent, **synthesis_kwargs)['image']
        
        # Predict the new yaw angle
        cur_yaw = yaw_predict(np.array([cur_img[0]]), True, False, False)

        # Append the new latent code and yaw angle to the lists
        latent_list.append(cur_latent)
        yaw_list.append(cur_yaw)

        iter_c += 1
        if iter_c >= max_iter:
            break

    end_time = time.time()
    print(f'Entire Frontalization Process Took {end_time - start_time:.2f} seconds.')

    return cur_latent, latent_list, yaw_list



def train_net(latent_codes,
                   scores,
                   chosen_num_or_ratio=0.2,
                   split_ratio=0.7,
                   num_layers=2,
                   num_neurons=128,
                   invalid_value=None,
                   logger=None):
  
  if not logger:
    logger = setup_logger(work_dir='', logger_name='train_network')

  if (not isinstance(latent_codes, np.ndarray) or
      not len(latent_codes.shape) == 2):
    raise ValueError(f'Input `latent_codes` should be with type'
                     f'`numpy.ndarray`, and shape [num_samples, '
                     f'latent_space_dim]!')
  num_samples = latent_codes.shape[0]
  latent_space_dim = latent_codes.shape[1]
  if (not isinstance(scores, np.ndarray) or not len(scores.shape) == 2 or
      not scores.shape[0] == num_samples or not scores.shape[1] == 1):
    raise ValueError(f'Input `scores` should be with type `numpy.ndarray`, and '
                     f'shape [num_samples, 1], where `num_samples` should be '
                     f'exactly same as that of input `latent_codes`!')
  if chosen_num_or_ratio <= 0:
    raise ValueError(f'Input `chosen_num_or_ratio` should be positive, '
                     f'but {chosen_num_or_ratio} received!')
  
  nn=NNBoundary(latent_space_dim,num_layers,num_neurons,name='stylegan')
  trainer= PolyBoundary(latent_codes,scores,split_ratio=split_ratio,chosen_num_or_ratio=chosen_num_or_ratio)
  trainer.train_nn(nn,10,0.01)
  return nn


def nonlinear_interpolate(latent_code, 
                          late_classifier, 
                          end_multiplier=3.0, 
                          steps=10):
    """Manipulates the given latent code iteratively with respect to a particular latent classifier gradient.

    This function takes a latent code, a boundary, and a classifier as inputs, and
    outputs a collection of manipulated latent codes. The latent codes are manipulated
    non-linearly by applying the classifier-based manipulation iteratively.

    Args:
        latent_code (np.array): The input latent code for manipulation.
        late_classifier (nn.Module): The latent classifier.
        end_multiplier (float): The multiplier to the boundary where the manipulation ends. (default: 3.0)
        steps (int): Number of steps to move the latent code from start position to end position. (default: 10)

    Returns:
        np.array: A collection of manipulated latent codes.
    """
    
    
    manipulated_latents = [latent_code]

    multiplier=(end_multiplier)/steps
    manipulated_latent=latent_code.copy()
    for _ in range(steps):   
      manipulated_latent = apply_manipulation(manipulated_latent, late_classifier, multiplier=multiplier, steps=1)
      manipulated_latents.append(manipulated_latent)

    return np.array(manipulated_latents)

