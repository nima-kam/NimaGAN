# python3.7
"""Utility functions for latent codes manipulation."""

import numpy as np
from sklearn import svm
import torch
from utils.pose import viz_pose, calc_pose
from utils.noise import apply_manipulation
import os
import time
import yaml
from FaceBoxes import FaceBoxes
from TDDFA import TDDFA
import cv2
import numpy as np

from .logger import setup_logger

__all__ = ['train_boundary', 'project_boundary', 'linear_interpolate', 'predict_yaw', 'frontalize_latent']

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


def frontalize_latent(start_latent, yaw_predict, yaw_bound, generator, synthesis_kwargs, max_iter=10, yaw_thresh=4):
    """
    Adjust the latent code to achieve a frontal view in the generated image.

    Args:
        start_latent (np.array): The starting latent code.
        yaw_predict (callable): Function to predict yaw angle from the image.
        yaw_bound (np.array): The boundary direction for yaw adjustment.
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
        # Determine the direction of adjustment based on the current yaw
        if cur_yaw > 0:
            m = -1
        else:
            m = 1

        # Adjust the latent code in the direction to reduce the yaw
        cur_latent += yaw_bound * m

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














# --------------------------------------
def train_boundary(latent_codes,
                   scores,
                   chosen_num_or_ratio=0.2,
                   split_ratio=0.7,
                   invalid_value=None,
                   logger=None):
  """Trains boundary in latent space with offline predicted attribute scores.

  Given a collection of latent codes and the attribute scores predicted from the
  corresponding images, this function will train a linear SVM by treating it as
  a bi-classification problem. Basically, the samples with highest attribute
  scores are treated as positive samples, while those with lowest scores as
  negative. For now, the latent code can ONLY be with 1 dimension.

  NOTE: The returned boundary is with shape (1, latent_space_dim), and also
  normalized with unit norm.

  Args:
    latent_codes: Input latent codes as training data.
    scores: Input attribute scores used to generate training labels.
    chosen_num_or_ratio: How many samples will be chosen as positive (negative)
      samples. If this field lies in range (0, 0.5], `chosen_num_or_ratio *
      latent_codes_num` will be used. Otherwise, `min(chosen_num_or_ratio,
      0.5 * latent_codes_num)` will be used. (default: 0.2)
    split_ratio: Ratio to split training and validation sets. (default: 0.7)
    invalid_value: This field is used to filter out data. (default: None)
    logger: Logger for recording log messages. If set as `None`, a default
      logger, which prints messages from all levels to screen, will be created.
      (default: None)

  Returns:
    A decision boundary with type `numpy.ndarray`.

  Raises:
    ValueError: If the input `latent_codes` or `scores` are with invalid format.
  """
  if not logger:
    logger = setup_logger(work_dir='', logger_name='train_boundary')

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

  logger.info(f'Filtering training data.')
  if invalid_value is not None:
    latent_codes = latent_codes[scores[:, 0] != invalid_value]
    scores = scores[scores[:, 0] != invalid_value]

  logger.info(f'Sorting scores to get positive and negative samples.')
  sorted_idx = np.argsort(scores, axis=0)[::-1, 0]
  latent_codes = latent_codes[sorted_idx]
  scores = scores[sorted_idx]
  num_samples = latent_codes.shape[0]
  if 0 < chosen_num_or_ratio <= 1:
    chosen_num = int(num_samples * chosen_num_or_ratio)
  else:
    chosen_num = int(chosen_num_or_ratio)
  chosen_num = min(chosen_num, num_samples // 2)

  logger.info(f'Spliting training and validation sets:')
  train_num = int(chosen_num * split_ratio)
  val_num = chosen_num - train_num
  # Positive samples.
  positive_idx = np.arange(chosen_num)
  np.random.shuffle(positive_idx)
  positive_train = latent_codes[:chosen_num][positive_idx[:train_num]]
  positive_val = latent_codes[:chosen_num][positive_idx[train_num:]]
  # Negative samples.
  negative_idx = np.arange(chosen_num)
  np.random.shuffle(negative_idx)
  negative_train = latent_codes[-chosen_num:][negative_idx[:train_num]]
  negative_val = latent_codes[-chosen_num:][negative_idx[train_num:]]
  # Training set.
  train_data = np.concatenate([positive_train, negative_train], axis=0)
  train_label = np.concatenate([np.ones(train_num, dtype=np.int32),
                                np.zeros(train_num, dtype=np.int32)], axis=0)
  logger.info(f'  Training: {train_num} positive, {train_num} negative.')
  # Validation set.
  val_data = np.concatenate([positive_val, negative_val], axis=0)
  val_label = np.concatenate([np.ones(val_num, dtype=np.int32),
                              np.zeros(val_num, dtype=np.int32)], axis=0)
  logger.info(f'  Validation: {val_num} positive, {val_num} negative.')
  # Remaining set.
  remaining_num = num_samples - chosen_num * 2
  remaining_data = latent_codes[chosen_num:-chosen_num]
  remaining_scores = scores[chosen_num:-chosen_num]
  decision_value = (scores[0] + scores[-1]) / 2
  remaining_label = np.ones(remaining_num, dtype=np.int32)
  remaining_label[remaining_scores.ravel() < decision_value] = 0
  remaining_positive_num = np.sum(remaining_label == 1)
  remaining_negative_num = np.sum(remaining_label == 0)
  logger.info(f'  Remaining: {remaining_positive_num} positive, '
              f'{remaining_negative_num} negative.')

  logger.info(f'Training boundary.')
  clf = svm.LinearSVC(max_iter=20000)
  classifier = clf.fit(train_data, train_label)
  logger.info(f'Finish training.')

  train_prediction = classifier.predict(train_data)
  correct_num = np.sum(train_label == train_prediction)
  logger.info(f'Accuracy for train set: '
              f'{correct_num} / {train_num * 2} = '
              f'{correct_num / (train_num * 2):.6f}')

  if val_num:
    val_prediction = classifier.predict(val_data)
    correct_num = np.sum(val_label == val_prediction)
    logger.info(f'Accuracy for validation set: '
                f'{correct_num} / {val_num * 2} = '
                f'{correct_num / (val_num * 2):.6f}')

  if remaining_num:
    remaining_prediction = classifier.predict(remaining_data)
    correct_num = np.sum(remaining_label == remaining_prediction)
    logger.info(f'Accuracy for remaining set: '
                f'{correct_num} / {remaining_num} = '
                f'{correct_num / remaining_num:.6f}')

  a = classifier.coef_.reshape(1, latent_space_dim).astype(np.float32)
  return a / np.linalg.norm(a)


def project_boundary(primal, *args):
  """Projects the primal boundary onto condition boundaries.
  
  The function is used for conditional manipulation, where the projected vector
  will be subscribed from the normal direction of the original boundary. Here,
  all input boundaries are supposed to have already been normalized to unit
  norm, and with same shape [1, latent_space_dim].
  
  Args:
    primal: The primal boundary.
    *args: Other boundaries as conditions.
  
  Returns:
    A projected boundary (also normalized to unit norm), which is orthogonal to
      all condition boundaries.
  
  Raises:
    LinAlgError: If there are more than two condition boundaries and the method fails 
                 to find a projected boundary orthogonal to all condition boundaries.
  """
  assert len(primal.shape) == 2 and primal.shape[0] == 1

  if not args:
    return primal
  if len(args) == 1:
    cond = args[0]
    assert (len(cond.shape) == 2 and cond.shape[0] == 1 and
            cond.shape[1] == primal.shape[1])
    new = primal - primal.dot(cond.T) * cond
    return new / np.linalg.norm(new)
  elif len(args) == 2:
    cond_1 = args[0]
    cond_2 = args[1]
    assert (len(cond_1.shape) == 2 and cond_1.shape[0] == 1 and
            cond_1.shape[1] == primal.shape[1])
    assert (len(cond_2.shape) == 2 and cond_2.shape[0] == 1 and
            cond_2.shape[1] == primal.shape[1])
    primal_cond_1 = primal.dot(cond_1.T)
    primal_cond_2 = primal.dot(cond_2.T)
    cond_1_cond_2 = cond_1.dot(cond_2.T)
    alpha = (primal_cond_1 - primal_cond_2 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    beta = (primal_cond_2 - primal_cond_1 * cond_1_cond_2) / (
        1 - cond_1_cond_2 ** 2 + 1e-8)
    new = primal - alpha * cond_1 - beta * cond_2
    return new / np.linalg.norm(new)
  else:
    for cond_boundary in args:
      assert (len(cond_boundary.shape) == 2 and cond_boundary.shape[0] == 1 and
              cond_boundary.shape[1] == primal.shape[1])
    cond_boundaries = np.squeeze(np.asarray(args))
    A = np.matmul(cond_boundaries, cond_boundaries.T)
    B = np.matmul(cond_boundaries, primal.T)
    x = np.linalg.solve(A, B)
    new = primal - (np.matmul(x.T, cond_boundaries))
    return new / np.linalg.norm(new)







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










def linear_interpolate(latent_code,
                       boundary,
                       start_distance=-3.0,
                       end_distance=3.0,
                       steps=10):
  """Manipulates the given latent code with respect to a particular boundary.

  Basically, this function takes a latent code and a boundary as inputs, and
  outputs a collection of manipulated latent codes. For example, let `steps` to
  be 10, then the input `latent_code` is with shape [1, latent_space_dim], input
  `boundary` is with shape [1, latent_space_dim] and unit norm, the output is
  with shape [10, latent_space_dim]. The first output latent code is
  `start_distance` away from the given `boundary`, while the last output latent
  code is `end_distance` away from the given `boundary`. Remaining latent codes
  are linearly interpolated.

  Input `latent_code` can also be with shape [1, num_layers, latent_space_dim]
  to support W+ space in Style GAN. In this case, all features in W+ space will
  be manipulated same as each other. Accordingly, the output will be with shape
  [10, num_layers, latent_space_dim].

  NOTE: Distance is sign sensitive.

  Args:
    latent_code: The input latent code for manipulation.
    boundary: The semantic boundary as reference.
    start_distance: The distance to the boundary where the manipulation starts.
      (default: -3.0)
    end_distance: The distance to the boundary where the manipulation ends.
      (default: 3.0)
    steps: Number of steps to move the latent code from start position to end
      position. (default: 10)
  """
  assert (latent_code.shape[0] == 1 and boundary.shape[0] == 1 and
          len(boundary.shape) == 2 and
          boundary.shape[1] == latent_code.shape[-1])

  linspace = np.linspace(start_distance, end_distance, steps)
  if len(latent_code.shape) == 2:
    linspace = linspace - latent_code.dot(boundary.T)
    linspace = linspace.reshape(-1, 1).astype(np.float32)
    return latent_code + linspace * boundary
  if len(latent_code.shape) == 3:
    linspace = linspace.reshape(-1, 1, 1).astype(np.float32)
    return latent_code + linspace * boundary.reshape(1, 1, -1)
  raise ValueError(f'Input `latent_code` should be with shape '
                   f'[1, latent_space_dim] or [1, N, latent_space_dim] for '
                   f'W+ space in Style GAN!\n'
                   f'But {latent_code.shape} is received.')
