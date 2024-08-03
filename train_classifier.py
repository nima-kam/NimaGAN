# python3.7
"""
Trains a neural network for classification of latent codes.

This file takes a collection of `latent code - attribute score`
pairs and trains a neural network to classify these latent codes.
The neural network is trained to distinguish between different 
attributes based on the given scores. Once trained, the gradient 
of the neural network with respect to the input latent codes is 
used as the direction for latent manipulation. The trained neural 
network will be saved and can be used to manipulate the corresponding 
attribute of the synthesis by applying the gradients derived from 
the neural network.

"""

import os.path
import argparse
import numpy as np

from utils.logger import setup_logger
from utils.manipulator import train_boundary
from utils.nl_manipulator import train_net

def parse_args():
  """Parses arguments."""
  parser = argparse.ArgumentParser(
      description='Train NN classifier with given latent codes and '
                  'attribute scores.')
  parser.add_argument('-o', '--output_dir', type=str, required=True,
                      help='Directory to save the output results. (required)')
  parser.add_argument('-c', '--latent_codes_path', type=str, required=True,
                      help='Path to the input latent codes. (required)')
  parser.add_argument('-s', '--scores_path', type=str, required=True,
                      help='Path to the input attribute scores. (required)')
  parser.add_argument('-n', '--chosen_num_or_ratio', type=float, default=0.7,
                      help='How many samples to choose for training. '
                           '(default: 0.2)')
  parser.add_argument('-r', '--split_ratio', type=float, default=0.7,
                      help='Ratio with which to split training and validation '
                           'sets. (default: 0.7)')
  parser.add_argument('-V', '--invalid_value', type=float, default=None,
                      help='Sample whose attribute score is equal to this '
                           'field will be ignored. (default: None)')
  parser.add_argument('-h', '--num_layers', type=int, default=2,
                      help='Number of hidden layers for the neural network which is '
                           'training. (default: 2)')
  parser.add_argument('-l', '--num_neurons', type=int, default=256,
                      help='Number of neurons in each layer of the neural network which is '
                           'training. (default: 256)')

  return parser.parse_args()


def main():
  """Main function."""
  args = parse_args()
  logger = setup_logger(args.output_dir, logger_name='generate_data')

  logger.info('Loading latent codes.')
  if not os.path.isfile(args.latent_codes_path):
    raise ValueError(f'Latent codes `{args.latent_codes_path}` does not exist!')
  latent_codes = np.load(args.latent_codes_path)

  logger.info('Loading attribute scores.')
  if not os.path.isfile(args.scores_path):
    raise ValueError(f'Attribute scores `{args.scores_path}` does not exist!')
  scores = np.load(args.scores_path)

  nn = train_net(latent_codes=latent_codes,
                            scores=scores,
                            chosen_num_or_ratio=args.chosen_num_or_ratio,
                            split_ratio=args.split_ratio,
                            num_layers=args.num_layers,
                            num_neurons=args.num_neurons,
                            invalid_value=args.invalid_value,
                            logger=logger)
  
  nn.save_pkl(base_path=args.output_dir)

if __name__ == '__main__':
  main()
