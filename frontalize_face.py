# python3.7

import os.path
import argparse
from collections import defaultdict
import cv2
import numpy as np
from tqdm import tqdm
from utils.nl_manipulator import frontalize_latent,predict_yaw

from models.model_settings import MODEL_POOL
from models.pggan_generator import PGGANGenerator
from models.stylegan_generator import StyleGANGenerator
from models.stylegan2_generator import StyleGAN2Generator
from utils.logger import setup_logger

def parse_args():
    """Parses arguments."""
    parser = argparse.ArgumentParser(description='Generate images with given model and frontalize latent code.')
    parser.add_argument('-m', '--model_name', type=str, required=True,
                        choices=list(MODEL_POOL),
                        help='Name of the model for generation. (required)')
    parser.add_argument('-o', '--output_dir', type=str, required=True,
                        help='Directory to save the output results. (required)')
    parser.add_argument('-l', '--latent_code_path', type=str, required=True,
                        help='Path to the initial latent code (.npy file). (required)')
    parser.add_argument('-b', '--boundary_path', type=str, required=True,
                        help='Path to the saved boundary for editing yaw (.npy file). (required)')
    parser.add_argument('-s', '--latent_space_type', type=str, default='z',
                        choices=['z', 'Z', 'w', 'W', 'wp', 'wP', 'Wp', 'WP'],
                        help='Latent space used in Style GAN. (default: `Z`)')
    parser.add_argument('-t', '--yaw_threshold', type=float, default=4.0,
                        help='Threshold for yaw angle to consider the image as frontal. (default: 4.0)')
    parser.add_argument('-i', '--max_iter', type=int, default=10,
                        help='Maximum number of iterations for adjustment. (default: 10)')
    parser.add_argument('-p', '--img_classifier', required=True,
                        help='Path to pre-trained image classifier .PKL file for generating labels.')

    return parser.parse_args()


# Main function to execute the frontalization process
def main():
    """Main function."""
    args = parse_args()
    logger = setup_logger(args.output_dir, logger_name='generate_data')

    logger.info(f'Initializing generator.')
    gan_type = MODEL_POOL[args.model_name]['gan_type']
    if gan_type == 'pggan':
        model = PGGANGenerator(args.model_name, logger)
        kwargs = {}
    elif gan_type == 'stylegan':
        model = StyleGANGenerator(args.model_name, logger)
        kwargs = {'latent_space_type': args.latent_space_type}
    elif gan_type == 'stylegan2':
        model = StyleGAN2Generator(args.model_name, logger)
        kwargs = {'latent_space_type': args.latent_space_type}  
    else:
        raise NotImplementedError(f'Not implemented GAN type `{gan_type}`!')

    # Load the initial latent code
    logger.info(f'Loading initial latent code from {args.latent_code_path}.')
    initial_latent = np.load(args.latent_code_path)

    # Load the yaw boundary
    logger.info(f'Loading yaw boundary from {args.boundary_path}.')
    yaw_bound = np.load(args.boundary_path)


    # Perform frontalization
    logger.info(f'Performing frontalization.')
    final_latent, latent_list, yaw_list = frontalize_latent(
        start_latent=initial_latent,
        yaw_predict=predict_yaw,
        yaw_bound=yaw_bound,
        generator=model,
        synthesis_kwargs=kwargs,
        max_iter=args.max_iter,
        yaw_thresh=args.yaw_threshold
    )

    # Save the final latent code
    final_latent_path = os.path.join(args.output_dir, os.path.basename(args.latent_code_path).replace('.npy', '_front.npy'))
    logger.info(f'Saving final latent code to {final_latent_path}.')
    np.save(final_latent_path, final_latent)

if __name__ == '__main__':
    main()
