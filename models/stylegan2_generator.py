"""Contains the generator class of StyleGAN.

Basically, this class is derived from the `BaseGenerator` class defined in
`base_generator.py`.
"""

import os
import numpy as np

import torch

from . import model_settings
# from .stylegan2.model import Generator as StyleGAN2GeneratorModel
from .stylegan2.stylegan2_model import StyleGAN2Generator as StyleGAN2GeneratorModel
from .base_generator import BaseGenerator

__all__ = ['StyleGAN2Generator']

class StyleGAN2Generator(BaseGenerator):
  """Defines the generator class of StyleGAN2.
  
  (1) Z space, with dimension (512,)
  (2) W space, with dimension (512,)
  (3) W+ space, with dimension (18, 512)
  """

  def __init__(self, model_name, logger=None):
    self.truncation_psi = model_settings.STYLEGAN_TRUNCATION_PSI
    self.truncation_layers = model_settings.STYLEGAN_TRUNCATION_LAYERS
    self.randomize_noise = model_settings.STYLEGAN_RANDOMIZE_NOISE
    self.model_specific_vars = []
    super().__init__(model_name, logger)
    self.num_layers = (int(np.log2(self.resolution)) - 1) * 2
    assert self.gan_type == 'stylegan2'


  def build(self):
    self.check_attr('w_space_dim')
    # self.check_attr('fused_scale')
    self.model = StyleGAN2GeneratorModel(
        resolution=self.resolution,
        z_dim=self.w_space_dim,
        w_dim=self.w_space_dim,
        # trunc_psi=self.truncation_psi,
        # truncation_layers=self.truncation_layers,
        )


  def load(self):
    self.logger.info(f'Loading pytorch model from `{self.model_path}`.')
    
    
    state_dict = torch.load(self.model_path)['models']['generator']
    print('gan model spec type:\n',[a for a in self.model.state_dict().keys() if a not in state_dict.keys()])
    for var_name in self.model_specific_vars:
      state_dict[var_name] = self.model.state_dict()[var_name]
    self.model.load_state_dict(state_dict)
    self.logger.info(f'Successfully loaded!')
    # self.lod = self.model.synthesis.lod.to(self.cpu_device).tolist()
    # self.logger.info(f'  `lod` of the loaded model is {self.lod}.')


  def sample(self, num, latent_space_type='Z'):
    """Samples latent codes randomly.

    Args:
      num: Number of latent codes to sample. Should be positive.
      latent_space_type: Type of latent space from which to sample latent code.
        Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

    Returns:
      A `numpy.ndarray` as sampled latend codes.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
      latent_codes = np.random.randn(num, self.latent_space_dim)
      norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
      latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
    elif latent_space_type == 'W':
      latent_codes = np.random.randn(num, self.w_space_dim)      
      
    elif latent_space_type == 'WP':
      latent_codes = np.random.randn(num, self.num_layers, self.w_space_dim)      
      
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)


  def preprocess(self, latent_codes, latent_space_type='Z'):
    """Preprocesses the input latent code if needed.

    Args:
      latent_codes: The input latent codes for preprocessing.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)

    Returns:
      The preprocessed latent codes which can be used as final input for the
        generator.

    Raises:
      ValueError: If the given `latent_space_type` is not supported.
    """
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    latent_space_type = latent_space_type.upper()
    if latent_space_type == 'Z':
      latent_codes = latent_codes.reshape(-1, self.latent_space_dim)
      norm = np.linalg.norm(latent_codes, axis=1, keepdims=True)
      latent_codes = latent_codes / norm * np.sqrt(self.latent_space_dim)
    elif latent_space_type == 'W':
      latent_codes = latent_codes.reshape(-1, self.w_space_dim)
    elif latent_space_type == 'WP':
      latent_codes = latent_codes.reshape(-1, self.num_layers, self.w_space_dim)
    else:
      raise ValueError(f'Latent space type `{latent_space_type}` is invalid!')

    return latent_codes.astype(np.float32)


  
  def easy_sample(self, num, latent_space_type='Z'):
    return self.preprocess(self.sample(num, latent_space_type),
                           latent_space_type)
  

  def synthesize(self,
                 latent_codes,
                 latent_space_type='Z',
                 generate_style=False,
                 generate_image=True):
    """Synthesizes images with given latent codes.

    One can choose whether to generate the layer-wise style codes.

    Args:
      latent_codes: Input latent codes for image synthesis.
      latent_space_type: Type of latent space to which the latent codes belong.
        Only [`Z`, `W`, `WP`] are supported. Case insensitive. (default: `Z`)
      generate_style: Whether to generate the layer-wise style codes. (default:
        False)
      generate_image: Whether to generate the final image synthesis. (default:
        True)

    Returns:
      A dictionary whose values are raw outputs from the generator.
    """
    if not isinstance(latent_codes, np.ndarray):
      raise ValueError(f'Latent codes should be with type `numpy.ndarray`!')

    results = {}

    latent_space_type = latent_space_type.upper()
    latent_codes_shape = latent_codes.shape

    # if self.randomize_noise:
    #     noise = [None] * self.num_layers
    # else:
    #     noise = [
    #         getattr(self.noises, f'noise_{i}') for i in range(self.num_layers)
    #     ]

    # Generate from Z space. 
    if latent_space_type == 'Z':
      if not (len(latent_codes_shape) == 2 and
              latent_codes_shape[0] <= self.batch_size and
              latent_codes_shape[1] == self.latent_space_dim):
        raise ValueError(f'Latent_codes should be with shape [batch_size, '
                         f'latent_space_dim], where `batch_size` no larger '
                         f'than {self.batch_size}, and `latent_space_dim` '
                         f'equal to {self.latent_space_dim}!\n'
                         f'But {latent_codes_shape} received!')
      zs = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      zs = zs.to(self.run_device)
      
      mapping_results = self.model.mapping(zs, impl=self.run_device)
      ws = mapping_results['w']
      wp = mapping_results.pop('wp')
      

      trunc_psi = 1.0 if self.truncation_psi is None else self.truncation_psi
      trunc_layers = 0 if self.truncation_layers is None else self.truncation_layers
      if trunc_psi < 1.0 and trunc_layers > 0:
          w_avg = self.model.w_avg.reshape(1, -1, self.w_space_dim)[:, :trunc_layers]
          wp[:, :trunc_layers] = w_avg.lerp(
              wp[:, :trunc_layers], trunc_psi)

      # synthesis_results = self.model.synthesis(wp,                                               
      #                                      noise_mode=self.randomize_noise,                                           
      #                                      impl=self.run_device,
      #                                      )

      results['z'] = latent_codes
      results['w'] = self.get_value(ws)

      
    # Generate from W space.
    elif latent_space_type == 'W':

      # inject_index = self.num_layers
      ws = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      ws = ws.to(self.run_device)
      wp = ws.unsqueeze(1).repeat((1, self.num_layers, 1))
      trunc_psi = 1.0 if self.truncation_psi is None else self.truncation_psi
      trunc_layers = 0 if self.truncation_layers is None else self.truncation_layers
      if trunc_psi < 1.0 and trunc_layers > 0:
          w_avg = self.model.w_avg.reshape(1, -1, self.w_space_dim)[:, :trunc_layers]
          wp[:, :trunc_layers] = w_avg.lerp(
              wp[:, :trunc_layers], trunc_psi)
          
      results['w'] = self.get_value(ws)
    # # Generate from W+ space.
    elif latent_space_type == 'WP':
      wp = torch.from_numpy(latent_codes).type(torch.FloatTensor)
      wp = wp.to(self.run_device)
      trunc_psi = 1.0 if self.truncation_psi is None else self.truncation_psi
      trunc_layers = 0 if self.truncation_layers is None else self.truncation_layers
      if trunc_psi < 1.0 and trunc_layers > 0:
          w_avg = self.model.w_avg.reshape(1, -1, self.w_space_dim)[:, :trunc_layers]
          wp[:, :trunc_layers] = w_avg.lerp(
              wp[:, :trunc_layers], trunc_psi)

    results['wp'] = self.get_value(wp)
    synthesis_results = self.model.synthesis(wp,                                               
                                           noise_mode='none' if self.randomize_noise else 'const',                                           
                                           impl=self.run_device,
                                           )
    
    
    

    images = synthesis_results['image']


    if generate_image:
      results['image'] = self.get_value(images)


    return results
  

