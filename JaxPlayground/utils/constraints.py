__all__ = ['clip_layer_kernel', 'clip_layer_kernel_bias', 'clip_layer', 'clip_param']

from jax import numpy as jnp
import flax
from flax.core import freeze

def clip_layer_kernel(params, # PyTree containing the parameters to clip.
                      layer_name: str, # String indicating the name of the layer. It can be a generic like "Conv" or specific like "Conv_0". 
                      a_min=None, # Min value to clip to.
                      a_max=None, # Max value to clip to.
                      ): # Same PyTree as `params` but with the corresponding values clipped.
    return freeze(flax.traverse_util.path_aware_map(lambda path, v: jnp.clip(v, a_min, a_max) if (layer_name in "_".join(path)) and ("kernel" in path) else v, params))

def clip_layer_kernel_bias(params, # PyTree containing the parameters to clip.
                           layer_name: str, # String indicating the name of the layer. It can be a generic like "Conv" or specific like "Conv_0". 
                           a_min=None, # Min value to clip to.
                           a_max=None, # Max value to clip to.
                           ): # Same PyTree as `params` but with the corresponding values clipped.
    return freeze(flax.traverse_util.path_aware_map(lambda path, v: jnp.clip(v, a_min, a_max) if (layer_name in "_".join(path)) and (("kernel" in path) or ("bias" in path)) else v, params))

def clip_layer(params, # PyTree containing the parameters to clip.
               layer_name: str, # String indicating the name of the layer. It can be a generic like "Conv" or specific like "Conv_0". 
               a_min=None, # Min value to clip to.
               a_max=None, # Max value to clip to.
               ): # Same PyTree as `params` but with the corresponding values clipped.
    return freeze(flax.traverse_util.path_aware_map(lambda path, v: jnp.clip(v, a_min, a_max) if layer_name in "_".join(path) else v, params))

def clip_param(params, # PyTree containing the parameters to clip.
               param_name: str, # String indicating the name of the parameter.
               a_min=None, # Min value to clip to.
               a_max=None, # Max value to clip to.
               ): # Same PyTree as `params` but with the corresponding values clipped.
    return freeze(flax.traverse_util.path_aware_map(lambda path, v: jnp.clip(v, a_min, a_max) if param_name in path else v, params))
