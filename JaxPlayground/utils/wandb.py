__all__ = ['flatten_params']

import flax

def flatten_params(params): return {"/".join(k): v for k, v in flax.traverse_util.flatten_dict(params).items()}
