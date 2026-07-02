import os,jax
# disable triton_gemm for jax versions > 0.3
if int(jax.__version__.split(".")[1]) > 3:
  os.environ["XLA_FLAGS"] = "--xla_gpu_enable_triton_gemm=false"

# Blackwell / jax >=0.10: restore jax.tree_* aliases removed upstream (moved to jax.tree_util).
for _n in ("tree_map", "tree_flatten", "tree_leaves", "tree_unflatten", "tree_structure", "tree_all"):
  if not hasattr(jax, _n) and hasattr(jax.tree_util, _n):
    setattr(jax, _n, getattr(jax.tree_util, _n))
if not hasattr(jax, "tree_multimap"):
  jax.tree_multimap = jax.tree_util.tree_map  # tree_multimap merged into tree_map
# jax >=0.10 removed jax.util (moved to jax._src.util); restore for colabdesign's mapping.py
import jax._src.util as _jax_src_util
if not hasattr(jax, "util"):
  jax.util = _jax_src_util

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from .shared.utils import clear_mem
from .af.model import mk_af_model
from .tr.model import mk_tr_model
from .mpnn.model import mk_mpnn_model

# backward compatability
mk_design_model = mk_afdesign_model = mk_af_model
mk_trdesign_model = mk_tr_model