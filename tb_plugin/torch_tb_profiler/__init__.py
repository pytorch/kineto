# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

# pyre-unsafe

# Entry point for Pytorch TensorBoard plugin package.

__version__ = '0.4.3'

import warnings
warnings.warn(
   (
       "Deprecation Notice: The 'tb_plugin' submodule in Kineto is deprecated and will be removed on 03/05/2026.\n"
       "We do not plan to maintain or support this module going forward. If you rely on 'tb_plugin', please comment on the RFC issue in the Kineto repository and consider migrating your workflow.\n"
       "For more details, see the RFC issue: https://github.com/pytorch/kineto/issues/1248."
   ),
   FutureWarning,
   stacklevel=2
)

