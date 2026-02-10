# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# --------------------------------------------------------------------------

# pyre-unsafe

# Entry point for Pytorch TensorBoard plugin package.

__version__ = '0.4.3'

import warnings
warnings.warn(
    (
        "The TensorBoard integration with the PyTorch profiler ('tb_plugin') is deprecated and scheduled for removal on 03/05/2026. For further details, please see the RFC: https://github.com/pytorch/kineto/issues/1248."
        "If your workflow depends on 'tb_plugin', we encourage you to comment on the RFC issue or begin migrating to alternative solutions."
    ),
    UserWarning,
    stacklevel=2
)
