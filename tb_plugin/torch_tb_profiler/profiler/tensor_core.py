# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# -------------------------------------------------------------------------
class TC_Allowlist_Meta(type):
    # Enable grammar sugar as 'v in TC_Allowlist'.
    def __contains__(cls, item):
        return cls.__contains__(item)


class TC_Allowlist(metaclass=TC_Allowlist_Meta):
    # Refer to https://github.com/NVIDIA/PyProf/blob/fd1b2902e3306119eee40ba6b6e8b2f816920c29/pyprof/prof/tc.py#L19
    allowlist = ['h884', 's884', 'h1688', 's1688', 'hmma', 'i8816', '16816',
                 'dgrad_1x1_stride_2x2', 'first_layer_wgrad_kernel', 'conv1x1',
                 'conv2d_c1_k1', 'direct_group', 'xmma_implicit_gemm',
                 'xmma_sparse_conv', 'xmma_warp_specialized_implicit_gemm',
                 'xmma_gemm', 'xmma_sparse_gemm', 'c1688']

    @classmethod
    def __contains__(cls, item):
        # If kernel name contains substring equal to any one in allowlist, then it uses tensor core.
        for pattern in cls.allowlist:
            if pattern in item:
                return True
        return False


class TC_OP_Allowlist(metaclass=TC_Allowlist_Meta):
    # Refer to https://github.com/pytorch/pytorch/blob/69b2bf70f9c0e591ce5e566afa59e19618031ead/aten/src/ATen/autocast_mode.cpp#L290-L351 # noqa: E501
    allowlist = ['aten::_convolution', 'aten::conv1d', 'aten::conv2d', 'aten::conv3d', 'aten::conv_tbc',
                 'aten::conv_transpose1d', 'aten::conv_transpose2d', 'aten::conv_transpose3d',
                 'aten::convolution', 'aten::cudnn_convolution', 'aten::cudnn_convolution_transpose',
                 'aten::prelu', 'aten::addmm', 'aten::addmv', 'aten::addr',
                 'aten::matmul', 'aten::mm', 'aten::mv',
                 'aten::linear', 'aten::addbmm', 'aten::baddbmm', 'aten::bmm',
                 'aten::chain_matmul', 'aten::linalg_multi_dot',
                 'aten::_thnn_fused_lstm_cell', 'aten::_thnn_fused_gru_cell', 'aten::lstm_cell',
                 'aten::gru_cell', 'aten::rnn_tanh_cell', 'aten::rnn_relu_cell',
                 # The backward ops are got by running above ops' backward
                 # and recording whether it launched kernels.
                 'CudnnConvolutionBackward', 'BmmBackward0',
                 'aten::cudnn_convolution_transpose_backward', 'CudnnConvolutionTransposeBackward',
                 'MmBackward', 'aten::cudnn_convolution_backward_weight', 'aten::addmm_',
                 'AddmvBackward', 'MvBackward',
                 'aten::cudnn_convolution_transpose_backward_weight',
                 'aten::cudnn_convolution_transpose_backward_input',
                 'AddmmBackward', 'aten::cudnn_convolution_backward_input',
                 'AddbmmBackward', 'aten::cudnn_convolution_backward']

    @classmethod
    def __contains__(cls, item):
        # If operator name equals to any one in allowlist, then it is tensor core eligible.
        return item in cls.allowlist
