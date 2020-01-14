import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def _fake_quantize(parameter, bits, device):
    """
    Fake quantize a pretrained parameter.
    :param parameter: weight parameter
    :param bits: quantization bits
    :param device: CUDA device
    :return:
    """
    array = parameter.data.cpu().numpy()
    minval = np.min(array)
    maxval = np.max(array)

    num_intervals = (2 ** bits - 1)
    interval_value = (maxval - minval) / num_intervals
    raw_values = np.round((array - minval) / interval_value)
    quantized_array = raw_values * interval_value + minval
    return quantized_array


def fake_quantize_model(net, bits, device="cuda"):
    for m in net.modules():
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):          
            parameter = m.weight.data
            quantized_array = _fake_quantize(parameter, bits, device)
            m.weight.data = torch.from_numpy(quantized_array)
