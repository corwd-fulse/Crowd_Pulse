"""Execution configuration checks only; no detection/tracking calculations."""
import re


class InferenceFailure(ValueError):
    pass


def normalize_device(value: str) -> str:
    value = str(value).strip().lower()
    if value == 'cpu':
        return value
    if value == 'cuda':
        return 'cuda:0'
    match = re.fullmatch(r'(?:cuda:)?(\d+)', value)
    if match:
        return f'cuda:{int(match[1])}'
    raise ValueError('device must be cpu, cuda:N, or a single GPU index')


def check_device(device: str, torch_module=None) -> str:
    device = normalize_device(device)
    if device != 'cpu':
        if torch_module is None:
            import torch as torch_module
        if not torch_module.cuda.is_available():
            raise InferenceFailure('CUDA_UNAVAILABLE: use a CUDA-enabled environment or --device cpu')
        index = int(device.split(':')[1])
        if index >= torch_module.cuda.device_count():
            raise InferenceFailure('CUDA_DEVICE_UNAVAILABLE: requested GPU index does not exist')
    return device


def predictor_settings(predictor, requested: str) -> dict:
    import torch
    requested = normalize_device(requested)
    model = predictor.model.model  # AutoBackend delegates to its actual inference model.
    parameters = list(model.parameters())
    devices = sorted({str(p.device) for p in parameters})
    if (devices != [requested] or str(predictor.device) != requested
            or str(predictor.model.device) != requested):
        raise InferenceFailure('INFERENCE_DEVICE_MISMATCH: predictor and model must match requested device')
    if any(p.is_floating_point() and p.dtype != torch.float32 for p in parameters):
        raise InferenceFailure('INFERENCE_PRECISION_MISMATCH: FP32 is required')
    return {'device': requested, 'requested_device': requested,
            'predictor_device': str(predictor.device), 'predictor_model_device': str(predictor.model.device),
            'inference_parameter_devices': devices, 'precision': 'float32',
            'torch_version': str(torch.__version__), 'cuda_version': torch.version.cuda,
            'cudnn_version': torch.backends.cudnn.version(), 'device_verified': True}


def is_cuda_oom(exc: Exception) -> bool:
    return type(exc).__name__ in {'OutOfMemoryError', 'CudaOutOfMemoryError'} or (
        isinstance(exc, RuntimeError) and ('cuda out of memory' in str(exc).lower()
                                          or 'cuda error: out of memory' in str(exc).lower()))
