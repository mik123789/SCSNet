"""Safe, strict loading and renaming of SCSNet inference weights."""
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
import pickle

import torch

from .config import SCSNetConfig


def rename_legacy_state_dict(state_dict):
    """Return renamed keys without changing the caller's mapping or tensors."""
    if not isinstance(state_dict, Mapping) or not state_dict:
        raise ValueError('The checkpoint must contain a nonempty state dictionary.')
    if any(not isinstance(key, str) or not torch.is_tensor(value)
           for key, value in state_dict.items()):
        raise ValueError('State dictionary entries must have string keys and tensor values.')

    def renamed(key):
        key = key[7:] if key.startswith('module.') else key
        for old, new in (('feature_extraction.', 'acfe.'),
                         ('volume_postprocess.', 'cost_aggregation.'),
                         ('refine_spn.', 'ssr.')):
            if key.startswith(old):
                key = new + key[len(old):]
                break
        if key.startswith('acfe.') and key.endswith('.weights'):
            key = key[:-len('.weights')] + '.omega'
        return key

    pairs = tuple((renamed(key), value) for key, value in state_dict.items())
    if len({key for key, _ in pairs}) != len(pairs):
        raise ValueError('Checkpoint keys collide after legacy-name conversion.')
    return dict(pairs)


def _read_checkpoint(path):
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError('Checkpoint file does not exist: {}'.format(path))
    # Never fall back to arbitrary pickle execution for an unsupported checkpoint.
    try:
        payload = torch.load(str(path), map_location='cpu', weights_only=True)
    except pickle.UnpicklingError as error:
        raise ValueError('Safe checkpoint loading failed. Use a compatible PyTorch version '
                         'or a checkpoint containing only tensor weights.') from error
    if not isinstance(payload, Mapping):
        raise ValueError('Checkpoint must be a state dictionary or contain state_dict.')
    return payload


def _config_from_payload(payload):
    values = payload.get('model_config')
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise ValueError('model_config must be a mapping.')
    values = dict(values)
    for name in ('stage_search_ranges', 'aggregation_growth'):
        if name in values:
            try:
                values[name] = tuple(values[name])
            except TypeError as error:
                raise ValueError('{} must be a sequence.'.format(name)) from error
    try:
        return SCSNetConfig(**values)
    except (TypeError, ValueError) as error:
        raise ValueError('Invalid checkpoint model_config: {}'.format(error)) from error


def _validated_state(model, payload):
    state = rename_legacy_state_dict(payload.get('state_dict', payload))
    expected = model.state_dict()
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    incompatible = sorted(key for key in set(expected) & set(state)
                          if expected[key].shape != state[key].shape)
    if missing or unexpected or incompatible:
        raise ValueError('Checkpoint does not match this SCSNet: missing={}, unexpected={}, '
                         'wrong_shape={}'.format(missing, unexpected, incompatible))
    saved_config = _config_from_payload(payload)
    if saved_config is not None and hasattr(model, 'config') and saved_config != model.config:
        raise ValueError('Checkpoint model_config does not match the model configuration.')
    return state


def load_checkpoint(model, path):
    """Load legacy or public weights strictly, checking all keys before loading."""
    payload = _read_checkpoint(path)
    state = _validated_state(model, payload)
    model.load_state_dict(state, strict=True)
    return {'tensor_count': len(state), 'model_config': _config_from_payload(payload)}


def load_model(path, device='cpu'):
    """Construct an evaluation model using saved configuration or paper defaults."""
    from .model import SCSNet
    payload = _read_checkpoint(path)
    config = _config_from_payload(payload) or SCSNetConfig()
    model = SCSNet(config)
    model.load_state_dict(_validated_state(model, payload), strict=True)
    return model.to(torch.device(device)).eval()


def convert_checkpoint(source, destination, config=None):
    """Export new names and model configuration, omitting optimizer/training state."""
    from .model import SCSNet
    source, destination = Path(source), Path(destination)
    if source.resolve() == destination.resolve():
        raise ValueError('The destination must differ from the source checkpoint.')
    if destination.exists():
        raise FileExistsError('Destination already exists: {}'.format(destination))
    payload = _read_checkpoint(source)
    saved_config = _config_from_payload(payload)
    config = config if config is not None else (saved_config or SCSNetConfig())
    if not isinstance(config, SCSNetConfig):
        raise TypeError('config must be an SCSNetConfig instance.')
    model = SCSNet(config)
    state = _validated_state(model, payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as stream:
        torch.save({'state_dict': state, 'model_config': asdict(config)}, stream)
    return {'tensor_count': len(state), 'model_config': asdict(config)}
