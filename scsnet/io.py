"""Stereo image preparation and lossless floating-point disparity export."""
from pathlib import Path

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

IMAGE_EXTENSIONS = frozenset(('.png', '.jpg', '.jpeg'))
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def matching_pairs(left, right):
    """Return deterministic file pairs, rejecting missing or ambiguous matches."""
    left, right = Path(left), Path(right)
    if left.is_file() and right.is_file():
        if any(path.suffix.lower() not in IMAGE_EXTENSIONS for path in (left, right)):
            raise ValueError('Stereo files must be PNG or JPEG images.')
        return ((left, right),)
    if not left.is_dir() or not right.is_dir():
        raise ValueError('Provide two image files or two image directories.')
    left_files = tuple(sorted((path for path in left.iterdir()
                               if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS),
                              key=lambda path: path.name))
    right_files = tuple(path for path in right.iterdir()
                        if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS)
    if not left_files:
        raise ValueError('The left image directory contains no supported images.')
    if len({path.stem for path in left_files}) != len(left_files):
        raise ValueError('Left image filenames produce duplicate output stems.')
    if {path.name for path in left_files} != {path.name for path in right_files}:
        raise ValueError('Left and right directories must contain matching image filenames.')
    return tuple((path, right / path.name) for path in left_files)


def _normalized_rgb(path):
    with Image.open(str(path)) as image:
        pixels = np.array(image.convert('RGB'), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(pixels.transpose(2, 0, 1)).unsqueeze(0)
    mean = tensor.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = tensor.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return (tensor - mean) / std


def _padding_dimension(size, requested, minimum, name):
    if requested is None:
        return ((max(size, minimum) + 7) // 8) * 8
    if isinstance(requested, bool) or not isinstance(requested, int):
        raise ValueError('{} must be an integer.'.format(name))
    if requested < size or requested % 8 != 0:
        raise ValueError('{} must be a multiple of 8 and at least {}.'.format(name, size))
    return requested


def load_stereo_pair(left, right, pad_height=None, pad_width=None):
    """Normalize RGB and zero-pad above/right; return (I_L, I_R, original_hw)."""
    i_l, i_r = _normalized_rgb(left), _normalized_rgb(right)
    if i_l.shape != i_r.shape:
        raise ValueError('Left and right images must have the same dimensions.')
    height, width = i_l.shape[-2:]
    target_height = _padding_dimension(height, pad_height, 384, 'pad_height')
    target_width = _padding_dimension(width, pad_width, 1248, 'pad_width')
    padding = (0, target_width - width, target_height - height, 0)
    return F.pad(i_l, padding), F.pad(i_r, padding), (height, width)


def disparity_paths(stem):
    stem = Path(stem)
    return Path(str(stem) + '.npy'), Path(str(stem) + '.png')


def save_disparity(disparity, stem, original_hw):
    """Save raw float32 pixels and KITTI PNG (clip[0,192] × 256, truncated)."""
    if disparity.ndim != 4 or disparity.shape[:2] != (1, 1):
        raise ValueError('Disparity must have shape [1,1,H,W].')
    height, width = original_hw
    if height <= 0 or width <= 0 or height > disparity.shape[-2] or width > disparity.shape[-1]:
        raise ValueError('Original image dimensions do not fit the disparity map.')
    values = disparity[0, 0, -height:, :width].detach().cpu().numpy().astype(np.float32)
    if not np.isfinite(values).all():
        raise ValueError('The predicted disparity contains non-finite values.')
    raw_path, png_path = disparity_paths(stem)
    for path in (raw_path, png_path):
        if path.exists():
            raise FileExistsError('Refusing to overwrite output: {}'.format(path))
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    encoded = (np.clip(values, 0.0, 192.0) * 256.0).astype(np.uint16)
    with raw_path.open('xb') as stream:
        np.save(stream, values, allow_pickle=False)
    with png_path.open('xb') as stream:
        Image.fromarray(encoded).save(stream, format='PNG')
    return raw_path, png_path
