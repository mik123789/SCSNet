"""Command-line stereo inference for SCSNet."""
import argparse
from pathlib import Path
import sys

import torch

from .checkpoint import load_model
from .io import disparity_paths, load_stereo_pair, matching_pairs, save_disparity


def _parser():
    parser = argparse.ArgumentParser(description='SCSNet stereo disparity inference')
    parser.add_argument('--checkpoint', required=True, type=Path)
    parser.add_argument('--left', required=True, type=Path, help='Left image or image directory')
    parser.add_argument('--right', required=True, type=Path, help='Right image or image directory')
    parser.add_argument('--output', required=True, type=Path, help='Output directory')
    parser.add_argument('--device', default='cuda:0' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--save-stages', action='store_true', help='Also export stages 1 to 4')
    parser.add_argument('--pad-height', type=int, help='Explicit padded height, a multiple of 8')
    parser.add_argument('--pad-width', type=int, help='Explicit padded width, a multiple of 8')
    parser.add_argument('--threads', type=int, default=2, help='CPU threads (default: 2)')
    return parser


def _output_stems(output, pairs, save_stages):
    return tuple(stem for left, _ in pairs for stem in
                 ((output / left.stem,) +
                  (tuple(output / ('stage{}'.format(stage)) / left.stem
                         for stage in range(1, 5)) if save_stages else ())))


def _validate_destination(args, pairs):
    output = args.output.resolve()
    input_directories = {path.resolve().parent for pair in pairs for path in pair}
    if output in input_directories:
        raise ValueError('The output directory must differ from the input image directories.')
    if output.exists() and not output.is_dir():
        raise ValueError('Output must be a directory.')
    protected = {path.resolve() for pair in pairs for path in pair} | {args.checkpoint.resolve()}
    for stem in _output_stems(args.output, pairs, args.save_stages):
        for path in disparity_paths(stem):
            if path.resolve() in protected or path.exists():
                raise FileExistsError('Refusing to overwrite an existing file: {}'.format(path))


def run(args):
    if args.threads <= 0:
        raise ValueError('--threads must be a positive integer.')
    device = torch.device(args.device)
    if device.type not in ('cpu', 'cuda'):
        raise ValueError('--device must be cpu or cuda[:index].')
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise ValueError('CUDA is unavailable; use --device cpu.')
    pairs = matching_pairs(args.left, args.right)
    _validate_destination(args, pairs)
    torch.set_num_threads(args.threads)
    model = load_model(args.checkpoint, device)
    for index, (left, right) in enumerate(pairs, start=1):
        i_l, i_r, original_hw = load_stereo_pair(left, right, args.pad_height, args.pad_width)
        with torch.inference_mode():
            outputs = model(i_l.to(device), i_r.to(device))
        save_disparity(outputs[-1], args.output / left.stem, original_hw)
        if args.save_stages:
            for stage, disparity in enumerate(outputs, start=1):
                save_disparity(disparity, args.output / ('stage{}'.format(stage)) / left.stem,
                               original_hw)
        print('[{}/{}] {}'.format(index, len(pairs), left.name), flush=True)
    return 0


def main(argv=None):
    args = _parser().parse_args(argv)
    try:
        return run(args)
    except (OSError, ValueError, TypeError, RuntimeError) as error:
        print('SCSNet inference failed: {}'.format(error), file=sys.stderr)
        return 1
