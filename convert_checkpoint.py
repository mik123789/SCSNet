"""Export inference tensors using the paper's module names."""
import argparse

from scsnet.checkpoint import convert_checkpoint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', required=True, help='Existing checkpoint path')
    parser.add_argument('--destination', required=True, help='New inference checkpoint path')
    args = parser.parse_args()
    try:
        convert_checkpoint(args.source, args.destination)
    except (OSError, ValueError, RuntimeError) as error:
        parser.exit(2, 'Checkpoint conversion failed: {}\n'.format(error))
    print('Saved inference checkpoint to {}'.format(args.destination))


if __name__ == '__main__':
    main()
