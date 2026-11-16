"""Checkpoint, image handling, and command-line inference contracts."""
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn

from scsnet.checkpoint import convert_checkpoint, load_checkpoint, rename_legacy_state_dict
from scsnet.io import load_stereo_pair, matching_pairs, save_disparity


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.acfe = nn.Linear(2, 1)


class InferenceTests(unittest.TestCase):
    def setUp(self):
        self.workspace = tempfile.TemporaryDirectory(prefix='scsnet_')
        self.root = Path(self.workspace.name)

    def tearDown(self):
        self.workspace.cleanup()

    def image_pair(self, width=19, height=13):
        pixels = np.zeros((height, width, 3), dtype=np.uint8)
        pixels[..., 0] = 255
        left, right = self.root / 'left.png', self.root / 'right.png'
        Image.fromarray(pixels).save(str(left))
        Image.fromarray(pixels).save(str(right))
        return left, right

    def test_legacy_rename_is_complete_and_immutable(self):
        tensor = torch.tensor([1.0])
        state = {'module.feature_extraction.blocks.0.1.weights': tensor,
                 'module.volume_postprocess.0.0.2.weight': tensor,
                 'module.refine_spn.final_conv.bias': tensor}
        converted = rename_legacy_state_dict(state)
        self.assertEqual(set(converted), {'acfe.blocks.0.1.omega', 'cost_aggregation.0.0.2.weight', 'ssr.final_conv.bias'})
        self.assertIn('module.feature_extraction.blocks.0.1.weights', state)
        self.assertIs(converted['acfe.blocks.0.1.omega'], tensor)

    def test_rename_rejects_collisions_and_non_tensor(self):
        with self.assertRaises(ValueError):
            rename_legacy_state_dict({'module.acfe.weight': torch.ones(1), 'acfe.weight': torch.ones(1)})
        with self.assertRaises((TypeError, ValueError)):
            rename_legacy_state_dict({'acfe.weight': 'not a tensor'})

    def test_strict_load_legacy_and_plain_states(self):
        model = TinyModel()
        desired = {key: torch.full_like(value, 3) for key, value in model.state_dict().items()}
        old = {'module.' + key.replace('acfe.', 'feature_extraction.'): value for key, value in desired.items()}
        checkpoint = self.root / 'old.pt'
        torch.save({'state_dict': old, 'optimizer': {'unused': torch.tensor(4)}}, str(checkpoint))
        load_checkpoint(model, checkpoint)
        self.assertTrue(all(torch.equal(value, desired[key]) for key, value in model.state_dict().items()))
        torch.save(desired, str(checkpoint))
        load_checkpoint(model, checkpoint)

    def test_bad_state_is_rejected_before_model_mutation(self):
        model = TinyModel()
        original = {key: value.clone() for key, value in model.state_dict().items()}
        bad = {'acfe.weight': torch.full_like(model.acfe.weight, 9), 'acfe.bias': torch.ones(2)}
        checkpoint = self.root / 'bad.pt'
        torch.save(bad, str(checkpoint))
        with self.assertRaises((ValueError, RuntimeError)):
            load_checkpoint(model, checkpoint)
        self.assertTrue(all(torch.equal(value, original[key]) for key, value in model.state_dict().items()))
        for bad in ({'acfe.weight': model.acfe.weight}, dict(original, extra=torch.ones(1))):
            torch.save(bad, str(checkpoint))
            with self.assertRaises((ValueError, RuntimeError)):
                load_checkpoint(model, checkpoint)

    def test_rgb_normalization_top_right_padding(self):
        left, right = self.image_pair()
        i_l, i_r, size = load_stereo_pair(left, right, pad_height=16, pad_width=24)
        self.assertEqual(i_l.shape, (1, 3, 16, 24))
        self.assertEqual(size, (13, 19))
        torch.testing.assert_close(i_l, i_r)
        self.assertEqual(float(i_l[:, :, :3, :].abs().sum()), 0.0)
        self.assertEqual(float(i_l[:, :, :, 19:].abs().sum()), 0.0)
        expected = torch.tensor([(1-.485)/.229, (0-.456)/.224, (0-.406)/.225])
        torch.testing.assert_close(i_l[0, :, -1, 0], expected)

    def test_default_padding_and_invalid_sizes(self):
        left, right = self.image_pair()
        i_l, _, _ = load_stereo_pair(left, right)
        self.assertEqual(i_l.shape[-2:], (384, 1248))
        for height, width in ((8, 24), (16, 16), (17, 24)):
            with self.assertRaises(ValueError):
                load_stereo_pair(left, right, height, width)
        Image.new('RGB', (20, 13)).save(str(right))
        with self.assertRaises(ValueError):
            load_stereo_pair(left, right, 16, 24)

    def test_matching_pairs_and_invalid_directory(self):
        left, right = self.root / 'left', self.root / 'right'
        left.mkdir(); right.mkdir()
        for name in ('b.jpg', 'a.png'):
            Image.new('RGB', (8, 8)).save(str(left / name))
            Image.new('RGB', (8, 8)).save(str(right / name))
        pairs = matching_pairs(left, right)
        self.assertEqual([p[0].name for p in pairs], ['a.png', 'b.jpg'])
        (right / 'b.jpg').unlink()
        with self.assertRaises(ValueError):
            matching_pairs(left, right)
        with self.assertRaises(ValueError):
            matching_pairs(left / 'a.png', right)
        (left / 'a.jpg').write_bytes((left / 'a.png').read_bytes())
        with self.assertRaises(ValueError):
            matching_pairs(left, right)

    def test_save_raw_and_uint16_disparity_without_overwrite(self):
        values = torch.tensor([[[[-1., 1.1234], [191.999, 500.]]]])
        save_disparity(values, self.root / 'result', (2, 2))
        raw = np.load(str(self.root / 'result.npy'))
        image = np.asarray(Image.open(str(self.root / 'result.png')))
        self.assertEqual(raw.dtype, np.float32)
        np.testing.assert_array_equal(raw, values.numpy()[0, 0])
        np.testing.assert_array_equal(image, (np.clip(raw, 0, 192)*256).astype(np.uint16))
        with self.assertRaises(FileExistsError):
            save_disparity(values, self.root / 'result', (2, 2))
        with self.assertRaises(ValueError):
            save_disparity(torch.full_like(values, float('nan')), self.root / 'nan', (2, 2))

    def test_safe_loading_rejects_arbitrary_objects(self):
        checkpoint = self.root / 'object.pt'
        torch.save({'state_dict': TinyModel()}, str(checkpoint))
        with self.assertRaisesRegex(ValueError, 'Safe checkpoint loading failed'):
            load_checkpoint(TinyModel(), checkpoint)
        torch.save([torch.ones(1)], str(checkpoint))
        with self.assertRaises(ValueError):
            load_checkpoint(TinyModel(), checkpoint)
        with self.assertRaises(FileNotFoundError):
            load_checkpoint(TinyModel(), self.root / 'missing.pt')

    def test_saved_config_must_match_model(self):
        from scsnet.config import SCSNetConfig
        from scsnet.model import SCSNet
        config = SCSNetConfig(c=8, G=8, stage_search_ranges=(4, 2, 2), sse_blocks=1,
                            aggregation_layers=1, aggregation_channels=2,
                            aggregation_growth=(1, 1, 1), ssr_channels=2)
        model = SCSNet(config)
        checkpoint = self.root / 'metadata.pt'
        metadata = dict(asdict(config), stage_search_ranges=(5, 2, 2))
        torch.save({'state_dict': model.state_dict(), 'model_config': metadata}, str(checkpoint))
        with self.assertRaisesRegex(ValueError, 'model_config does not match'):
            load_checkpoint(model, checkpoint)
        for metadata in ('invalid', {'unknown': 3}, {'stage_search_ranges': 5}):
            torch.save({'state_dict': model.state_dict(), 'model_config': metadata}, str(checkpoint))
            with self.assertRaises(ValueError):
                load_checkpoint(model, checkpoint)

    def test_cli_validates_user_inputs(self):
        from scsnet.cli import main
        left, right = self.image_pair()
        base = ['--checkpoint', str(self.root / 'none.pt'), '--left', str(left),
                '--right', str(right), '--output', str(self.root / 'output')]
        self.assertEqual(main(base + ['--threads', '0']), 1)
        self.assertEqual(main(base + ['--device', 'meta']), 1)
        with patch('torch.cuda.is_available', return_value=False):
            self.assertEqual(main(base + ['--device', 'cuda:0']), 1)
        self.assertEqual(main(base + ['--device', 'cpu']), 1)

    def test_checkpoint_conversion_and_cli(self):
        from scsnet.config import SCSNetConfig
        from scsnet.model import SCSNet
        config = SCSNetConfig(c=8, G=8, stage_search_ranges=(4, 2, 2), sse_blocks=1,
                            aggregation_layers=1, aggregation_channels=2,
                            aggregation_growth=(1, 1, 1), ssr_channels=2)
        model = SCSNet(config)
        source, target = self.root / 'legacy.pt', self.root / 'public.pt'
        old = {('module.' + key.replace('acfe.', 'feature_extraction.').replace('cost_aggregation.', 'volume_postprocess.').replace('ssr.', 'refine_spn.').replace('.omega', '.weights')): value for key, value in model.state_dict().items()}
        torch.save({'state_dict': old, 'optimizer': {'unused': 7}}, str(source))
        convert_checkpoint(source, target, config)
        payload = torch.load(str(target), weights_only=True)
        self.assertEqual(set(payload), {'state_dict', 'model_config'})
        self.assertEqual(payload['model_config'], asdict(config))
        with self.assertRaises(FileExistsError):
            convert_checkpoint(source, target, config)
        with self.assertRaises(ValueError):
            convert_checkpoint(source, source, config)
        left, right = self.image_pair()
        command = [sys.executable, str(Path(__file__).resolve().parents[1] / 'infer.py'),
                   '--checkpoint', str(target), '--left', str(left), '--right', str(right),
                   '--output', str(self.root / 'output'), '--device', 'cpu',
                   '--pad-height', '16', '--pad-width', '24', '--save-stages']
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(np.load(str(self.root / 'output' / 'left.npy')).shape, (13, 19))
        self.assertTrue((self.root / 'output' / 'stage4' / 'left.png').is_file())
        from scsnet.cli import main
        direct = command[2:]
        direct[direct.index('--output') + 1] = str(self.root / 'direct_output')
        self.assertEqual(main(direct), 0)
        np.testing.assert_allclose(np.load(str(self.root / 'output' / 'left.npy')),
                                   np.load(str(self.root / 'direct_output' / 'left.npy')))
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(result.returncode, 0)
        bad_command = command[:]
        bad_command[bad_command.index('--output') + 1] = str(left.parent)
        result = subprocess.run(bad_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        self.assertNotEqual(result.returncode, 0)


if __name__ == '__main__':
    unittest.main()
