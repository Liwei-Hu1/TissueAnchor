import unittest
from types import SimpleNamespace

import torch

from tissueanchor.engines.infer import default_adaptation_rays
from tissueanchor.engines.inputs import validate_cache
from tissueanchor.metrics.image import masked_inputs, ssim
from tissueanchor.models.shared_adapter import SharedQueryNeRF


class CoreTest(unittest.TestCase):
    def test_parameter_count(self):
        self.assertEqual(sum(p.numel() for p in SharedQueryNeRF().parameters()), 472173)

    def test_historical_ray_budgets(self):
        self.assertEqual(default_adaptation_rays('stereomis', 100), 1024)
        self.assertEqual(default_adaptation_rays('stereomis', 1000), 512)
        self.assertEqual(default_adaptation_rays('drends', 1000), 1024)

    def test_masked_metrics(self):
        rgb = torch.arange(16*16*3).reshape(16, 16, 3).to(torch.uint8)
        mask = torch.ones(16, 16, dtype=torch.bool)
        mask[:5] = False
        alt = rgb.clone()
        alt[:5] = 0
        p, x, y = masked_inputs(rgb, alt, mask)
        self.assertAlmostEqual(float(p), 120, places=4)
        self.assertAlmostEqual(float(ssim(x, y)), 1, places=6)
        with self.assertRaises(ValueError):
            masked_inputs(rgb, alt, torch.zeros_like(mask))

    def test_cache_validation(self):
        cache = dict(schema='shared_query_cache_v1', tracks=torch.zeros(48, 1024, 3),
                     latent=torch.zeros(1024, 1280), reliability=torch.ones(48, 1024),
                     manifest={'frame_ids': ['1'], 'source': 'old'})
        scene = SimpleNamespace(name='synthetic', manifest=lambda: {'frame_ids': ['1'], 'source': 'new'})
        validate_cache(cache, scene)
        cache['tracks'] = cache['tracks'][:1]
        with self.assertRaises(ValueError):
            validate_cache(cache, scene)


if __name__ == '__main__':
    unittest.main()
