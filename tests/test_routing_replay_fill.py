"""fill_routing_replay's layer-major copy: the same per-layer routes as slicing each layer, in order."""
import unittest

import torch

from slime.utils.routing_replay import layer_major_routes, torch_threads

PIN = torch.cuda.is_available()


class LayerMajorRoutesTests(unittest.TestCase):
    def setUp(self):
        g = torch.Generator().manual_seed(0)
        self.routes = torch.randint(0, 32, (1000, 25, 4), generator=g, dtype=torch.int32)

    def check(self, layer_ids):
        layers = layer_major_routes(self.routes, layer_ids, pin_memory=PIN)
        self.assertEqual(len(layers), len(layer_ids))
        base = layers[0].untyped_storage().data_ptr() if layers else None
        for layer_id, experts in zip(layer_ids, layers):
            self.assertTrue(torch.equal(experts, self.routes[:, layer_id]))  # what the old per-layer copy recorded
            self.assertTrue(experts.is_contiguous())
            self.assertEqual(experts.untyped_storage().data_ptr(), base)  # one buffer for the micro-batch
            self.assertEqual(experts.is_pinned(), PIN)

    def test_contiguous_moe_layers(self):
        self.check(list(range(2, 25)))  # Chimera: two dense layers, then MoE

    def test_strided_moe_layers_and_none(self):
        self.check([1, 3, 5, 11, 24])
        self.assertEqual(layer_major_routes(self.routes, []), [])

    def test_thread_limit_is_restored(self):
        before = torch.get_num_threads()
        with torch_threads(1):
            self.assertEqual(torch.get_num_threads(), 1)
        self.assertEqual(torch.get_num_threads(), before)


if __name__ == '__main__':
    unittest.main()
