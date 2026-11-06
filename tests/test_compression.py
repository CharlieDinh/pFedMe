import os
import sys
import unittest

import torch

# Ensure repo root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from FLAlgorithms.trainmodel.models import Mclr_Logistic, Net
from utils.compression import (
    calc_compression_stats,
    dequantize_parameters,
    dequantize_tensor,
    quantize_parameters,
    quantize_tensor,
    simulate_quantization,
)


class TestPFedMeCompression(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)

    def test_quantize_dequantize_tensor_int4(self):
        for shape in [(64,), (32, 32), (16, 1, 2, 2), (7, 13, 11)]:
            tensor = torch.randn(shape, dtype=torch.float32)
            q_dict = quantize_tensor(tensor, nbits=4, block_size=32)
            recon = dequantize_tensor(q_dict)

            self.assertEqual(recon.shape, tensor.shape)
            self.assertEqual(recon.dtype, tensor.dtype)
            cos_sim = torch.nn.functional.cosine_similarity(
                tensor.flatten(), recon.flatten(), dim=0
            )
            self.assertGreater(cos_sim.item(), 0.98)

    def test_quantize_dequantize_tensor_int8(self):
        tensor = torch.randn(64, 64, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=8, block_size=64)
        recon = dequantize_tensor(q_dict)

        self.assertEqual(recon.shape, tensor.shape)
        cos_sim = torch.nn.functional.cosine_similarity(
            tensor.flatten(), recon.flatten(), dim=0
        )
        self.assertGreater(cos_sim.item(), 0.999)

    def test_zero_and_empty_tensor(self):
        zero_tensor = torch.zeros(128, dtype=torch.float32)
        q_dict = quantize_tensor(zero_tensor, nbits=4, block_size=32)
        recon = dequantize_tensor(q_dict)

        self.assertTrue(torch.equal(zero_tensor, recon))
        self.assertFalse(torch.isnan(recon).any())
        self.assertFalse(torch.isinf(recon).any())

        empty_tensor = torch.empty((0, 10), dtype=torch.float32)
        q_empty = quantize_tensor(empty_tensor, nbits=4, block_size=32)
        recon_empty = dequantize_tensor(q_empty)
        self.assertEqual(recon_empty.numel(), 0)

    def test_sub_byte_bit_packing(self):
        numel = 256
        tensor = torch.randn(numel, dtype=torch.float32)
        q_dict = quantize_tensor(tensor, nbits=4, block_size=64)

        self.assertEqual(q_dict["data"].dtype, torch.uint8)
        self.assertEqual(q_dict["data"].numel(), numel // 2)

    def test_quantize_model_module(self):
        model = Mclr_Logistic(input_dim=784, output_dim=10)
        q_dict = quantize_parameters(model, nbits=4, block_size=32)

        target_model = Mclr_Logistic(input_dim=784, output_dim=10)
        dequantize_parameters(q_dict, target_model=target_model)

        for p_orig, p_target in zip(model.parameters(), target_model.parameters()):
            self.assertEqual(p_orig.shape, p_target.shape)
            cos = torch.nn.functional.cosine_similarity(
                p_orig.flatten(), p_target.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_quantize_parameters_list(self):
        grads = [torch.randn(10, 5), torch.randn(10), torch.randn(4, 2, 2)]
        q_list = quantize_parameters(grads, nbits=4, block_size=16)
        restored = dequantize_parameters(q_list)

        self.assertEqual(len(restored), len(grads))
        for orig, recon in zip(grads, restored):
            self.assertEqual(orig.shape, recon.shape)
            cos = torch.nn.functional.cosine_similarity(
                orig.flatten(), recon.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_error_feedback_residuals_list(self):
        residuals = []
        true_accum = torch.zeros(64)
        quant_accum = torch.zeros(64)

        for _ in range(10):
            step_delta = torch.randn(64) * 0.05
            true_accum += step_delta

            param_list = [step_delta]
            q_list = quantize_parameters(
                param_list, nbits=4, block_size=32, residuals=residuals
            )
            recon_list = dequantize_parameters(q_list)
            quant_accum += recon_list[0]

        drift = torch.norm(true_accum - quant_accum).item()
        self.assertLess(drift, 0.05)

    def test_simulate_quantization(self):
        model = Net()
        simulated = simulate_quantization(model, nbits=4, block_size=64)

        for p_orig, p_sim in zip(model.parameters(), simulated.parameters()):
            self.assertEqual(p_orig.shape, p_sim.shape)
            cos = torch.nn.functional.cosine_similarity(
                p_orig.flatten(), p_sim.flatten(), dim=0
            )
            self.assertGreater(cos.item(), 0.98)

    def test_calc_compression_stats(self):
        model = Mclr_Logistic(input_dim=784, output_dim=10)
        q_dict = quantize_parameters(model, nbits=4, block_size=64)
        stats = calc_compression_stats(model, q_dict)

        self.assertGreater(stats["compression_ratio"], 6.0)
        self.assertGreater(stats["bandwidth_savings_pct"], 80.0)
        self.assertGreater(stats["raw_bytes"], stats["compressed_bytes"])

    def test_invalid_inputs(self):
        tensor = torch.randn(32)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=5)
        with self.assertRaises(ValueError):
            quantize_tensor(tensor, nbits=4, block_size=7)


if __name__ == "__main__":
    unittest.main()
