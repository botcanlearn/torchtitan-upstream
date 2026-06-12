# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Checkpoint interop tests for FusedQKVLinear.

FusedQKVLinear stores a single fused ``wqkv`` parameter but checkpoints in the
stock ``QKVLinear`` layout (``wq.weight`` / ``wk.weight`` / ``wv.weight``) via
state_dict hooks, so its checkpoints round-trip with the non-fused module.

Tests cover:
- Module-level save/load interop (FusedQKVLinear ↔ QKVLinear)
- Full-model state_dict round-trip and cross-compatibility (Llama3)
- HF adapter (to_hf / from_hf) with fused QKV
- DCP-like ModelStateDictWrapper pattern

All tests run on CPU.
"""

import re
import unittest

import torch
from torchtitan.models.common.attention import FusedQKVLinear, QKVLinear
from torchtitan.models.common.nn_modules import Linear
from torchtitan.models.llama3 import llama3_configs
from torchtitan.models.llama3.model import Llama3Model
from torchtitan.models.llama3.state_dict_adapter import Llama3StateDictAdapter

_DIM = 16
_N_HEADS = 4
_N_KV_HEADS = 2
_HEAD_DIM = 8
_HPK = _N_HEADS // _N_KV_HEADS  # heads_per_kv = 2
_R_DIM = _HPK + 2  # 4
_WQKV_OUT = (_N_HEADS + 2 * _N_KV_HEADS) * _HEAD_DIM  # 64
_WQ_OUT = _N_HEADS * _HEAD_DIM  # 32
_WK_OUT = _N_KV_HEADS * _HEAD_DIM  # 16


def _build_fused(with_bias: bool = False) -> FusedQKVLinear:
    fused = FusedQKVLinear.Config(
        head_dim=_HEAD_DIM,
        n_heads=_N_HEADS,
        n_kv_heads=_N_KV_HEADS,
        wqkv=Linear.Config(in_features=_DIM, out_features=_WQKV_OUT, bias=with_bias),
    ).build()
    with torch.no_grad():
        fused.wqkv.weight.copy_(torch.randn(_WQKV_OUT, _DIM))
        if with_bias:
            fused.wqkv.bias.copy_(torch.randn(_WQKV_OUT))
    return fused


def _build_stock(with_bias: bool = False) -> QKVLinear:
    stock = QKVLinear.Config(
        head_dim=_HEAD_DIM,
        wq=Linear.Config(in_features=_DIM, out_features=_WQ_OUT, bias=with_bias),
        wkv=Linear.Config(in_features=_DIM, out_features=_WK_OUT, bias=with_bias),
    ).build()
    with torch.no_grad():
        for p in stock.parameters():
            p.copy_(torch.randn_like(p))
    return stock


class TestFusedQKVCheckpointInterop(unittest.TestCase):
    def test_saves_in_stock_layout(self):
        """state_dict() emits the stock wq/wk/wv layout, not the fused wqkv."""
        fused = _build_fused()
        sd = fused.state_dict()
        self.assertEqual(set(sd), {"wq.weight", "wk.weight", "wv.weight"})

        # Verify Q = wqkv[:, :heads_per_kv]
        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        expected_wq = wqkv[:, :_HPK, :, :].reshape(-1, _DIM)
        expected_wk = wqkv[:, _HPK : _HPK + 1, :, :].reshape(-1, _DIM)
        expected_wv = wqkv[:, _HPK + 1 : _HPK + 2, :, :].reshape(-1, _DIM)
        self.assertTrue(torch.equal(sd["wq.weight"], expected_wq))
        self.assertTrue(torch.equal(sd["wk.weight"], expected_wk))
        self.assertTrue(torch.equal(sd["wv.weight"], expected_wv))

    def test_saves_bias_in_stock_layout(self):
        """state_dict() with bias emits separate q/k/v biases too."""
        fused = _build_fused(with_bias=True)
        sd = fused.state_dict()
        self.assertEqual(
            set(sd),
            {"wq.weight", "wk.weight", "wv.weight", "wq.bias", "wk.bias", "wv.bias"},
        )

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        b_3d = fused.wqkv.bias.reshape(n_kv, _R_DIM, _HEAD_DIM)
        expected_bq = b_3d[:, :_HPK, :].reshape(-1)
        expected_bk = b_3d[:, _HPK : _HPK + 1, :].reshape(-1)
        expected_bv = b_3d[:, _HPK + 1 : _HPK + 2, :].reshape(-1)
        self.assertTrue(torch.equal(sd["wq.bias"], expected_bq))
        self.assertTrue(torch.equal(sd["wk.bias"], expected_bk))
        self.assertTrue(torch.equal(sd["wv.bias"], expected_bv))

    def test_fused_checkpoint_loads_into_stock(self):
        """A fused checkpoint loads into the stock QKVLinear, weights + output."""
        fused = _build_fused()
        stock = _build_stock()
        stock.load_state_dict(fused.state_dict())

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        self.assertTrue(
            torch.equal(stock.wq.weight, wqkv[:, :_HPK, :, :].reshape(-1, _DIM))
        )
        self.assertTrue(
            torch.equal(
                stock.wk.weight, wqkv[:, _HPK : _HPK + 1, :, :].reshape(-1, _DIM)
            )
        )
        self.assertTrue(
            torch.equal(
                stock.wv.weight, wqkv[:, _HPK + 1 : _HPK + 2, :, :].reshape(-1, _DIM)
            )
        )

        x = torch.randn(1, 4, _DIM)
        fq, fk, fv = fused(x)
        sq, sk, sv = stock(x)
        self.assertTrue(torch.allclose(fq, sq, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fk, sk, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fv, sv, atol=1e-5, rtol=1e-5))

    def test_stock_checkpoint_loads_into_fused(self):
        """A stock checkpoint loads into FusedQKVLinear, weights + output."""
        stock = _build_stock()
        fused = _build_fused()
        fused.load_state_dict(stock.state_dict())

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        self.assertTrue(
            torch.equal(wqkv[:, :_HPK, :, :].reshape(-1, _DIM), stock.wq.weight)
        )
        self.assertTrue(
            torch.equal(
                wqkv[:, _HPK : _HPK + 1, :, :].reshape(-1, _DIM), stock.wk.weight
            )
        )
        self.assertTrue(
            torch.equal(
                wqkv[:, _HPK + 1 : _HPK + 2, :, :].reshape(-1, _DIM), stock.wv.weight
            )
        )

        x = torch.randn(1, 4, _DIM)
        fq, fk, fv = fused(x)
        sq, sk, sv = stock(x)
        self.assertTrue(torch.allclose(fq, sq, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fk, sk, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fv, sv, atol=1e-5, rtol=1e-5))

    def test_stock_checkpoint_with_bias_loads_into_fused(self):
        """Stock checkpoint with bias loads into FusedQKVLinear, biases preserved."""
        stock = _build_stock(with_bias=True)
        fused = _build_fused(with_bias=True)
        fused.load_state_dict(stock.state_dict())

        n_kv = _WQKV_OUT // (_R_DIM * _HEAD_DIM)
        wqkv_w = fused.wqkv.weight.reshape(n_kv, _R_DIM, _HEAD_DIM, _DIM)
        wqkv_b = fused.wqkv.bias.reshape(n_kv, _R_DIM, _HEAD_DIM)
        self.assertTrue(
            torch.equal(wqkv_w[:, :_HPK, :, :].reshape(-1, _DIM), stock.wq.weight)
        )
        self.assertTrue(torch.equal(wqkv_b[:, :_HPK, :].reshape(-1), stock.wq.bias))
        self.assertTrue(
            torch.equal(wqkv_b[:, _HPK : _HPK + 1, :].reshape(-1), stock.wk.bias)
        )
        self.assertTrue(
            torch.equal(wqkv_b[:, _HPK + 1 : _HPK + 2, :].reshape(-1), stock.wv.bias)
        )

        x = torch.randn(1, 4, _DIM)
        fq, fk, fv = fused(x)
        sq, sk, sv = stock(x)
        self.assertTrue(torch.allclose(fq, sq, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fk, sk, atol=1e-5, rtol=1e-5))
        self.assertTrue(torch.allclose(fv, sv, atol=1e-5, rtol=1e-5))

    def test_fused_roundtrip(self):
        """fused -> save -> load into a fresh fused preserves wqkv exactly."""
        src = _build_fused()
        dst = _build_fused()
        dst.load_state_dict(src.state_dict())
        self.assertTrue(torch.equal(dst.wqkv.weight, src.wqkv.weight))

    def test_fused_roundtrip_with_bias(self):
        """fused -> save -> load into a fresh fused preserves wqkv + bias."""
        src = _build_fused(with_bias=True)
        dst = _build_fused(with_bias=True)
        dst.load_state_dict(src.state_dict())
        self.assertTrue(torch.equal(dst.wqkv.weight, src.wqkv.weight))
        self.assertTrue(torch.equal(dst.wqkv.bias, src.wqkv.bias))

    def test_strict_load_reports_missing(self):
        """strict load flags a genuinely incomplete checkpoint."""
        fused = _build_fused()
        with self.assertRaises(RuntimeError):
            fused.load_state_dict({"wk.weight": torch.randn(_WK_OUT, _DIM)})


def _build_llama3(fuse_qkv: bool) -> Llama3Model:
    if fuse_qkv:
        config = llama3_configs["debugmodel_fused_qkv"](attn_backend="flex")
    else:
        config = llama3_configs["debugmodel"](attn_backend="flex")
    model = config.build()
    model.eval()
    return model


def _assert_state_dicts_equal(tc: unittest.TestCase, sd1: dict, sd2: dict):
    tc.assertEqual(set(sd1.keys()), set(sd2.keys()))
    for k in sorted(sd1.keys()):
        tc.assertTrue(
            torch.equal(sd1[k], sd2[k]),
            f"Mismatch at {k}: max diff="
            f"{torch.max(torch.abs(sd1[k].float() - sd2[k].float())).item()}",
        )


class TestFusedQKVFullModelInterop(unittest.TestCase):
    """Full-model DCP-path interop: fused ↔ stock via state_dict."""

    def test_fused_model_state_dict_keys(self):
        """Fused model state_dict has wq/wk/wv keys, not wqkv."""
        model = _build_llama3(fuse_qkv=True)
        sd = model.state_dict()
        qkv_keys = [k for k in sd if "qkv_linear" in k]
        for k in qkv_keys:
            self.assertNotIn("wqkv", k)
            self.assertTrue(any(sub in k for sub in ("wq.", "wk.", "wv.")))

    def test_fused_to_stock_interop(self):
        """Fused model state_dict loads into stock model."""
        fused_model = _build_llama3(fuse_qkv=True)
        stock_model = _build_llama3(fuse_qkv=False)
        stock_model.load_state_dict(fused_model.state_dict())
        _assert_state_dicts_equal(
            self, fused_model.state_dict(), stock_model.state_dict()
        )

    def test_stock_to_fused_interop(self):
        """Stock model state_dict loads into fused model."""
        stock_model = _build_llama3(fuse_qkv=False)
        fused_model = _build_llama3(fuse_qkv=True)
        fused_model.load_state_dict(stock_model.state_dict())
        _assert_state_dicts_equal(
            self, stock_model.state_dict(), fused_model.state_dict()
        )

    def test_strict_false_with_extra_keys(self):
        """load_state_dict(strict=False) ignores non-model keys (DCP pattern)."""
        model = _build_llama3(fuse_qkv=True)
        sd = model.state_dict()
        sd["optimizer.some_key"] = torch.zeros(1)
        model2 = _build_llama3(fuse_qkv=True)
        model2.load_state_dict(sd, strict=False)
        del sd["optimizer.some_key"]
        _assert_state_dicts_equal(self, sd, model2.state_dict())


class TestFusedQKVHFInterop(unittest.TestCase):
    """HF adapter interop: fused ↔ stock via to_hf/from_hf."""

    def _get_adapter(self, fuse_qkv: bool) -> Llama3StateDictAdapter:
        if fuse_qkv:
            config = llama3_configs["debugmodel_fused_qkv"](attn_backend="flex")
        else:
            config = llama3_configs["debugmodel"](attn_backend="flex")
        return Llama3StateDictAdapter(config, hf_assets_path=None)

    def test_fused_hf_roundtrip(self):
        """Fused model → to_hf → from_hf → fused model preserves weights."""
        model = _build_llama3(fuse_qkv=True)
        adapter = self._get_adapter(fuse_qkv=True)

        sd_original = model.state_dict()
        hf_sd = adapter.to_hf(sd_original)
        sd_restored = adapter.from_hf(hf_sd)

        model2 = _build_llama3(fuse_qkv=True)
        model2.load_state_dict(sd_restored)
        _assert_state_dicts_equal(self, sd_original, model2.state_dict())

    def test_fused_to_stock_interop(self):
        """Fused model → to_hf → from_hf into stock model."""
        fused_model = _build_llama3(fuse_qkv=True)
        fused_adapter = self._get_adapter(fuse_qkv=True)
        stock_adapter = self._get_adapter(fuse_qkv=False)

        hf_sd = fused_adapter.to_hf(fused_model.state_dict())
        stock_sd = stock_adapter.from_hf(hf_sd)

        stock_model = _build_llama3(fuse_qkv=False)
        stock_model.load_state_dict(stock_sd)
        _assert_state_dicts_equal(
            self, fused_model.state_dict(), stock_model.state_dict()
        )

    def test_stock_to_fused_interop(self):
        """Stock model → to_hf → from_hf into fused model."""
        stock_model = _build_llama3(fuse_qkv=False)
        stock_adapter = self._get_adapter(fuse_qkv=False)
        fused_adapter = self._get_adapter(fuse_qkv=True)

        hf_sd = stock_adapter.to_hf(stock_model.state_dict())
        fused_sd = fused_adapter.from_hf(hf_sd)

        fused_model = _build_llama3(fuse_qkv=True)
        fused_model.load_state_dict(fused_sd)
        _assert_state_dicts_equal(
            self, stock_model.state_dict(), fused_model.state_dict()
        )


class TestConvertToHFRegression(unittest.TestCase):
    """Regression for convert_to_hf.py: DCP fused checkpoint → HF → stock model."""

    def test_convert_to_hf_fused_loads_into_stock(self):
        """ModelWrapper(fused) → state_dict → to_hf → from_hf → stock model."""
        from torchtitan.components.checkpoint import ModelWrapper

        model = _build_llama3(fuse_qkv=True)
        wrapper = ModelWrapper(model)
        state_dict = wrapper.state_dict()

        fused_config = llama3_configs["debugmodel_fused_qkv"](attn_backend="flex")
        adapter = Llama3StateDictAdapter(fused_config, hf_assets_path=None)
        hf_sd = adapter.to_hf(state_dict)

        for k in hf_sd:
            self.assertNotIn("wqkv", k)
            self.assertNotIn("qkv_linear", k)

        stock_config = llama3_configs["debugmodel"](attn_backend="flex")
        stock_adapter = Llama3StateDictAdapter(stock_config, hf_assets_path=None)
        stock_sd = stock_adapter.from_hf(hf_sd)

        stock_model = _build_llama3(fuse_qkv=False)
        stock_model.load_state_dict(stock_sd)
        _assert_state_dicts_equal(self, model.state_dict(), stock_model.state_dict())


if __name__ == "__main__":
    unittest.main()
