"""Tests for internal deblur loss functions."""

import torch

from deblur_inr_ssrs.losses import _DeblurFFTLoss, _DeblurSSIMLoss, _DeblurTVLoss


class TestDeblurSSIMLoss:
    def test_2d_scalar_output(self):
        loss_fn = _DeblurSSIMLoss(window_size=7)
        pred = torch.rand(1, 1, 8, 8)
        target = torch.rand(1, 1, 8, 8)
        result = loss_fn(pred, target)
        assert result.ndim == 0
        assert 0 <= result.item() <= 2

    def test_3d_scalar_output(self):
        loss_fn = _DeblurSSIMLoss(window_size=3)
        pred = torch.rand(1, 1, 4, 4, 4)
        target = torch.rand(1, 1, 4, 4, 4)
        result = loss_fn(pred, target)
        assert result.ndim == 0
        assert 0 <= result.item() <= 2

    def test_identical_inputs_near_zero(self):
        loss_fn = _DeblurSSIMLoss(window_size=7)
        x = torch.rand(1, 1, 8, 8)
        result = loss_fn(x, x)
        assert result.item() < 0.01


class TestDeblurFFTLoss:
    def test_2d_normalized(self):
        loss_fn = _DeblurFFTLoss(normalize=True)
        pred = torch.rand(1, 1, 8, 8)
        target = torch.rand(1, 1, 8, 8)
        result = loss_fn(pred, target)
        assert result.ndim == 0
        assert result.item() >= 0

    def test_2d_unnormalized(self):
        loss_fn = _DeblurFFTLoss(normalize=False)
        pred = torch.rand(1, 1, 8, 8)
        target = torch.rand(1, 1, 8, 8)
        result = loss_fn(pred, target)
        assert result.ndim == 0
        assert result.item() >= 0

    def test_3d_normalized(self):
        loss_fn = _DeblurFFTLoss(normalize=True)
        pred = torch.rand(1, 1, 4, 4, 4)
        target = torch.rand(1, 1, 4, 4, 4)
        result = loss_fn(pred, target)
        assert result.ndim == 0
        assert result.item() >= 0

    def test_3d_unnormalized(self):
        loss_fn = _DeblurFFTLoss(normalize=False)
        pred = torch.rand(1, 1, 4, 4, 4)
        target = torch.rand(1, 1, 4, 4, 4)
        result = loss_fn(pred, target)
        assert result.ndim == 0
        assert result.item() >= 0


class TestDeblurTVLoss:
    def test_2d_non_negative(self):
        loss_fn = _DeblurTVLoss()
        pred = torch.rand(1, 1, 8, 8)
        result = loss_fn(pred)
        assert result.ndim == 0
        assert result.item() >= 0

    def test_3d_non_negative(self):
        loss_fn = _DeblurTVLoss()
        pred = torch.rand(1, 1, 4, 4, 4)
        result = loss_fn(pred)
        assert result.ndim == 0
        assert result.item() >= 0

    def test_2d_constant_input_zero(self):
        loss_fn = _DeblurTVLoss()
        pred = torch.ones(1, 1, 8, 8)
        result = loss_fn(pred)
        assert result.item() == 0.0

    def test_3d_constant_input_zero(self):
        loss_fn = _DeblurTVLoss()
        pred = torch.ones(1, 1, 4, 4, 4)
        result = loss_fn(pred)
        assert result.item() == 0.0
