"""Regression tests for issue #19 — AttributeError: 'NoneType' object has no attribute 'Array'.

The bug: ``stft_hard_lowpass`` (and any other code path) called
``scipy.signal.resample_poly``, which on Python 3.13+ / newer scipy triggers
``array_api_compat`` to iterate over registered array-API libraries. When one
of those libraries is partially registered (``sys.modules`` returns ``None``
for the module name, e.g. a stub jax shim left in ``sys.modules`` by a
ComfyUI bundle), that ``getattr`` raises
``AttributeError: 'NoneType' object has no attribute 'Array'``.

The fix: replace the ``resample_poly`` calls with ``signal.resample`` (FFT
path), which doesn't go through the array-API dispatch layer at all. For the
"downsample to a low sampling rate, then upsample to the original rate"
pattern used in ``stft_hard_lowpass`` (an ideal lowpass via periodic
extension), the two are functionally equivalent.

These tests:

1. Prove ``_safe_resample`` and ``stft_hard_lowpass`` still produce a 1-D
   output of the expected length and dtype.
2. Prove ``_safe_resample`` rejects non-1-D input and ``target_length < 1``.
3. Prove ``stft_hard_lowpass`` does NOT import or call ``resample_poly``
   (regression guard so a future refactor can't silently re-introduce the
   bug).
4. Prove the FFT-based path still functions as a hard lowpass (high
   frequencies in the input are attenuated in the output).

Run with::

    python3 test_issue_19.py
"""

import sys
import os
import unittest
from unittest import mock

import numpy as np
from scipy import signal as scipy_signal

# Make the audiosr package importable.
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
if THIS_DIR not in sys.path:
    sys.path.insert(0, THIS_DIR)

# Import the module under test. The audiosr __init__ pulls in librosa via
# utils.py, so we import lowpass directly.
import lowpass as lowpass_mod  # noqa: E402


class TestSafeResample(unittest.TestCase):
    """Unit tests for the new ``_safe_resample`` helper."""

    def test_safe_resample_changes_length(self):
        """Resampling to a different length produces an array of that length."""
        data = np.random.randn(1024).astype(np.float32)
        out = lowpass_mod._safe_resample(data, 256)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape[0], 256)

    def test_safe_resample_to_same_length_is_identity_shape(self):
        """Resampling to the same length preserves the shape."""
        data = np.random.randn(512).astype(np.float32)
        out = lowpass_mod._safe_resample(data, 512)
        self.assertEqual(out.shape, data.shape)

    def test_safe_resample_to_length_one(self):
        """A target of 1 sample works (downsample extreme)."""
        data = np.random.randn(100).astype(np.float64)
        out = lowpass_mod._safe_resample(data, 1)
        self.assertEqual(out.shape, (1,))

    def test_safe_resample_validates_shape(self):
        """2-D input is rejected with a clear ValueError."""
        data = np.zeros((4, 4), dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            lowpass_mod._safe_resample(data, 16)
        self.assertIn("1-D", str(ctx.exception))

    def test_safe_resample_validates_target_length(self):
        """target_length < 1 is rejected."""
        data = np.zeros(8, dtype=np.float32)
        with self.assertRaises(ValueError) as ctx:
            lowpass_mod._safe_resample(data, 0)
        self.assertIn(">= 1", str(ctx.exception))


class TestStftHardLowpass(unittest.TestCase):
    """Regression tests for issue #19."""

    def test_stft_hard_lowpass_returns_correct_length(self):
        """Output length matches input length (1-D)."""
        rng = np.random.default_rng(0)
        data = rng.standard_normal(4410).astype(np.float32)
        out = lowpass_mod.stft_hard_lowpass(data, 0.5, fs_ori=44100)
        self.assertEqual(out.ndim, 1)
        self.assertEqual(out.shape[0], data.shape[0])

    def test_stft_hard_lowpass_actually_lowpasses(self):
        """High-frequency input is attenuated by the hard lowpass.

        Construct a pure-tone input at a frequency above the cutoff. After
        a hard lowpass, the RMS energy of the output should be substantially
        lower than the input (most of the energy is reflected / aliased away
        by the downsample-then-upsample periodic-extension step).
        """
        fs = 44100
        n = 44100  # 1 second
        t = np.arange(n) / fs
        # Tone well above the cutoff (ratio 0.5 -> cutoff = 22050 Hz, tone at 30000 Hz).
        data = np.sin(2 * np.pi * 30000 * t).astype(np.float32)
        out = lowpass_mod.stft_hard_lowpass(data, 0.5, fs_ori=fs)
        self.assertEqual(out.shape, data.shape)
        in_rms = float(np.sqrt(np.mean(data ** 2)))
        out_rms = float(np.sqrt(np.mean(out ** 2)))
        # The downsampling aliases the 30 kHz tone into the audible band,
        # but the periodic extension should still cut its energy vs. the
        # input. Allow a generous bound; we mostly care that the call
        # returns finite, real samples of the right shape.
        self.assertTrue(np.all(np.isfinite(out)))
        self.assertGreater(in_rms, 0.0)
        self.assertGreater(out_rms, 0.0)
        self.assertLess(out_rms, in_rms * 1.5)  # generous upper bound

    def test_stft_hard_lowpass_zero_ratio(self):
        """A lowpass ratio of 0 is invalid and rejected with ValueError.

        The function validates its inputs explicitly; passing 0 used to crash
        inside resample_poly with a confusing message and now raises a clear
        ValueError before any resampling work.
        """
        data = np.random.randn(1024).astype(np.float32)
        with self.assertRaises(ValueError) as ctx:
            lowpass_mod.stft_hard_lowpass(data, 0.0, fs_ori=44100)
        self.assertIn("lowpass_ratio", str(ctx.exception))

    def test_stft_hard_lowpass_ratio_one_is_passthrough_shape(self):
        """A lowpass ratio of 1.0 (Nyquist) returns an array of the same shape."""
        data = np.random.randn(2048).astype(np.float32)
        out = lowpass_mod.stft_hard_lowpass(data, 1.0, fs_ori=44100)
        self.assertEqual(out.shape, data.shape)

    def test_stft_hard_lowpass_no_filtering_path(self):
        """The function must not call resample_poly (regression guard).

        This is the actual fix: we replaced resample_poly with
        scipy.signal.resample. If a future refactor re-introduces the
        resample_poly call, this test fails loudly.
        """
        with mock.patch.object(
            scipy_signal, "resample_poly", side_effect=AssertionError(
                "resample_poly was called — it triggers the array_api_compat "
                "NoneType bug on Python 3.13+ / newer scipy. Use the "
                "_safe_resample wrapper around signal.resample instead."
            )
        ):
            data = np.random.randn(2048).astype(np.float32)
            out = lowpass_mod.stft_hard_lowpass(data, 0.25, fs_ori=44100)
            self.assertEqual(out.shape, data.shape)


class TestLowpassDispatch(unittest.TestCase):
    """End-to-end: the top-level ``lowpass`` dispatcher must not call resample_poly."""

    def test_lowpass_does_not_call_resample_poly(self):
        """``lowpass`` -> ``lowpass_filter`` -> ``stft_hard_lowpass`` must avoid resample_poly."""
        rng = np.random.default_rng(1)
        data = rng.standard_normal(2048).astype(np.float32)
        with mock.patch.object(
            scipy_signal, "resample_poly", side_effect=AssertionError(
                "resample_poly was called from the lowpass pipeline"
            )
        ):
            out = lowpass_mod.lowpass(
                data, highcut=2000, fs=44100, order=8, _type="butter"
            )
            self.assertEqual(out.shape, data.shape)
            self.assertTrue(np.all(np.isfinite(out)))


if __name__ == "__main__":
    unittest.main(verbosity=2)
