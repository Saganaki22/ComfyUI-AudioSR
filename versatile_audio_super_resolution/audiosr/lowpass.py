from scipy.signal import butter, lfilter
import torch
from scipy import signal
import librosa
import numpy as np

from scipy.signal import sosfiltfilt
from scipy.signal import butter, cheby1, cheby2, ellip, bessel


def _safe_resample(data: np.ndarray, target_length: int) -> np.ndarray:
    """Resample ``data`` to ``target_length`` using scipy's FFT-based path.

    We deliberately avoid :func:`scipy.signal.resample_poly` here because, on
    Python 3.13+ / newer scipy, ``resample_poly`` internally calls
    :func:`array_api_compat.common._helpers.is_array_api_obj`, which iterates
    over a list of registered array-API libraries (numpy, torch, cupy, jax,
    dask, …) and does ``getattr(mod, "Array")`` on each. If a library is
    partially registered in ``sys.modules`` (e.g. ``mod is None`` — common in
    ComfyUI bundles that import a stub jax shim), that ``getattr`` raises
    ``AttributeError: 'NoneType' object has no attribute 'Array'``.

    The FFT path in :func:`scipy.signal.resample` is functionally equivalent
    for the "downsample to a low sampling rate, then upsample to the original
    rate" pattern used in :func:`stft_hard_lowpass` (an ideal lowpass via
    periodic extension), and it does not touch the array-API dispatch layer.

    Args:
        data: 1-D numpy array of samples to resample.
        target_length: Desired output length in samples (>= 1).

    Returns:
        1-D numpy array of length ``target_length``.

    Raises:
        ValueError: If ``data`` is not 1-D or ``target_length`` < 1.
    """
    if data.ndim != 1:
        raise ValueError(
            f"_safe_resample expects a 1-D array, got shape {data.shape!r}"
        )
    if target_length < 1:
        raise ValueError(
            f"_safe_resample target_length must be >= 1, got {target_length}"
        )
    return signal.resample(data, target_length)


def align_length(x=None, y=None, Lx=None):
    """align the length of y to that of x

    Args:
        x (np.array): reference signal
        y (np.array): the signal needs to be length aligned

    Return:
        yy (np.array): signal with the same length as x
    """
    assert y is not None

    if Lx is None:
        Lx = len(x)
    Ly = len(y)

    if Lx == Ly:
        return y
    elif Lx > Ly:
        # pad y with zeros
        return np.pad(y, (0, Lx - Ly), mode="constant")
    else:
        # cut y
        return y[:Lx]


def bandpass_filter(x, lowcut, highcut, fs, order, ftype):
    """process input signal x using bandpass filter

    Args:
        x (np.array): input signal
        lowcut (float): low cutoff frequency
        highcut (float): high cutoff frequency
        order (int): the order of filter
        ftype (string): type of filter
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

    Return:
        y (np.array): filtered signal
    """
    nyq = 0.5 * fs
    lo = lowcut / nyq
    hi = highcut / nyq

    if ftype == "butter":
        # b, a = butter(order, [lo, hi], btype='band')
        sos = butter(order, [lo, hi], btype="band", output="sos")
    elif ftype == "cheby1":
        sos = cheby1(order, 0.1, [lo, hi], btype="band", output="sos")
    elif ftype == "cheby2":
        sos = cheby2(order, 60, [lo, hi], btype="band", output="sos")
    elif ftype == "ellip":
        sos = ellip(order, 0.1, 60, [lo, hi], btype="band", output="sos")
    elif ftype == "bessel":
        sos = bessel(order, [lo, hi], btype="band", output="sos")
    else:
        raise Exception(f"The bandpass filter {ftype} is not supported!")

    # y = lfilter(b, a, x)
    y = sosfiltfilt(sos, x)

    if len(y) != len(x):
        y = align_length(x, y)
    return y


def lowpass_filter(x, highcut, fs, order, ftype):
    """process input signal x using lowpass filter

    Args:
        x (np.array): input signal
        highcut (float): high cutoff frequency
        order (int): the order of filter
        ftype (string): type of filter
            ['butter', 'cheby1', 'cheby2', 'ellip', 'bessel']

    Return:
        y (np.array): filtered signal
    """
    nyq = 0.5 * fs
    hi = highcut / nyq

    if ftype == "butter":
        sos = butter(order, hi, btype="low", output="sos")
    elif ftype == "cheby1":
        sos = cheby1(order, 0.1, hi, btype="low", output="sos")
    elif ftype == "cheby2":
        sos = cheby2(order, 60, hi, btype="low", output="sos")
    elif ftype == "ellip":
        sos = ellip(order, 0.1, 60, hi, btype="low", output="sos")
    elif ftype == "bessel":
        sos = bessel(order, hi, btype="low", output="sos")
    else:
        raise Exception(f"The lowpass filter {ftype} is not supported!")

    y = sosfiltfilt(sos, x)

    if len(y) != len(x):
        y = align_length(x, y)

    y_len = len(y)

    y = stft_hard_lowpass(y, hi, fs_ori=fs)

    y = sosfiltfilt(sos, y)

    if len(y) != y_len:
        y = align_length(y=y, Lx=y_len)

    return y


def stft_hard_lowpass(data: np.ndarray, lowpass_ratio: float, fs_ori: int = 44100) -> np.ndarray:
    """Hard lowpass via downsample→upsample (ideal lowpass / periodic extension).

    Replaces the previous ``resample_poly``-based implementation to avoid
    ``AttributeError: 'NoneType' object has no attribute 'Array'`` raised by
    ``scipy.signal.resample_poly``'s internal ``array_api_compat`` dispatch
    on Python 3.13+ when a partially-registered array-API library (e.g. jax)
    is in ``sys.modules`` as ``None``. See :func:`_safe_resample` for details.

    Args:
        data: 1-D numpy array of input samples.
        lowpass_ratio: Ratio of the target low sampling rate to ``fs_ori``
            (e.g. ``0.1`` for 4.41 kHz down from 44.1 kHz).
        fs_ori: Original sampling rate in Hz. Default 44100.

    Returns:
        1-D numpy array of the same length as ``data`` containing the
        lowpass-filtered signal.
    """
    if data.ndim != 1:
        raise ValueError(
            f"stft_hard_lowpass expects a 1-D array, got shape {data.shape!r}"
        )
    if not (0.0 < lowpass_ratio <= 1.0):
        raise ValueError(
            f"lowpass_ratio must be in (0, 1], got {lowpass_ratio!r}"
        )

    fs_down = int(lowpass_ratio * fs_ori)
    n_down = max(1, int(round(len(data) * fs_down / fs_ori)))

    # Downsample to the low sampling rate, then upsample back to the original
    # length. Both hops go through scipy.signal.resample (FFT path) instead of
    # resample_poly, so the array_api_compat dispatch layer is never reached.
    y = _safe_resample(data, n_down)
    y = _safe_resample(y, len(data))

    if len(y) != len(data):
        y = align_length(data, y)
    return y


def limit(integer, high, low):
    if integer > high:
        return high
    elif integer < low:
        return low
    else:
        return int(integer)


def lowpass(data, highcut, fs, order=5, _type="butter"):
    """
    :param data: np.float32 type 1d time numpy array, (samples,) , can not be (samples, 1) !!!!!!!!!!!!
    :param highcut: cutoff frequency
    :param fs: sample rate of the original data
    :param order: order of the filter
    :return: filtered data, (samples,)
    """

    if len(list(data.shape)) != 1:
        raise ValueError(
            "Error (chebyshev_lowpass_filter): Data "
            + str(data.shape)
            + " should be type 1d time array, (samples,) , can not be (samples, 1)"
        )

    if _type in "butter":
        order = limit(order, high=10, low=2)
        return lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="butter"
        )
    elif _type in "cheby1":
        order = limit(order, high=10, low=2)
        return lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="cheby1"
        )
    elif _type in "ellip":
        order = limit(order, high=10, low=2)
        return lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="ellip"
        )
    elif _type in "bessel":
        order = limit(order, high=10, low=2)
        return lowpass_filter(
            x=data, highcut=int(highcut), fs=fs, order=order, ftype="bessel"
        )
    # elif(_type in "stft"):
    #     return stft_hard_lowpass(data, lowpass_ratio=highcut / int(fs / 2))
    # elif(_type in "stft_hard"):
    #     return stft_hard_lowpass_v0(data, lowpass_ratio=highcut / int(fs / 2))
    else:
        raise ValueError("Error: Unexpected filter type " + _type)


def bandpass(data, lowcut, highcut, fs, order=5, _type="butter"):
    """
    :param data: np.float32 type 1d time numpy array, (samples,) , can not be (samples, 1) !!!!!!!!!!!!
    :param lowcut: low cutoff frequency
    :param highcut: high cutoff frequency
    :param fs: sample rate of the original data
    :param order: order of the filter
    :param _type: type of filter
    :return: filtered data, (samples,)
    """
    if len(list(data.shape)) != 1:
        raise ValueError(
            "Error (chebyshev_lowpass_filter): Data "
            + str(data.shape)
            + " should be type 1d time array, (samples,) , can not be (samples, 1)"
        )
    if _type in "butter":
        order = limit(order, high=10, low=2)
        return bandpass_filter(
            x=data,
            lowcut=int(lowcut),
            highcut=int(highcut),
            fs=fs,
            order=order,
            ftype="butter",
        )
    elif _type in "cheby1":
        order = limit(order, high=10, low=2)
        return bandpass_filter(
            x=data,
            lowcut=int(lowcut),
            highcut=int(highcut),
            fs=fs,
            order=order,
            ftype="cheby1",
        )
    # elif(_type in "cheby2"):
    #     return bandpass_filter(x=data,lowcut=int(lowcut),highcut=int(highcut), fs=fs, order=order,ftype="cheby2")
    elif _type in "ellip":
        order = limit(order, high=10, low=2)
        return bandpass_filter(
            x=data,
            lowcut=int(lowcut),
            highcut=int(highcut),
            fs=fs,
            order=order,
            ftype="ellip",
        )
    elif _type in "bessel":
        order = limit(order, high=10, low=2)
        return bandpass_filter(
            x=data,
            lowcut=int(lowcut),
            highcut=int(highcut),
            fs=fs,
            order=order,
            ftype="bessel",
        )
    else:
        raise ValueError("Error: Unexpected filter type " + _type)
