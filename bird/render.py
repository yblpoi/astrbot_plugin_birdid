"""Image rendering helpers for the bird identification plugin.

Only Pillow and numpy are used: the AstrBot runtime container ships neither
matplotlib nor scipy, so both the bounding-box crop helper and the spectrogram
renderer are implemented from those two packages alone.

Typical use::

    crop = crop_image(photo_path, detection_box, margin_ratio=0.15)
    if crop is None:
        ...  # degenerate detection, ask the user for another photo
    png = image_to_png_bytes(crop)

    figure = spectrogram_image(samples, 32_000, title="Recording 1")
    png = image_to_png_bytes(figure)
"""

from __future__ import annotations

import functools
import math
from collections.abc import Sequence
from io import BytesIO
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

__all__ = [
    "crop_image",
    "downsample_waveform",
    "image_to_png_bytes",
    "spectrogram_image",
]

# Axes margins of the composed spectrogram frame, in pixels.
_MARGIN_LEFT = 52
_MARGIN_RIGHT = 14
_MARGIN_BOTTOM = 30
_MARGIN_TOP_PLAIN = 6
_MARGIN_TOP_TITLE = 24

_BACKGROUND = (14, 16, 20)
_AXIS_COLOR = (226, 230, 236)
_TEXT_COLOR = (222, 228, 236)
_GRID_COLOR = (78, 84, 96)

_DEFAULT_FMAX_HZ = 16_000.0
_MAX_FRAMES = 1600
# A recording whose peak stays below this level is scaled up to it, so that a
# quiet but usable clip keeps its contrast instead of collapsing to one colour.
_MIN_REFERENCE_DBFS = -60.0
_MAX_TICKS_PER_AXIS = 6
_PNG_MODES = frozenset({"1", "L", "LA", "P", "RGB", "RGBA", "I", "I;16"})

# Log-spaced axis ticks: the 1-2-5 progression reads naturally on a log axis.
_FREQUENCY_TICKS_HZ = (
    50.0,
    100.0,
    200.0,
    500.0,
    1000.0,
    2000.0,
    5000.0,
    10_000.0,
    20_000.0,
    40_000.0,
)
_TIME_TICKS_S = (
    0.1,
    0.2,
    0.25,
    0.5,
    1.0,
    2.0,
    2.5,
    5.0,
    10.0,
    15.0,
    30.0,
    60.0,
    120.0,
    300.0,
    600.0,
    900.0,
)

_ColorStop = tuple[float, tuple[int, int, int]]
_ColorStops = tuple[_ColorStop, ...]

# Nine-stop approximations of the matplotlib perceptually uniform ramps.
_COLOR_STOPS: dict[str, _ColorStops] = {
    "inferno": (
        (0.000, (0, 0, 4)),
        (0.125, (22, 11, 57)),
        (0.250, (66, 10, 104)),
        (0.375, (106, 23, 110)),
        (0.500, (147, 38, 103)),
        (0.625, (188, 55, 84)),
        (0.750, (221, 81, 58)),
        (0.875, (243, 120, 25)),
        (1.000, (252, 255, 164)),
    ),
    "viridis": (
        (0.000, (68, 1, 84)),
        (0.125, (72, 40, 120)),
        (0.250, (62, 74, 137)),
        (0.375, (49, 104, 142)),
        (0.500, (38, 130, 142)),
        (0.625, (31, 158, 137)),
        (0.750, (53, 183, 121)),
        (0.875, (109, 205, 89)),
        (1.000, (253, 231, 37)),
    ),
    "gray": (
        (0.0, (0, 0, 0)),
        (0.5, (128, 128, 128)),
        (1.0, (255, 255, 255)),
    ),
}


def crop_image(
    image_path: str | Path,
    box: Sequence[float],
    margin_ratio: float = 0.1,
) -> Image.Image | None:
    """Cut a bounding box out of an image file.

    The box is expressed in the pixel coordinate system of the decoded file as
    [x1, y1, x2, y2]; corner order does not matter. It is padded by
    margin_ratio of its own width and height on every side and then clamped to
    the image, so the returned region never leaves the source frame.

    EXIF orientation is not applied: the coordinates must refer to the pixels
    PIL.Image.open decodes, which is also the frame detectors report.

    Args:
        image_path: Path of the source image.
        box: [x1, y1, x2, y2] in source pixels.
        margin_ratio: Context margin added on each side, relative to the box
            width and height. Zero keeps the bare box.

    Returns:
        The cropped region converted to RGB, or None when the box is
        degenerate (zero area, non-finite values, or fully outside the image).

    Raises:
        ValueError: If box does not hold four numbers or margin_ratio is
            negative.
        OSError: If the file cannot be opened or decoded.
    """
    x1, y1, x2, y2 = _parse_box(box)
    if margin_ratio < 0:
        raise ValueError(f"margin_ratio must be non-negative, got {margin_ratio!r}")
    if not all(math.isfinite(value) for value in (x1, y1, x2, y2)):
        return None

    pad_x = abs(x2 - x1) * margin_ratio
    pad_y = abs(y2 - y1) * margin_ratio

    with Image.open(image_path) as image:
        image.load()
        width, height = image.size
        left = _clamp_int(min(x1, x2) - pad_x, 0, width)
        top = _clamp_int(min(y1, y2) - pad_y, 0, height)
        right = _clamp_int(max(x1, x2) + pad_x, 0, width)
        bottom = _clamp_int(max(y1, y2) + pad_y, 0, height)
        if right - left < 1 or bottom - top < 1:
            return None
        crop = image.crop((left, top, right, bottom))

    if crop.mode != "RGB":
        crop = crop.convert("RGB")
    return crop


def image_to_png_bytes(image: Image.Image, *, optimize: bool = False) -> bytes:
    """Encode a Pillow image as PNG bytes.

    Args:
        image: Any image; modes that PNG cannot store are converted to RGB
            first.
        optimize: Passed to PIL save; costs CPU time but yields smaller
            payloads.

    Returns:
        The complete PNG file as bytes, ready to hand to a chat adapter.
    """
    if image.mode not in _PNG_MODES:
        image = image.convert("RGB")
    buffer = BytesIO()
    image.save(buffer, format="PNG", optimize=optimize)
    return buffer.getvalue()


def downsample_waveform(
    samples: np.ndarray,
    num_points: int = 1024,
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce samples to a per-column min/max envelope for plotting.

    A waveform preview only needs one vertical extent per screen column, so the
    signal is split into num_points even buckets and each bucket is collapsed
    to its minimum and maximum. This keeps transients visible while bounding
    the drawing work.

    Args:
        samples: Audio samples, mono or multi-channel, integer or float.
        num_points: Upper bound on the number of envelope points.

    Returns:
        (lows, highs) float32 arrays of identical length, at most num_points
        long. Empty input yields two empty arrays.

    Raises:
        ValueError: If num_points is smaller than one or the input is not
            numeric audio data.
    """
    if num_points < 1:
        raise ValueError(f"num_points must be positive, got {num_points!r}")
    mono = _to_mono(samples)
    total = mono.size
    if total == 0:
        return np.empty(0, dtype=np.float32), np.empty(0, dtype=np.float32)
    if total <= num_points:
        return mono.copy(), mono.copy()

    starts = np.linspace(0, total, num_points + 1).astype(np.intp)[:-1]
    lows = np.minimum.reduceat(mono, starts)
    highs = np.maximum.reduceat(mono, starts)
    return lows.astype(np.float32, copy=False), highs.astype(np.float32, copy=False)


def spectrogram_image(
    samples: np.ndarray,
    sample_rate: int,
    *,
    width: int = 900,
    height: int = 420,
    title: str | None = None,
    fmin_hz: float = 60.0,
    fmax_hz: float | None = None,
    dynamic_range_db: float = 80.0,
    analysis_rate_hz: int = 16_000,
    nfft: int = 1024,
    colormap: str = "inferno",
) -> Image.Image:
    """Render a log-frequency spectrogram of a recording.

    The signal is block-averaged down to analysis_rate_hz, framed with a Hann
    window and transformed with numpy.fft.rfft. Power is turned into dB,
    interpolated onto a logarithmic frequency axis and mapped through a
    perceptual colour ramp; time and frequency ticks are drawn around the plot.

    Work stays bounded for long recordings: the frame hop grows so that at most
    _MAX_FRAMES frames are transformed, and the frequency grid is computed at
    plot resolution instead of covering every FFT bin.

    Levels are measured in dBFS against a full-scale sine, so silence renders
    black and a quiet but usable clip keeps its contrast.

    Args:
        samples: Audio samples, mono or multi-channel, integer or float.
        sample_rate: Sample rate of samples in hertz.
        width: Plot width in pixels, excluding the axes margins.
        height: Plot height in pixels, excluding the axes margins.
        title: Optional caption drawn above the plot.
        fmin_hz: Lowest frequency on the logarithmic axis, in hertz.
        fmax_hz: Highest frequency on the axis; defaults to the smaller of
            analysis_rate_hz / 2 and 16 kHz, and is clamped to Nyquist.
        dynamic_range_db: dB range spread over the full colour ramp.
        analysis_rate_hz: Rate the signal is decimated to before the STFT.
        nfft: STFT window length, which also sets the frequency resolution.
        colormap: One of inferno, viridis or gray.

    Returns:
        An RGB image sized width/height plus the axes margins.

    Raises:
        ValueError: If the samples, the rate, the axis bounds or any of the
            layout arguments are unusable.
    """
    rate = int(sample_rate)
    if rate <= 0:
        raise ValueError(f"sample_rate must be positive, got {sample_rate!r}")
    if width < 64 or height < 64:
        raise ValueError("width and height must both be at least 64 pixels")
    if dynamic_range_db <= 0:
        raise ValueError(f"dynamic_range_db must be positive, got {dynamic_range_db!r}")
    nfft = int(nfft)
    if nfft < 32:
        raise ValueError(f"nfft must be at least 32, got {nfft!r}")
    lut = _colormap_lut(colormap)

    mono = _to_mono(samples)
    if mono.size == 0:
        raise ValueError("samples must contain at least one value")

    duration = mono.size / float(rate)
    signal, analysis_rate = _decimate_signal(mono, rate, analysis_rate_hz)

    nyquist = analysis_rate / 2.0
    fmax = nyquist if fmax_hz is None else min(float(fmax_hz), nyquist)
    fmax = min(fmax, _DEFAULT_FMAX_HZ)
    fmin = float(fmin_hz)
    if not 0.0 < fmin < fmax:
        raise ValueError(f"expected 0 < fmin_hz < fmax_hz, got {fmin} and {fmax}")

    if signal.size < nfft:
        nfft = max(32, 1 << int(math.log2(max(1, signal.size))))
        signal = np.pad(signal, (0, max(0, nfft - signal.size)))

    power, freqs = _stft_power(signal, analysis_rate, nfft, max_frames=_MAX_FRAMES)
    # Reference full scale so that silence stays black: a full-scale sine puts
    # nfft / 4 into one bin, and the Hann window halves the amplitude again.
    decibel = 10.0 * np.log10(power + 1e-12) - 20.0 * math.log10(nfft / 4.0)
    peak_db = max(float(decibel.max()), _MIN_REFERENCE_DBFS)
    floor_db = peak_db - float(dynamic_range_db)

    grid = _frequency_grid(decibel, freqs, fmin, fmax, height)
    grid = _resample_columns(grid, width)
    scaled = np.clip((grid - floor_db) / float(dynamic_range_db), 0.0, 1.0)
    indices = (scaled * 255.0 + 0.5).astype(np.uint8)
    plot = Image.fromarray(lut[indices])

    return _compose_frame(
        plot,
        duration=duration,
        fmin_hz=fmin,
        fmax_hz=fmax,
        title=title,
    )


def _parse_box(box: Sequence[float]) -> tuple[float, float, float, float]:
    """Convert a bounding box into four floats.

    Args:
        box: [x1, y1, x2, y2] as numbers.

    Returns:
        The four coordinates as floats.

    Raises:
        ValueError: If box does not hold exactly four numeric values.
    """
    try:
        values = [float(value) for value in box]
    except (TypeError, ValueError) as exc:
        raise ValueError(f"box must contain four numbers, got {box!r}") from exc
    if len(values) != 4:
        raise ValueError(f"box must contain four numbers, got {len(values)}")
    return values[0], values[1], values[2], values[3]


def _clamp_int(value: float, low: int, high: int) -> int:
    """Round a value to the nearest integer and clamp it to a range.

    Args:
        value: Coordinate to round.
        low: Lower bound, inclusive.
        high: Upper bound, inclusive.

    Returns:
        The rounded and clamped coordinate.
    """
    return int(min(max(math.floor(value + 0.5), low), high))


def _to_mono(samples: np.ndarray) -> np.ndarray:
    """Convert arbitrary sample data into a contiguous 1-D float32 signal.

    Args:
        samples: Array-like audio data in any numeric dtype. Integer samples
            are scaled to [-1.0, 1.0]; multi-channel data is averaged to mono
            and non-finite values become zero.

    Returns:
        A contiguous float32 array.

    Raises:
        ValueError: If the data is not numeric audio or has more than two
            dimensions.
    """
    data = np.asarray(samples)
    if data.dtype.kind not in "iuf":
        raise ValueError(f"unsupported sample dtype {data.dtype}")
    if data.ndim == 0:
        data = data.reshape(1)
    elif data.ndim == 2:
        # Treat the shorter axis as channels; both (channels, samples) and
        # (samples, channels) layouts occur in the wild.
        axis = 0 if data.shape[0] < data.shape[1] else 1
        data = data.mean(axis=axis)
    elif data.ndim > 2:
        raise ValueError(f"expected 1-D or 2-D samples, got {data.ndim}-D")

    if data.dtype.kind in "iu":
        info = np.iinfo(data.dtype)
        data = data.astype(np.float32) / float(max(abs(info.min), info.max))
    else:
        data = data.astype(np.float32, copy=False)
    data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
    return np.ascontiguousarray(data, dtype=np.float32)


def _decimate_signal(
    signal: np.ndarray,
    rate: int,
    target_rate: int,
) -> tuple[np.ndarray, float]:
    """Block-average a mono signal down to a target rate.

    Args:
        signal: Mono float32 samples.
        rate: Sample rate of signal in hertz.
        target_rate: Desired rate after decimation, in hertz.

    Returns:
        (decimated, rate) where rate is the effective rate actually reached, as
        a float. Signals already at or below target_rate pass through
        untouched.
    """
    if target_rate <= 0:
        return signal, float(rate)
    factor = max(1, int(round(rate / float(target_rate))))
    if factor == 1:
        return signal, float(rate)
    usable = (signal.size // factor) * factor
    if usable == 0:
        return signal, float(rate)
    blocks = signal[:usable].reshape(-1, factor)
    # Averaging already acts as a crude anti-alias filter, which is enough for
    # a display-only spectrogram and keeps the cost linear.
    return blocks.mean(axis=1, dtype=np.float32), rate / factor


@functools.lru_cache(maxsize=8)
def _hann(nfft: int) -> np.ndarray:
    """Return a cached float32 Hann window of nfft samples.

    Args:
        nfft: Window length.

    Returns:
        The window coefficients.
    """
    return np.hanning(nfft).astype(np.float32)


def _stft_power(
    signal: np.ndarray,
    rate: float,
    nfft: int,
    max_frames: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the short-time power spectrum of a mono signal.

    Args:
        signal: Mono samples, at least nfft long.
        rate: Sample rate of signal in hertz.
        nfft: Window length.
        max_frames: Upper bound on the number of frames; the hop grows to
            respect it.

    Returns:
        (power, freqs) with power shaped (frames, nfft // 2 + 1) and freqs the
        matching bin centres in hertz.
    """
    hop = max(1, nfft // 4)
    span = signal.size - nfft
    if span > 0 and span // hop + 1 > max_frames:
        hop = max(hop, int(math.ceil(span / float(max(1, max_frames - 1)))))
    frames = np.lib.stride_tricks.sliding_window_view(signal, nfft)[::hop]
    spectrum = np.fft.rfft(frames * _hann(nfft), axis=1)
    power = (np.abs(spectrum) ** 2).astype(np.float32, copy=False)
    return power, np.fft.rfftfreq(nfft, d=1.0 / rate)


def _frequency_grid(
    decibel: np.ndarray,
    freqs: np.ndarray,
    fmin_hz: float,
    fmax_hz: float,
    height: int,
) -> np.ndarray:
    """Interpolate dB values onto a log-spaced vertical axis.

    Args:
        decibel: dB values shaped (frames, bins).
        freqs: Bin centres in hertz, ascending.
        fmin_hz: Frequency of the bottom row.
        fmax_hz: Frequency of the top row.
        height: Number of rows.

    Returns:
        Array shaped (height, frames) with the highest frequency on row 0.
    """
    targets = np.geomspace(fmax_hz, fmin_hz, height)
    upper = np.clip(np.searchsorted(freqs, targets, side="right"), 1, freqs.size - 1)
    lower = upper - 1
    span = freqs[upper] - freqs[lower]
    weight = np.divide(
        targets - freqs[lower],
        span,
        out=np.zeros(span.shape, dtype=np.float64),
        where=span > 0,
    ).astype(np.float32)
    grid = decibel[:, lower] * (1.0 - weight) + decibel[:, upper] * weight
    return grid.T


def _resample_columns(grid: np.ndarray, width: int) -> np.ndarray:
    """Resample the time axis of a spectrogram grid to a target width.

    Shrinking pools an equal-width window per column with a maximum so short
    transients survive; stretching interpolates linearly to avoid empty
    columns.

    Args:
        grid: Array shaped (rows, frames).
        width: Target number of columns.

    Returns:
        Array shaped (rows, width).
    """
    frames = grid.shape[1]
    if frames == width:
        return grid
    if frames > width:
        # Every column must pool the same number of frames. Buckets of unequal
        # size modulate the noise floor and paint a comb of vertical stripes
        # over the plot, so the window is fixed and only its offset varies.
        window = int(math.ceil(frames / float(width)))
        centres = np.linspace(0, frames - 1, width).astype(np.intp)
        starts = np.clip(centres - window // 2, 0, frames - window)
        windows = np.lib.stride_tricks.sliding_window_view(grid, window, axis=1)
        return windows[:, starts, :].max(axis=2)
    source = np.linspace(0.0, 1.0, frames)
    target = np.linspace(0.0, 1.0, width)
    return np.stack([np.interp(target, source, row) for row in grid])


def _compose_frame(
    plot: Image.Image,
    *,
    duration: float,
    fmin_hz: float,
    fmax_hz: float,
    title: str | None,
) -> Image.Image:
    """Place the spectrogram on a canvas and draw axes, ticks and labels.

    Args:
        plot: Spectrogram pixels.
        duration: Clip length in seconds, used for the time axis.
        fmin_hz: Lowest plotted frequency.
        fmax_hz: Highest plotted frequency.
        title: Optional caption.

    Returns:
        The annotated RGB frame.
    """
    width, height = plot.size
    label_font = _load_font(11)
    title_font = _load_font(13)
    top = _MARGIN_TOP_TITLE if title else _MARGIN_TOP_PLAIN
    canvas = Image.new(
        "RGB",
        (width + _MARGIN_LEFT + _MARGIN_RIGHT, height + top + _MARGIN_BOTTOM),
        _BACKGROUND,
    )
    canvas.paste(plot, (_MARGIN_LEFT, top))
    draw = ImageDraw.Draw(canvas)

    plot_right = _MARGIN_LEFT + width - 1
    plot_bottom = top + height - 1
    log_min = math.log(fmin_hz)
    log_span = math.log(fmax_hz) - log_min

    for frequency in _frequency_ticks(fmin_hz, fmax_hz):
        offset = (math.log(frequency) - log_min) / log_span
        y = int(round(top + (1.0 - offset) * (height - 1)))
        draw.line((_MARGIN_LEFT, y, plot_right, y), fill=_GRID_COLOR)
        draw.line((_MARGIN_LEFT - 5, y, _MARGIN_LEFT - 1, y), fill=_AXIS_COLOR)
        label = _format_frequency(frequency)
        text_width, text_height = _text_size(draw, label, label_font)
        draw.text(
            (
                int(round(_MARGIN_LEFT - 9 - text_width)),
                int(round(y - text_height / 2.0)),
            ),
            label,
            fill=_TEXT_COLOR,
            font=label_font,
        )

    for tick in _time_ticks(duration):
        x = int(round(_MARGIN_LEFT + (tick / duration) * (width - 1)))
        draw.line((x, top, x, plot_bottom), fill=_GRID_COLOR)
        draw.line((x, plot_bottom + 1, x, plot_bottom + 5), fill=_AXIS_COLOR)
        label = f"{tick:g}s"
        text_width, _ = _text_size(draw, label, label_font)
        draw.text(
            (int(round(x - text_width / 2.0)), plot_bottom + 8),
            label,
            fill=_TEXT_COLOR,
            font=label_font,
        )

    draw.rectangle(
        (_MARGIN_LEFT - 1, top - 1, plot_right + 1, plot_bottom + 1),
        outline=_AXIS_COLOR,
    )
    if title:
        draw.text((_MARGIN_LEFT, 5), title, fill=_TEXT_COLOR, font=title_font)
    return canvas


def _text_size(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont | ImageFont.FreeTypeFont,
) -> tuple[float, float]:
    """Measure a single line of text.

    Args:
        draw: Draw context used for measuring.
        text: Text to measure.
        font: Font the text will be drawn with.

    Returns:
        (width, height) in pixels.
    """
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top


def _frequency_ticks(fmin_hz: float, fmax_hz: float) -> list[float]:
    """Select log-axis tick frequencies inside a range.

    Args:
        fmin_hz: Lowest plotted frequency.
        fmax_hz: Highest plotted frequency.

    Returns:
        Ascending tick frequencies, thinned out to keep the axis readable.
    """
    ticks = [value for value in _FREQUENCY_TICKS_HZ if fmin_hz <= value <= fmax_hz]
    if not ticks:
        return [fmin_hz, fmax_hz]
    if len(ticks) > _MAX_TICKS_PER_AXIS:
        stride = int(math.ceil(len(ticks) / _MAX_TICKS_PER_AXIS))
        ticks = ticks[::stride]
    return ticks


def _time_ticks(duration: float) -> list[float]:
    """Select time-axis tick positions for a clip length.

    Args:
        duration: Clip length in seconds.

    Returns:
        Ascending tick positions in seconds, starting at zero.
    """
    target = duration / float(_MAX_TICKS_PER_AXIS)
    step = next(
        (value for value in _TIME_TICKS_S if value >= target),
        _TIME_TICKS_S[-1],
    )
    return [index * step for index in range(int(duration // step) + 1)]


def _format_frequency(frequency: float) -> str:
    """Format a tick frequency for the vertical axis.

    Args:
        frequency: Frequency in hertz.

    Returns:
        500 for sub-kilohertz values, 2k above.
    """
    if frequency >= 1000.0:
        return f"{frequency / 1000.0:g}k"
    return f"{frequency:g}"


@functools.lru_cache(maxsize=8)
def _colormap_lut(name: str) -> np.ndarray:
    """Build a 256-entry RGB lookup table for a named colour ramp.

    Args:
        name: Key of _COLOR_STOPS.

    Returns:
        Array shaped (256, 3) with uint8 values.

    Raises:
        ValueError: If the name is unknown.
    """
    stops = _COLOR_STOPS.get(name)
    if stops is None:
        raise ValueError(
            f"unknown colormap {name!r}; choose from {sorted(_COLOR_STOPS)}"
        )
    positions = np.array([stop[0] for stop in stops], dtype=np.float64)
    ramp = np.linspace(0.0, 1.0, 256)
    table = np.empty((256, 3), dtype=np.uint8)
    for channel in range(3):
        values = np.array([stop[1][channel] for stop in stops], dtype=np.float64)
        table[:, channel] = np.clip(np.interp(ramp, positions, values), 0, 255).astype(
            np.uint8
        )
    return table


@functools.lru_cache(maxsize=8)
def _load_font(size: int) -> ImageFont.ImageFont | ImageFont.FreeTypeFont:
    """Load a small label font, preferring a scalable default.

    Args:
        size: Requested font size in pixels.

    Returns:
        A usable font object; older Pillow builds, or builds without FreeType,
        fall back to the fixed-size bitmap font.
    """
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError):
        return ImageFont.load_default()
