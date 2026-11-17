"""Tests for the rendering helpers in bird.render.

The module is exercised against the real Pillow and numpy stack, so the suite
is meant to run inside the AstrBot runtime container::

    docker exec -u root astrbot python -m pytest tests/test_render.py -q
"""

from __future__ import annotations

import io
import sys
import time
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from bird import render  # noqa: E402  (path bootstrap must run first)

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def _write_image(path: Path, size=(200, 100), mode="RGB") -> Path:
    """Create a two-tone test image whose halves are easy to identify.

    Args:
        path: Destination path.
        size: Image size as (width, height).
        mode: Pillow image mode.

    Returns:
        The path that was written.
    """
    width, height = size
    image = Image.new(mode, size, (255, 0, 0))
    for x in range(width // 2, width):
        for y in range(height):
            image.putpixel((x, y), (0, 0, 255))
    image.save(path)
    return path


def _plot_region(width: int, height: int, titled: bool) -> tuple[int, int]:
    """Return the top-left corner of the plot area inside a composed frame.

    Args:
        width: Plot width passed to spectrogram_image.
        height: Plot height passed to spectrogram_image.
        titled: Whether a title margin was reserved.

    Returns:
        (left, top) in pixels.
    """
    top = render._MARGIN_TOP_TITLE if titled else render._MARGIN_TOP_PLAIN
    return render._MARGIN_LEFT, top


def _peak_frequency(
    image: Image.Image,
    width: int,
    height: int,
    fmax_hz: float,
) -> float:
    """Locate the brightest row of a composed spectrogram and invert the axis.

    Args:
        image: Frame returned by spectrogram_image.
        width: Plot width that was requested.
        height: Plot height that was requested.
        fmax_hz: Highest frequency on the rendered log axis.

    Returns:
        The frequency in hertz that the brightest row maps to.
    """
    left, top = _plot_region(width, height, titled=False)
    pixels = np.asarray(image.convert("RGB"), dtype=np.float64)
    plot = pixels[top : top + height, left : left + width]
    row = int(np.argmax(plot.sum(axis=(1, 2))))
    fmin = 60.0
    return fmax_hz * (fmin / fmax_hz) ** (row / float(height - 1))


@pytest.fixture
def tone() -> tuple[np.ndarray, int]:
    """Build a 1-second 3 kHz sine tone.

    Returns:
        (samples, sample_rate) ready for spectrogram_image.
    """
    rate = 32_000
    times = np.arange(rate, dtype=np.float32) / rate
    return (0.6 * np.sin(2.0 * np.pi * 3000.0 * times)).astype(np.float32), rate


def test_crop_returns_bare_box_without_margin(tmp_path: Path) -> None:
    """A zero margin returns exactly the requested pixels."""
    source = _write_image(tmp_path / "source.png")
    crop = render.crop_image(source, [50, 20, 150, 80], margin_ratio=0.0)
    assert crop is not None
    assert crop.size == (100, 60)
    assert crop.mode == "RGB"
    # The box starts in the red half and crosses into the blue half.
    assert crop.getpixel((0, 0)) == (255, 0, 0)
    assert crop.getpixel((99, 0)) == (0, 0, 255)


def test_crop_expands_by_margin_ratio(tmp_path: Path) -> None:
    """The margin is proportional to the box width and height."""
    source = _write_image(tmp_path / "source.png", size=(400, 200))
    crop = render.crop_image(source, [100, 50, 200, 100], margin_ratio=0.25)
    assert crop is not None
    # 25 percent of a 100x50 box adds 25 px horizontally and 12.5 px vertically.
    assert crop.size == (150, 75)


def test_crop_clamps_to_image_bounds(tmp_path: Path) -> None:
    """Boxes that hang over the edge are clipped, not padded."""
    source = _write_image(tmp_path / "source.png")
    crop = render.crop_image(source, [-50.0, -50.0, 50.0, 50.0], margin_ratio=0.0)
    assert crop is not None
    assert crop.size == (50, 50)

    # A huge margin is clipped back to the full frame.
    crop = render.crop_image(source, [150.0, 80.0, 900.0, 900.0], margin_ratio=0.5)
    assert crop is not None
    assert crop.size == (200, 100)


def test_crop_accepts_reversed_corners(tmp_path: Path) -> None:
    """Corners may be given in any order."""
    source = _write_image(tmp_path / "source.png")
    crop = render.crop_image(source, [150, 80, 50, 20], margin_ratio=0.0)
    assert crop is not None
    assert crop.size == (100, 60)


@pytest.mark.parametrize(
    "box",
    [
        [50.0, 50.0, 50.0, 50.0],  # zero area
        [500.0, 500.0, 600.0, 600.0],  # entirely outside
        [float("nan"), 0.0, 10.0, 10.0],
        [float("inf"), 0.0, 10.0, 10.0],
    ],
)
def test_crop_degenerate_boxes_return_none(tmp_path: Path, box) -> None:
    """Unusable boxes report None instead of raising."""
    source = _write_image(tmp_path / "source.png")
    assert render.crop_image(source, box, margin_ratio=0.0) is None


@pytest.mark.parametrize(
    ("box", "margin"),
    [
        ([1.0, 2.0, 3.0], 0.0),
        ([1.0, 2.0, 3.0, 4.0, 5.0], 0.0),
        ([1.0, 2.0, 3.0, 4.0], -0.5),
    ],
)
def test_crop_rejects_invalid_arguments(tmp_path: Path, box, margin) -> None:
    """Bad geometry arguments raise ValueError."""
    source = _write_image(tmp_path / "source.png")
    with pytest.raises(ValueError):
        render.crop_image(source, box, margin_ratio=margin)


def test_crop_converts_non_rgb_sources(tmp_path: Path) -> None:
    """Palette and alpha images come back as RGB."""
    path = tmp_path / "rgba.png"
    Image.new("RGBA", (64, 64), (10, 20, 30, 128)).save(path)
    crop = render.crop_image(path, [0, 0, 32, 32], margin_ratio=0.0)
    assert crop is not None
    assert crop.mode == "RGB"
    assert crop.size == (32, 32)


def test_image_to_png_bytes_roundtrip(tmp_path: Path) -> None:
    """PNG bytes decode back to an identical image."""
    image = Image.new("RGB", (37, 19), (12, 34, 56))
    payload = render.image_to_png_bytes(image)
    assert payload.startswith(PNG_SIGNATURE)
    with Image.open(io.BytesIO(payload)) as decoded:
        assert decoded.size == (37, 19)


def test_image_to_png_bytes_converts_cmyk() -> None:
    """Modes PNG cannot store are converted instead of failing."""
    payload = render.image_to_png_bytes(Image.new("CMYK", (8, 8), (0, 0, 0, 0)))
    with Image.open(__import__("io").BytesIO(payload)) as decoded:
        assert decoded.mode == "RGB"


def test_downsample_caps_point_count() -> None:
    """The envelope never exceeds the requested number of points."""
    samples = np.sin(np.linspace(0.0, 40.0 * np.pi, 32_000)).astype(np.float32)
    lows, highs = render.downsample_waveform(samples, num_points=256)
    assert lows.shape == highs.shape == (256,)
    assert np.all(lows <= highs)


def test_downsample_keeps_peaks() -> None:
    """A single-sample spike survives the reduction."""
    samples = np.zeros(10_000, dtype=np.float32)
    samples[5000] = 0.9
    samples[7000] = -0.8
    _, highs = render.downsample_waveform(samples, num_points=200)
    lows, _ = render.downsample_waveform(samples, num_points=200)
    assert highs.max() == pytest.approx(0.9)
    assert lows.min() == pytest.approx(-0.8)


def test_downsample_scales_integer_samples() -> None:
    """Int16 input is normalised to the unit range."""
    samples = np.array([0, 32767, -32768, 16384], dtype=np.int16)
    lows, highs = render.downsample_waveform(samples, num_points=8)
    assert highs.max() <= 1.0
    assert lows.min() >= -1.0
    assert highs.max() == pytest.approx(1.0, abs=1e-3)


@pytest.mark.parametrize("shape", [(2, 4000), (4000, 2)])
def test_downsample_handles_stereo_layouts(shape: tuple[int, int]) -> None:
    """Both channel orders collapse to a single envelope."""
    samples = np.zeros(shape, dtype=np.float32)
    samples[0] = 0.5
    lows, highs = render.downsample_waveform(samples, num_points=64)
    assert lows.shape == (64,)
    assert highs.max() <= 0.5 + 1e-6


def test_downsample_short_and_empty_input() -> None:
    """Short and empty inputs are handled without padding tricks."""
    short = np.array([0.1, -0.2, 0.3], dtype=np.float32)
    lows, highs = render.downsample_waveform(short, num_points=16)
    assert lows.shape == (3,)
    assert np.allclose(lows, short)

    empty_lows, empty_highs = render.downsample_waveform(
        np.empty(0, dtype=np.float32), num_points=16
    )
    assert empty_lows.size == empty_highs.size == 0


def test_downsample_rejects_bad_point_count() -> None:
    """A non-positive point count is a programming error."""
    with pytest.raises(ValueError):
        render.downsample_waveform(np.zeros(10, dtype=np.float32), num_points=0)


def test_spectrogram_default_size(tone) -> None:
    """The composed frame is the plot plus the axes margins."""
    samples, rate = tone
    image = render.spectrogram_image(samples, rate, width=640, height=360)
    assert image.mode == "RGB"
    assert image.size == (
        640 + render._MARGIN_LEFT + render._MARGIN_RIGHT,
        360 + render._MARGIN_TOP_PLAIN + render._MARGIN_BOTTOM,
    )


def test_spectrogram_title_adds_height(tone) -> None:
    """A title reserves extra room above the plot."""
    samples, rate = tone
    plain = render.spectrogram_image(samples, rate, width=400, height=240)
    titled = render.spectrogram_image(
        samples, rate, width=400, height=240, title="Recording"
    )
    assert titled.width == plain.width
    assert titled.height - plain.height == (
        render._MARGIN_TOP_TITLE - render._MARGIN_TOP_PLAIN
    )


def test_spectrogram_peak_matches_tone(tone) -> None:
    """A 3 kHz tone lands on the expected row of the log axis."""
    samples, rate = tone
    height = 420
    image = render.spectrogram_image(samples, rate, width=300, height=height)
    # The signal is decimated to 16 kHz, so the axis tops out at 8 kHz.
    peak = _peak_frequency(image, 300, height, fmax_hz=8000.0)
    assert peak == pytest.approx(3000.0, rel=0.08)


def test_spectrogram_silence_is_dark() -> None:
    """Silence yields a uniformly dark plot instead of noise."""
    image = render.spectrogram_image(np.zeros(8000, dtype=np.float32), 16_000)
    left, top = _plot_region(900, 420, titled=False)
    plot = np.asarray(image.convert("RGB"))[top : top + 420, left : left + 900]
    assert float(plot.mean()) < 60.0


def test_spectrogram_handles_short_signal() -> None:
    """Signals shorter than one FFT window are zero padded, not rejected."""
    samples = np.zeros(700, dtype=np.float32)
    samples[:200] = 0.5
    image = render.spectrogram_image(samples, 16_000, width=200, height=160)
    assert image.size[0] > 200


def test_spectrogram_stereo_matches_mono(tone) -> None:
    """A duplicated channel renders the same peak as the mono signal."""
    samples, rate = tone
    stereo = np.stack([samples, samples])
    mono_peak = _peak_frequency(
        render.spectrogram_image(samples, rate, width=200, height=300),
        200,
        300,
        8000.0,
    )
    stereo_peak = _peak_frequency(
        render.spectrogram_image(stereo, rate, width=200, height=300),
        200,
        300,
        8000.0,
    )
    assert stereo_peak == pytest.approx(mono_peak, rel=0.03)


def test_spectrogram_colormaps_differ(tone) -> None:
    """The named ramps produce visibly different pixels."""
    samples, rate = tone
    inferno = np.asarray(render.spectrogram_image(samples, rate, width=120, height=120))
    gray = np.asarray(
        render.spectrogram_image(samples, rate, width=120, height=120, colormap="gray")
    )
    assert inferno.shape == gray.shape
    assert not np.array_equal(inferno, gray)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"sample_rate": 0}, "sample_rate"),
        ({"colormap": "jet"}, "colormap"),
        ({"width": 32}, "at least 64"),
        ({"dynamic_range_db": 0.0}, "dynamic_range_db"),
        ({"nfft": 16}, "nfft"),
        ({"fmin_hz": 5000.0, "fmax_hz": 1000.0}, "fmin_hz"),
    ],
)
def test_spectrogram_rejects_invalid_arguments(kwargs, match) -> None:
    """Unusable arguments fail loudly with a helpful message."""
    rate = kwargs.pop("sample_rate", 16_000)
    with pytest.raises(ValueError, match=match):
        render.spectrogram_image(np.zeros(4000, dtype=np.float32), rate, **kwargs)


def test_spectrogram_rejects_empty_samples() -> None:
    """An empty recording is an error, not a blank picture."""
    with pytest.raises(ValueError, match="at least one value"):
        render.spectrogram_image(np.empty(0, dtype=np.float32), 16_000)


def test_spectrogram_long_recording_is_fast() -> None:
    """A 30 s 32 kHz int16 clip stays well inside the interactive budget."""
    rate = 32_000
    rng = np.random.default_rng(20240517)
    times = np.arange(rate * 30, dtype=np.float32) / rate
    carrier = np.sin(2.0 * np.pi * 2200.0 * times)
    samples = (carrier * 0.3 + rng.normal(0.0, 0.01, times.size)) * 32767.0
    samples = samples.astype(np.int16)

    start = time.perf_counter()
    image = render.spectrogram_image(samples, rate)
    elapsed = time.perf_counter() - start

    assert image.size == (
        900 + render._MARGIN_LEFT + render._MARGIN_RIGHT,
        420 + render._MARGIN_TOP_PLAIN + render._MARGIN_BOTTOM,
    )
    assert elapsed < 3.0, f"rendering took {elapsed:.2f}s"
