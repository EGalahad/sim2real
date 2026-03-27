from __future__ import annotations

from contextlib import contextmanager
import os
import tempfile
from pathlib import Path


VIEWER_VISUAL_XML = """\
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.1 0.1 0.1" specular="0.9 0.9 0.9"/>
    <rgba haze="0.15 0.25 0.35 1"/>
    <global azimuth="-140" elevation="-20"/>
  </visual>
"""

VIEWER_ASSET_XML_TEMPLATE = """\
    <texture type="skybox" builtin="flat" rgb1="0 0 0" rgb2="0 0 0" width="512" height="3072"/>
    <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="{rgb1}" rgb2="{rgb2}" markrgb="0.8 0.8 0.8" width="300" height="300"/>
    <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
"""

VIEWER_WORLDBODY_XML = """\
    <light pos="1 0 3.5" dir="0 0 -1" directional="true"/>
    <geom name="floor" size="0 0 0.05" type="plane" material="groundplane"/>
"""


def _format_rgb(rgb: tuple[float, float, float]) -> str:
    return " ".join(f"{channel:.3f}" for channel in rgb)


def _scale_rgb(rgb: tuple[float, float, float], scale: float) -> tuple[float, float, float]:
    return tuple(min(max(channel * scale, 0.0), 1.0) for channel in rgb)


def inject_viewer_xml(
    xml_text: str,
    *,
    ground_rgb: tuple[float, float, float] = (0.2, 0.3, 0.4),
) -> str:
    ground_rgb_dark = _scale_rgb(ground_rgb, 0.75)
    viewer_asset_xml = VIEWER_ASSET_XML_TEMPLATE.format(
        rgb1=_format_rgb(ground_rgb),
        rgb2=_format_rgb(ground_rgb_dark),
    )

    if "<visual>" not in xml_text:
        insertion_point = xml_text.find("<asset>")
        if insertion_point < 0:
            raise ValueError("Expected <asset> block in MJCF")
        xml_text = xml_text[:insertion_point] + VIEWER_VISUAL_XML + xml_text[insertion_point:]

    asset_close = xml_text.find("</asset>")
    if asset_close < 0:
        raise ValueError("Expected </asset> block in MJCF")
    xml_text = xml_text[:asset_close] + viewer_asset_xml + xml_text[asset_close:]

    worldbody_close = xml_text.find("</worldbody>")
    if worldbody_close < 0:
        raise ValueError("Expected </worldbody> block in MJCF")
    xml_text = xml_text[:worldbody_close] + VIEWER_WORLDBODY_XML + xml_text[worldbody_close:]
    return xml_text


@contextmanager
def temp_mjcf_with_floor(
    mjcf_path: Path,
    *,
    ground_rgb: tuple[float, float, float] = (0.2, 0.3, 0.4),
) -> Path:
    xml_text = mjcf_path.read_text(encoding="utf-8")
    viewer_xml = inject_viewer_xml(xml_text, ground_rgb=ground_rgb)

    path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            suffix=".xml",
            prefix=".sim2real_viewer_",
            dir=mjcf_path.parent,
            delete=False,
            encoding="utf-8",
        ) as tmp:
            tmp.write(viewer_xml)
            path = Path(tmp.name)
        yield path
    finally:
        if path is None:
            return
        try:
            os.unlink(path)
        except FileNotFoundError:
            pass
