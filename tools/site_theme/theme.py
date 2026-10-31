"""Resolve palette swatches and semantic roles without plotting dependencies."""
from __future__ import annotations

import copy
import hashlib
import json
import math
import os
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_THEME = ROOT / "theme" / "theme.json"
MODES = ("normal", "accessible")


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        if key not in result:
            raise ValueError(f"Unknown override: {key}")
        if isinstance(value, dict) and isinstance(result[key], dict) and not any(
            operation in value or operation in result[key] for operation in ("mix", "alpha")
        ):
            result[key] = merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def rgb(color):
    if not isinstance(color, str) or not re.fullmatch(r"#[0-9a-fA-F]{6}", color):
        raise ValueError(f"Expected a six-digit hex color, got {color!r}")
    return tuple(int(color[i:i + 2], 16) / 255 for i in (1, 3, 5))


def contrast(a, b):
    def luminance(color):
        channels = [v / 12.92 if v <= 0.04045 else ((v + 0.055) / 1.055) ** 2.4 for v in rgb(color)]
        return sum(v * weight for v, weight in zip(channels, (0.2126, 0.7152, 0.0722)))
    lo, hi = sorted((luminance(a), luminance(b)))
    return (hi + 0.05) / (lo + 0.05)


class Theme:
    def __init__(self, path=None):
        self.path = Path(path or os.environ.get("WEBSITE_THEME", DEFAULT_THEME)).resolve()
        self.data = json.loads(self.path.read_text())
        if set(self.data) != {"schema_version", "name", "palettes", "roles", "assets", "overrides"}:
            raise ValueError("Theme needs schema_version, name, palettes, roles, assets, and overrides")
        if self.data["schema_version"] != 1:
            raise ValueError("Unsupported theme schema version")
        if set(self.data["palettes"]) != set(MODES):
            raise ValueError("Provide normal and accessible palettes")
        if set(self.data["overrides"]) - {"accessible"}:
            raise ValueError("Only accessible overrides are supported")
        keys = set(self.data["palettes"]["normal"])
        for mode in MODES:
            if set(self.data["palettes"][mode]) != keys:
                raise ValueError("Both palettes must have the same swatch names")
            for color in self.data["palettes"][mode].values():
                rgb(color)
        self.resolved = {}
        for mode in MODES:
            tree = merge({"roles": self.data["roles"], "assets": self.data["assets"]},
                         self.data["overrides"].get(mode, {}))
            tree["palette"] = self.data["palettes"][mode]

            def resolve(value, trail=()):
                if isinstance(value, str) and value.startswith("@"):
                    key = value[1:]
                    if key in trail:
                        raise ValueError("Circular color reference: " + " -> ".join((*trail, key)))
                    target = tree
                    try:
                        for part in key.split("."):
                            target = target[part]
                    except (KeyError, TypeError) as error:
                        raise ValueError(f"Unknown color reference: {value}") from error
                    return resolve(target, (*trail, key))
                if isinstance(value, dict):
                    if "mix" in value:
                        if set(value) != {"mix", "weight"} or len(value["mix"]) != 2:
                            raise ValueError("A mix requires two colors and a weight")
                        weight = value["weight"]
                        if not isinstance(weight, (int, float)) or not 0 <= weight <= 1:
                            raise ValueError("Mix weight must be between zero and one")
                        foreground, background = [rgb(resolve(v, trail)) for v in value["mix"]]
                        channels = [round(255 * (weight * a + (1 - weight) * b))
                                    for a, b in zip(foreground, background)]
                        return "#" + "".join(f"{v:02x}" for v in channels)
                    if "alpha" in value:
                        if set(value) != {"alpha", "opacity"} or not 0 <= value["opacity"] <= 1:
                            raise ValueError("An alpha color requires a color and opacity between zero and one")
                        channels = [round(255 * v) for v in rgb(resolve(value["alpha"], trail))]
                        return f"rgba({channels[0]}, {channels[1]}, {channels[2]}, {value['opacity']:g})"
                    return {k: resolve(v, trail) for k, v in value.items()}
                if isinstance(value, list):
                    return [resolve(v, trail) for v in value]
                if isinstance(value, float) and not math.isfinite(value):
                    raise ValueError("Theme numbers must be finite")
                return value

            self.resolved[mode] = resolve(tree)
        self.validate()

    @property
    def digest(self):
        return hashlib.sha256(json.dumps(self.data, sort_keys=True).encode()).hexdigest()

    def validate(self):
        for mode in MODES:
            def check(value, path):
                if isinstance(value, dict):
                    for key, child in value.items():
                        check(child, f"{path}.{key}")
                elif isinstance(value, list):
                    if path.endswith(("ramp", "seq", "fills")) and len(value) < 2:
                        raise ValueError(f"{path} needs at least two colors")
                    for child in value:
                        check(child, path)
                elif isinstance(value, str) and value.startswith("#"):
                    rgb(value)
            check(self.resolved[mode], mode)
            for name, value in self.resolved[mode]["roles"]["web"].items():
                if not re.fullmatch(r"[a-z][a-z0-9-]*", name):
                    raise ValueError(f"Invalid CSS role name: {name}")
                if not isinstance(value, str) or not (re.fullmatch(r"#[0-9a-fA-F]{6}", value)
                        or re.fullmatch(r"rgba\([0-9., ]+\)", value)):
                    raise ValueError(f"Invalid CSS color for {name}: {value!r}")

    def style(self, asset, mode="normal"):
        return copy.deepcopy(self.resolved[mode]["assets"][asset])

    def css(self):
        lines = ["/* Generated from theme/theme.json by tools/theme.py. Do not edit. */"]
        for mode, selector in (("normal", ":root"), ("accessible", "body.accessible")):
            lines.append(f"{selector} {{")
            for name, value in self.resolved[mode]["roles"]["web"].items():
                lines.append(f"    --{name}: {value};")
            lines.append("}")
        return "\n".join(lines) + "\n"


def asset_style(asset, mode="normal"):
    return Theme().style(asset, mode)


def data_path(name):
    return ROOT / "tools" / "data" / f"{name}.npz"


def font_files():
    directory = Path(os.environ.get("WEBSITE_FONT_DIR", "~/.local/share/fonts/Poppins")).expanduser()
    files = sorted(directory.glob("*.ttf"))
    if not files:
        raise ValueError(f"Poppins fonts are missing from {directory}")
    return files


def output_dir():
    return Path(os.environ.get("WEBSITE_OUTPUT", ROOT / ".theme-build" / "renders"))
