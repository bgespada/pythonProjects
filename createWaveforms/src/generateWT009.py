from __future__ import annotations

import math
import re
from pathlib import Path

from config import load_default_config
import generateHtml


INITIAL_WAVEFORM_PATH = Path(r"C:\DaisyProjects\lib\samples\waveforms\wf019.h")
FINAL_WAVEFORM_PATH = Path(r"C:\DaisyProjects\lib\samples\waveforms\wf023.h")


def _sanitize_identifier(name: str) -> str:
	"""Convert an arbitrary name into a valid C/C++ identifier."""
	sanitized = re.sub(r"[^0-9A-Za-z_]", "_", name)
	if sanitized and sanitized[0].isdigit():
		sanitized = f"_{sanitized}"
	return sanitized


def _render_waveform_name(pattern_name: str, table_index: int, waveform_index: int) -> str:
	"""Render a waveform name from pattern placeholders.

	Supported placeholders:
	- xxx: table index (3 digits)
	- yy: waveform index (2 digits)
	"""
	rendered = pattern_name

	if "xxx" in rendered:
		rendered = rendered.replace("xxx", f"{table_index:03d}")

	if "yy" in rendered:
		rendered = rendered.replace("yy", f"{waveform_index:02d}")

	if rendered == pattern_name:
		rendered = f"{pattern_name}_{waveform_index:02d}"

	return rendered


def _render_table_folder_name(pattern_name: str, table_index: int) -> str:
	"""Render folder name for a wavetable, e.g. WT001 from WTxxx_yy."""
	folder = pattern_name

	if "xxx" in folder:
		folder = folder.replace("xxx", f"{table_index:03d}")

	# Remove per-waveform token and collapse separators.
	folder = folder.replace("yy", "")
	folder = re.sub(r"[_\-]+$", "", folder)
	if not folder:
		folder = f"WT{table_index:03d}"

	return folder


def _parse_waveform_header_values(header_path: Path) -> list[float]:
	"""Extract float array values from a C++ waveform header file."""
	if not header_path.exists():
		raise FileNotFoundError(f"Waveform header not found: {header_path}")

	content = header_path.read_text(encoding="utf-8")
	match = re.search(
		r"const\s+float\s+\w+\s*\[.*?\]\s*=\s*\{([^}]+)\}",
		content,
		re.DOTALL,
	)
	if not match:
		raise ValueError(f"Could not parse waveform data from: {header_path}")

	raw_values = match.group(1)
	values = [float(v.strip()) for v in raw_values.split(",") if v.strip()]
	if not values:
		raise ValueError(f"No waveform values found in: {header_path}")

	return values


def _resample_waveform(values: list[float], target_size: int) -> list[float]:
	"""Resample waveform to target size with wrap-around linear interpolation."""
	if target_size <= 0:
		raise ValueError(f"target_size must be > 0, got {target_size}")
	if len(values) == target_size:
		return list(values)

	src_size = len(values)
	resampled: list[float] = []

	for sample_index in range(target_size):
		src_pos = (sample_index * src_size) / target_size
		idx0 = int(math.floor(src_pos)) % src_size
		idx1 = (idx0 + 1) % src_size
		frac = src_pos - math.floor(src_pos)
		value = ((1.0 - frac) * values[idx0]) + (frac * values[idx1])
		resampled.append(value)

	return resampled


def _load_morph_endpoints(size: int) -> tuple[list[float], list[float]]:
	"""Load and size-match initial/final waveforms used for morphing."""
	initial = _resample_waveform(_parse_waveform_header_values(INITIAL_WAVEFORM_PATH), size)
	final = _resample_waveform(_parse_waveform_header_values(FINAL_WAVEFORM_PATH), size)
	return initial, final


def _build_morphed_waveform(initial_wave: list[float], final_wave: list[float], morph: float) -> list[float]:
	"""Create one waveform by morphing from initial header waveform to final header waveform."""
	if len(initial_wave) != len(final_wave):
		raise ValueError("Initial and final waveforms must have the same size")

	size = len(initial_wave)
	values: list[float] = []

	for sample_index in range(size):
		value = ((1.0 - morph) * initial_wave[sample_index]) + (
			morph * final_wave[sample_index]
		)
		value = max(-1.0, min(1.0, value))
		values.append(value)

	return values


def _format_header_content(symbol_name: str, samples: list[float], size: int) -> str:
	"""Create C++ header text for one waveform array."""
	macro_name = f"{symbol_name.upper()}_SIZE"

	lines = [
		"#pragma once",
		f"#define {macro_name} {size}",
		f"const float {symbol_name}[{macro_name}] = {{",
	]

	row_width = 8
	for idx in range(0, len(samples), row_width):
		row = samples[idx : idx + row_width]
		numbers = ", ".join(f"{number:.6f}" for number in row)
		if idx + row_width < len(samples):
			lines.append(f"    {numbers},")
		else:
			lines.append(f"    {numbers}")

	lines.append("};")
	lines.append("")

	return "\n".join(lines)


def generate_headers(table_index: int = 1) -> tuple[list[Path], Path]:
	"""Generate all waveform headers using constants.cfg."""
	cfg = load_default_config()

	output_folder = cfg.get_output_new_waveforms_folder()
	pattern_header_path = cfg.get_pattern_header()
	pattern_name = cfg.get_pattern_name("WTxxx_yy")
	max_size = cfg.get_max_size()
	num_waveforms = cfg.get_num_of_waveforms()

	if not output_folder:
		raise ValueError("Missing OUTPUT_NEW_WAVEFORMS_FOLDER in constants.cfg")
	if not pattern_header_path:
		raise ValueError("Missing PATTERN_HEADER in constants.cfg")
	if max_size <= 0:
		raise ValueError(f"MAX_SIZE must be > 0, got {max_size}")
	if num_waveforms <= 0:
		raise ValueError(f"NUM_OF_WAVEFORMS must be > 0, got {num_waveforms}")

	initial_wave, final_wave = _load_morph_endpoints(max_size)

	template_header = Path(pattern_header_path)
	if not template_header.exists():
		raise FileNotFoundError(f"PATTERN_HEADER not found: {template_header}")

	table_folder_name = _render_table_folder_name(pattern_name, table_index)
	out_dir = Path(output_folder) / table_folder_name
	out_dir.mkdir(parents=True, exist_ok=True)

	created_files: list[Path] = []

	for waveform_index in range(num_waveforms):
		morph = 0.0 if num_waveforms == 1 else waveform_index / (num_waveforms - 1)
		wave_name = _render_waveform_name(pattern_name, table_index, waveform_index)
		symbol_name = _sanitize_identifier(wave_name)

		samples = _build_morphed_waveform(initial_wave, final_wave, morph)
		content = _format_header_content(symbol_name, samples, max_size)

		output_file = out_dir / f"{wave_name}.h"
		output_file.write_text(content, encoding="utf-8")
		created_files.append(output_file)

	return created_files, out_dir


def run_html_generator(headers_dir: Path) -> None:
	"""Run generateHtml.py to refresh waveform preview HTML after headers are generated."""
	output_html = headers_dir / "waveforms.html"
	generateHtml.main([
		"--headers-dir",
		str(headers_dir),
		"--output",
		str(output_html),
	])

