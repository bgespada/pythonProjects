from __future__ import annotations

import math
import re
from pathlib import Path

from config import load_default_config
import generateHtml


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


def _build_morphed_waveform(size: int, morph: float) -> list[float]:
	"""Create one waveform by morphing from sine (0.0) to saw (1.0)."""
	values: list[float] = []

	for sample_index in range(size):
		phase = sample_index / size

		sine_value = math.sin(2.0 * math.pi * phase)
		saw_value = (2.0 * phase) - 1.0

		value = ((1.0 - morph) * sine_value) + (morph * saw_value)
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

		samples = _build_morphed_waveform(max_size, morph)
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

