"""Launcher for wavetable header generation."""

from __future__ import annotations

import re

import generateWT001
import generateWT002
import generateWT003
import generateWT004
import generateWT005
import generateWT006
import generateWT007
import generateWT008
import generateWT009
import generateWT010
import generateWT011


GENERATOR_MAP = {
	"WT001": generateWT001,
	"WT002": generateWT002,
	"WT003": generateWT003,
	"WT004": generateWT004,
	"WT005": generateWT005,
	"WT006": generateWT006,
	"WT007": generateWT007,
	"WT008": generateWT008,
	"WT009": generateWT009,
	"WT010": generateWT010,
	"WT011": generateWT011,
}


def main() -> None:
	"""Run all registered generators sequentially."""
	for generator_name in sorted(GENERATOR_MAP.keys()):
		generator_module = GENERATOR_MAP[generator_name]

		match = re.search(r"(\d+)$", generator_name)
		table_index = int(match.group(1)) if match else 1

		print(f"Running {generator_name} with table index {table_index}...")
		generated_files, headers_dir = generator_module.generate_headers(
			table_index=table_index
		)
		print(f"Generated {len(generated_files)} header files:")
		for header in generated_files:
			print(f" - {header}")

		print("Generating waveform HTML preview...")
		generator_module.run_html_generator(headers_dir)


if __name__ == "__main__":
	main()
