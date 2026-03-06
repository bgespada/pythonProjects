from __future__ import annotations

from pathlib import Path
import re
from typing import Any


class ConfigManager:
	"""Read and write settings from constants.cfg.

	Supported line format:
	[KEY] "value"
	"""

	_LINE_PATTERN = re.compile(r'^\s*\[(?P<key>[^\]]+)\]\s+"(?P<value>.*)"\s*$')

	def __init__(self, config_path: str | Path) -> None:
		self.config_path = Path(config_path)
		self.params: dict[str, str] = {}

	def read(self) -> dict[str, str]:
		"""Load configuration values from file into memory."""
		self.params.clear()

		if not self.config_path.exists():
			raise FileNotFoundError(f"Config file not found: {self.config_path}")

		for line_number, raw_line in enumerate(
			self.config_path.read_text(encoding="utf-8").splitlines(), start=1
		):
			line = raw_line.strip()
			if not line:
				continue

			match = self._LINE_PATTERN.match(line)
			if not match:
				raise ValueError(
					f"Invalid config format at line {line_number}: {raw_line}"
				)

			key = match.group("key").strip()
			value = match.group("value")
			self.params[key] = value

		return dict(self.params)

	def write(self) -> None:
		"""Persist in-memory params to file using constants.cfg format."""
		lines = [f'[{key}] "{value}"' for key, value in self.params.items()]
		content = "\n".join(lines)

		if content:
			content += "\n"

		self.config_path.write_text(content, encoding="utf-8")

	def get_param(self, key: str, default: Any = None) -> str | Any:
		"""Get a value by key from loaded params."""
		return self.params.get(key, default)

	def set_param(self, key: str, value: Any) -> None:
		"""Set or update a config value in memory."""
		self.params[key] = str(value)

	def get_int_param(self, key: str, default: int | None = None) -> int:
		"""Get a key as int, useful for numeric params like MAX_SIZE."""
		value = self.params.get(key)
		if value is None:
			if default is None:
				raise KeyError(f"Missing config key: {key}")
			return default

		try:
			return int(value)
		except ValueError as exc:
			raise ValueError(f"Config value for '{key}' is not an int: {value}") from exc

	# Convenience getters/setters for known config keys.
	def get_waveforms_folder(self, default: str | None = None) -> str | None:
		return self.get_param("WAVEFORMS_FOLDER", default)

	def set_waveforms_folder(self, path: str) -> None:
		self.set_param("WAVEFORMS_FOLDER", path)

	def get_output_new_waveforms_folder(self, default: str | None = None) -> str | None:
		return self.get_param("OUTPUT_NEW_WAVEFORMS_FOLDER", default)

	def set_output_new_waveforms_folder(self, path: str) -> None:
		self.set_param("OUTPUT_NEW_WAVEFORMS_FOLDER", path)

	def get_all_waveforms_header(self, default: str | None = None) -> str | None:
		return self.get_param("ALL_WAVEFORMS_HEADER", default)

	def set_all_waveforms_header(self, file_name: str) -> None:
		self.set_param("ALL_WAVEFORMS_HEADER", file_name)

	def get_pattern_header(self, default: str | None = None) -> str | None:
		return self.get_param("PATTERN_HEADER", default)

	def set_pattern_header(self, path: str) -> None:
		self.set_param("PATTERN_HEADER", path)

	def get_max_size(self, default: int | None = None) -> int:
		return self.get_int_param("MAX_SIZE", default)

	def set_max_size(self, size: int) -> None:
		self.set_param("MAX_SIZE", size)

	def get_num_of_waveforms(self, default: int | None = None) -> int:
		return self.get_int_param("NUM_OF_WAVEFORMS", default)

	def set_num_of_waveforms(self, count: int) -> None:
		self.set_param("NUM_OF_WAVEFORMS", count)

	def get_pattern_name(self, default: str | None = None) -> str | None:
		return self.get_param("PATTERN_NAME", default)

	def set_pattern_name(self, name: str) -> None:
		self.set_param("PATTERN_NAME", name)


def load_default_config() -> ConfigManager:
	"""Load config from constants.cfg in the same directory as this module."""
	cfg = ConfigManager(Path(__file__).with_name("constants.cfg"))
	cfg.read()
	return cfg
