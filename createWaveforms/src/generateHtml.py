"""Generate an HTML file visualizing waveform headers from OUTPUT_NEW_WAVEFORMS_FOLDER."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

from config import load_default_config


def parse_header_file(filepath: Path) -> tuple[str, list[float]] | None:
	"""Parse a C++ header and return (array_name, values)."""
	content = filepath.read_text(encoding="utf-8")

	match = re.search(
		r"const\s+float\s+(\w+)\s*\[.*?\]\s*=\s*\{([^}]+)\}",
		content,
		re.DOTALL,
	)
	if not match:
		return None

	name = match.group(1)
	raw_values = match.group(2)
	values = [float(v.strip()) for v in raw_values.split(",") if v.strip()]
	return name, values


def collect_waveforms(headers_dir: Path, downsample: int) -> list[dict[str, object]]:
	"""Collect all waveform headers and sampled values from a directory."""
	waveforms: list[dict[str, object]] = []
	files = sorted(headers_dir.glob("*.h"), key=lambda p: p.name)

	for filepath in files:
		parsed = parse_header_file(filepath)
		if not parsed:
			print(f"Warning: could not parse {filepath.name}")
			continue

		name, values = parsed
		sampled = values[::downsample]
		waveforms.append(
			{
				"name": name,
				"filename": filepath.name,
				"num_samples": len(values),
				"data": sampled,
			}
		)

	return waveforms


def generate_html(waveforms: list[dict[str, object]], source_dir: Path, downsample: int) -> str:
	cards_html: list[str] = []
	charts_js: list[str] = []

	colors = [
		"#4f8ef7",
		"#e05c5c",
		"#4caf50",
		"#ff9800",
		"#9c27b0",
		"#00bcd4",
		"#ff5722",
		"#795548",
		"#607d8b",
		"#f06292",
	]

	for idx, wf in enumerate(waveforms):
		name = str(wf["name"])
		filename = str(wf["filename"])
		num_samples = int(wf["num_samples"])
		data = wf["data"]
		color = colors[idx % len(colors)]
		canvas_id = f"chart_{name}"
		data_var = f"data_{name}"

		cards_html.append(
			f"""
		<div class=\"card\">
			<div class=\"card-header\">
				<span class=\"wf-name\">{name}</span>
				<span class=\"wf-meta\">{filename} &nbsp;|&nbsp; {num_samples} samples</span>
			</div>
			<canvas id=\"{canvas_id}\" height=\"120\"></canvas>
		</div>"""
		)

		data_json = json.dumps(data)
		charts_js.append(
			f"""
	(function() {{
		var {data_var} = {data_json};
		var labels = Array.from({{length: {data_var}.length}}, function(_, i) {{ return i * {downsample}; }});
		new Chart(document.getElementById('{canvas_id}').getContext('2d'), {{
			type: 'line',
			data: {{
				labels: labels,
				datasets: [{{
					data: {data_var},
					borderColor: '{color}',
					borderWidth: 1.5,
					pointRadius: 0,
					fill: false,
					tension: 0.1
				}}]
			}},
			options: {{
				animation: false,
				responsive: true,
				plugins: {{
					legend: {{ display: false }},
					tooltip: {{
						mode: 'index',
						intersect: false,
						callbacks: {{
							title: function(items) {{ return 'Sample ' + items[0].label; }},
							label: function(item) {{ return Number(item.raw).toFixed(6); }}
						}}
					}}
				}},
				scales: {{
					x: {{
						ticks: {{ maxTicksLimit: 8, color: '#aaa', font: {{size: 10}} }},
						grid: {{ color: '#333' }}
					}},
					y: {{
						min: -1.1,
						max: 1.1,
						ticks: {{ stepSize: 0.5, color: '#aaa', font: {{size: 10}} }},
						grid: {{ color: '#333' }}
					}}
				}}
			}}
		}});
	}})();"""
		)

	cards_block = "\n".join(cards_html)
	charts_block = "\n".join(charts_js)

	return f"""<!DOCTYPE html>
<html lang=\"en\">
<head>
	<meta charset=\"UTF-8\">
	<meta name=\"viewport\" content=\"width=device-width, initial-scale=1.0\">
	<title>Generated Waveform Lookup Tables</title>
	<script src=\"https://cdn.jsdelivr.net/npm/chart.js@4\"></script>
	<style>
		* {{ box-sizing: border-box; margin: 0; padding: 0; }}
		body {{
			background: #1a1a2e;
			color: #e0e0e0;
			font-family: 'Segoe UI', system-ui, sans-serif;
			padding: 24px;
		}}
		h1 {{
			text-align: center;
			font-size: 1.6rem;
			margin-bottom: 8px;
			color: #90caf9;
		}}
		.subtitle {{
			text-align: center;
			font-size: 0.85rem;
			color: #888;
			margin-bottom: 28px;
		}}
		.grid {{
			display: grid;
			grid-template-columns: repeat(auto-fill, minmax(440px, 1fr));
			gap: 20px;
		}}
		.card {{
			background: #16213e;
			border: 1px solid #0f3460;
			border-radius: 10px;
			padding: 14px 16px 10px;
		}}
		.card-header {{
			display: flex;
			justify-content: space-between;
			align-items: baseline;
			margin-bottom: 8px;
		}}
		.wf-name {{
			font-size: 1rem;
			font-weight: 600;
			color: #90caf9;
		}}
		.wf-meta {{
			font-size: 0.72rem;
			color: #666;
		}}
	</style>
</head>
<body>
	<h1>Generated Waveform Lookup Tables</h1>
	<p class=\"subtitle\">
		{len(waveforms)} waveforms &nbsp;|&nbsp;
		Source: <code>{source_dir}</code>
		&nbsp;|&nbsp; displayed every {downsample} samples
	</p>
	<div class=\"grid\">
		{cards_block}
	</div>
	<script>
	{charts_block}
	</script>
</body>
</html>"""


def main(argv: list[str] | None = None) -> None:
	parser = argparse.ArgumentParser(
		description="Generate an HTML waveform preview from header files"
	)
	parser.add_argument(
		"--downsample",
		type=int,
		default=4,
		help="Keep every Nth sample in chart data (default: 4)",
	)
	parser.add_argument(
		"--output",
		type=str,
		default=None,
		help="Optional output HTML path (default: <headers-dir>/waveforms.html)",
	)
	parser.add_argument(
		"--headers-dir",
		type=str,
		default=None,
		help="Optional headers folder path (default: OUTPUT_NEW_WAVEFORMS_FOLDER from constants.cfg)",
	)
	args = parser.parse_args(argv)

	if args.downsample <= 0:
		raise ValueError("--downsample must be > 0")

	if args.headers_dir:
		headers_dir = Path(args.headers_dir)
	else:
		cfg = load_default_config()
		headers_dir_str = cfg.get_output_new_waveforms_folder()
		if not headers_dir_str:
			raise ValueError("Missing OUTPUT_NEW_WAVEFORMS_FOLDER in constants.cfg")
		headers_dir = Path(headers_dir_str)

	if not headers_dir.exists():
		raise FileNotFoundError(f"Headers folder not found: {headers_dir}")

	output_html = Path(args.output) if args.output else (headers_dir / "waveforms.html")

	print(f"Scanning: {headers_dir}")
	waveforms = collect_waveforms(headers_dir, args.downsample)
	print(f"Parsed {len(waveforms)} waveform files.")

	html = generate_html(waveforms, headers_dir, args.downsample)
	output_html.write_text(html, encoding="utf-8")

	print(f"Generated: {output_html}")

