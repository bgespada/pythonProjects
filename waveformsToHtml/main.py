"""
Generate an HTML file visualizing all wfXXX.h waveform lookup tables
from C:\\DaisyProjects\\lib\\samples\\waveforms\\
"""

import os
import re
import json

WAVEFORMS_DIR = r"C:\DaisyProjects\lib\samples\waveforms"
OUTPUT_HTML = os.path.join(os.path.dirname(__file__), "waveforms.html")
# Downsample factor – keep every Nth sample to reduce HTML size (1 = no downsampling)
DOWNSAMPLE = 4


def parse_header_file(filepath: str) -> tuple[str, list[float]] | None:
    """Parse a wfXXX.h file and return (array_name, values)."""
    with open(filepath, "r") as f:
        content = f.read()

    # Extract array name and values from lines like:
    # const float WF001[WF001_SIZE] = {0.0, 1.0, ...};
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


def collect_waveforms() -> list[dict]:
    """Collect all parsed waveforms, sorted by filename."""
    waveforms = []
    files = sorted(
        f for f in os.listdir(WAVEFORMS_DIR) if re.match(r"wf\d+\.h$", f, re.IGNORECASE)
    )
    for filename in files:
        filepath = os.path.join(WAVEFORMS_DIR, filename)
        result = parse_header_file(filepath)
        if result:
            name, values = result
            # Downsample
            sampled = values[::DOWNSAMPLE]
            waveforms.append({"name": name, "filename": filename, "data": sampled})
        else:
            print(f"Warning: could not parse {filename}")
    return waveforms


def generate_html(waveforms: list[dict]) -> str:
    charts_js = []
    cards_html = []

    colors = [
        "#4f8ef7", "#e05c5c", "#4caf50", "#ff9800", "#9c27b0",
        "#00bcd4", "#ff5722", "#795548", "#607d8b", "#f06292",
    ]

    for i, wf in enumerate(waveforms):
        name = wf["name"]
        filename = wf["filename"]
        color = colors[i % len(colors)]
        canvas_id = f"chart_{name}"
        data_var = f"data_{name}"
        num_samples = len(wf["data"]) * DOWNSAMPLE

        # Build the card HTML
        cards_html.append(f"""
        <div class="card">
            <div class="card-header">
                <span class="wf-name">{name}</span>
                <span class="wf-meta">{filename} &nbsp;|&nbsp; {num_samples} samples</span>
            </div>
            <canvas id="{canvas_id}" height="120"></canvas>
        </div>""")

        # Build the JS for this chart
        data_json = json.dumps(wf["data"])
        charts_js.append(f"""
    (function() {{
        var {data_var} = {data_json};
        var labels = Array.from({{length: {data_var}.length}}, function(_, i) {{ return i * {DOWNSAMPLE}; }});
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
                            label: function(item) {{ return item.raw.toFixed(6); }}
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
    }})();""")

    cards_block = "\n".join(cards_html)
    charts_block = "\n".join(charts_js)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Waveform Lookup Tables</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js@4"></script>
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
    <h1>Waveform Lookup Tables</h1>
    <p class="subtitle">
        {len(waveforms)} waveforms &nbsp;|&nbsp;
        Source: <code>C:\\DaisyProjects\\lib\\samples\\waveforms\\</code>
        &nbsp;|&nbsp; displayed every {DOWNSAMPLE} samples
    </p>
    <div class="grid">
        {cards_block}
    </div>
    <script>
    {charts_block}
    </script>
</body>
</html>"""
    return html


def main():
    print(f"Scanning: {WAVEFORMS_DIR}")
    waveforms = collect_waveforms()
    print(f"Parsed {len(waveforms)} waveform files.")

    html = generate_html(waveforms)

    with open(OUTPUT_HTML, "w", encoding="utf-8") as f:
        f.write(html)

    print(f"Generated: {OUTPUT_HTML}")


if __name__ == "__main__":
    main()
