"""
Generate a step-by-step pipeline flowchart using custom SVG icons.
Outputs: figures/end_to_end_pipeline_icons.svg
"""

from pathlib import Path

OUT_DIR = Path(__file__).resolve().parents[0]
OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_PATH = OUT_DIR / "end_to_end_pipeline_icons.svg"

WIDTH = 900
HEIGHT = 1600

# Box geometry
BOX_W = 620
BOX_H = 140
START_X = (WIDTH - BOX_W) // 2
START_Y = 60
GAP_Y = 40

COLORS = [
    "#1565c0",  # 1. Data Acquisition
    "#00897b",  # 2. Sequence Cleaning
    "#6a1b9a",  # 3. Metadata Parsing
    "#ef6c00",  # 4. Feature Engineering
    "#c62828",  # 5. Model / Distance
    "#455a64",  # 6. Validation & QA
    "#2e7d32",  # 7. Site Analysis
    "#0277bd",  # 8. Streamlit Dashboard
]

STAGES = [
    {
        "title": "1. Data Acquisition",
        "body": "NCBI / EpiFlu / Manuscripts\nRaw FASTA files",
        "icon": "globe",
    },
    {
        "title": "2. Sequence Cleaning",
        "body": "Remove X/B/Z wildcards\nStrip U/O, drop short fragments",
        "icon": "filter",
    },
    {
        "title": "3. Metadata Parsing",
        "body": "Extract accession, country\nCollection date, lineage",
        "icon": "search",
    },
    {
        "title": "4. Feature Engineering",
        "body": "Path A: AA frequencies + length\nPath B: ESM-2 embeddings",
        "icon": "features",
        "split": True,
    },
    {
        "title": "5. Model Training & Calibration",
        "body": "Classifier training + class centroids\nProbability calibration & risk bands",
        "icon": "tree",
        "split": True,
    },
    {
        "title": "6. Validation & QA",
        "body": "Stratified & grouped CV\nAblation & edge-case panels",
        "icon": "flask",
    },
    {
        "title": "7. Site-Level Analysis",
        "body": "Shannon entropy evaluation\nGlycoprotein alignment mapping",
        "icon": "dna",
    },
    {
        "title": "8. Streamlit Dashboard",
        "body": "Drag-and-drop FASTA upload\nReal-time charts, CSV & PDF export",
        "icon": "dashboard",
    },
]


def make_icon(icon_type: str, x: float, y: float, size: float = 80) -> str:
    """Return SVG path group for a given icon type."""
    cx = x + size / 2
    cy = y + size / 2
    r = size * 0.38
    s = size

    if icon_type == "globe":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round">
            <circle cx="{s/2}" cy="{s/2}" r="{r}" />
            <ellipse cx="{s/2}" cy="{s/2}" rx="{r*0.45}" ry="{r}" />
            <line x1="{s/2-r}" y1="{s/2}" x2="{s/2+r}" y2="{s/2}" />
            <line x1="{s/2}" y1="{s/2-r}" x2="{s/2}" y2="{s/2+r}" />
        </g>'''

    if icon_type == "filter":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <polygon points="{s*0.25},{s*0.2} {s*0.75},{s*0.2} {s*0.55},{s*0.5} {s*0.55},{s*0.85} {s*0.45},{s*0.85} {s*0.45},{s*0.5}" />
        </g>'''

    if icon_type == "search":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round">
            <circle cx="{s*0.42}" cy="{s*0.42}" r="{s*0.22}" />
            <line x1="{s*0.58}" y1="{s*0.58}" x2="{s*0.8}" y2="{s*0.8}" />
            <rect x="{s*0.18}" y="{s*0.72}" width="{s*0.64}" height="{s*0.14}" rx="2" />
        </g>'''

    if icon_type == "features":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <rect x="{s*0.15}" y="{s*0.55}" width="{s*0.15}" height="{s*0.30}" rx="2" fill="white" />
            <rect x="{s*0.42}" y="{s*0.35}" width="{s*0.15}" height="{s*0.50}" rx="2" fill="white" />
            <rect x="{s*0.70}" y="{s*0.45}" width="{s*0.15}" height="{s*0.40}" rx="2" fill="white" />
            <circle cx="{s*0.75}" cy="{s*0.25}" r="{s*0.10}" fill="white" />
        </g>'''

    if icon_type == "tree":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <line x1="{s/2}" y1="{s*0.20}" x2="{s/2}" y2="{s*0.45}" />
            <line x1="{s/2}" y1="{s*0.45}" x2="{s*0.30}" y2="{s*0.65}" />
            <line x1="{s/2}" y1="{s*0.45}" x2="{s*0.70}" y2="{s*0.65}" />
            <line x1="{s*0.30}" y1="{s*0.65}" x2="{s*0.20}" y2="{s*0.85}" />
            <line x1="{s*0.30}" y1="{s*0.65}" x2="{s*0.40}" y2="{s*0.85}" />
            <line x1="{s*0.70}" y1="{s*0.65}" x2="{s*0.60}" y2="{s*0.85}" />
            <line x1="{s*0.70}" y1="{s*0.65}" x2="{s*0.80}" y2="{s*0.85}" />
            <circle cx="{s/2}" cy="{s*0.20}" r="{s*0.07}" fill="white" />
        </g>'''

    if icon_type == "flask":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <path d="M {s*0.35} {s*0.15} L {s*0.35} {s*0.45} L {s*0.20} {s*0.85} L {s*0.80} {s*0.85} L {s*0.65} {s*0.45} L {s*0.65} {s*0.15}" />
            <line x1="{s*0.30}" y1="{s*0.15}" x2="{s*0.70}" y2="{s*0.15}" />
            <line x1="{s*0.25}" y1="{s*0.68}" x2="{s*0.75}" y2="{s*0.68}" stroke-dasharray="4,3" />
        </g>'''

    if icon_type == "dna":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round">
            <path d="M {s*0.30} {s*0.15} Q {s*0.55} {s*0.30} {s*0.30} {s*0.50} Q {s*0.05} {s*0.70} {s*0.30} {s*0.85}" />
            <path d="M {s*0.70} {s*0.15} Q {s*0.45} {s*0.30} {s*0.70} {s*0.50} Q {s*0.95} {s*0.70} {s*0.70} {s*0.85}" />
            <line x1="{s*0.38}" y1="{s*0.32}" x2="{s*0.62}" y2="{s*0.32}" />
            <line x1="{s*0.38}" y1="{s*0.68}" x2="{s*0.62}" y2="{s*0.68}" />
        </g>'''

    if icon_type == "dashboard":
        return f'''<g transform="translate({x},{y})" fill="none" stroke="white" stroke-width="3" stroke-linecap="round" stroke-linejoin="round">
            <rect x="{s*0.15}" y="{s*0.15}" width="{s*0.70}" height="{s*0.70}" rx="4" />
            <line x1="{s*0.15}" y1="{s*0.32}" x2="{s*0.85}" y2="{s*0.32}" />
            <circle cx="{s*0.72}" cy="{s*0.235}" r="{s*0.05}" fill="white" />
            <circle cx="{s*0.80}" cy="{s*0.235}" r="{s*0.05}" fill="white" />
            <rect x="{s*0.25}" y="{s*0.45}" width="{s*0.18}" height="{s*0.30}" rx="2" fill="white" />
            <rect x="{s*0.50}" y="{s*0.55}" width="{s*0.18}" height="{s*0.20}" rx="2" fill="white" />
        </g>'''

    return ""


def box_svg(x: float, y: float, w: float, h: float, color: str, title: str, body: str, icon: str) -> str:
    icon_x = x + 20
    icon_y = y + (h - 80) / 2
    text_x = x + 110
    text_y_title = y + 35
    text_y_body = y + 65
    body_lines = body.split('\n')
    body_tspans = ''.join(
        f'<tspan x="{text_x}" dy="{18 if i == 0 else 18}">{line}</tspan>'
        for i, line in enumerate(body_lines)
    )

    return f'''<g>
        <rect x="{x}" y="{y}" width="{w}" height="{h}" rx="12" ry="12" fill="{color}" stroke="white" stroke-width="2" />
        {make_icon(icon, icon_x, icon_y)}
        <text x="{text_x}" y="{text_y_title}" font-family="Arial, Helvetica, sans-serif" font-size="18" font-weight="bold" fill="white">{title}</text>
        <text x="{text_x}" y="{text_y_body}" font-family="Arial, Helvetica, sans-serif" font-size="13" fill="white">
            {body_tspans}
        </text>
    </g>'''


def arrow_svg(x1: float, y1: float, x2: float, y2: float, color: str = "#37474f") -> str:
    return f'''<g>
        <line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="3" marker-end="url(#arrowhead)" />
    </g>'''


def build_svg() -> str:
    parts = []
    parts.append(f'''<svg xmlns="http://www.w3.org/2000/svg" width="{WIDTH}" height="{HEIGHT}" viewBox="0 0 {WIDTH} {HEIGHT}" style="background-color:#f5f5f5;">
        <defs>
            <marker id="arrowhead" markerWidth="10" markerHeight="7" refX="9" refY="3.5" orient="auto">
                <polygon points="0 0, 10 3.5, 0 7" fill="#37474f" />
            </marker>
            <filter id="shadow" x="-20%" y="-20%" width="140%" height="140%">
                <feDropShadow dx="2" dy="3" stdDeviation="3" flood-color="#000" flood-opacity="0.2" />
            </filter>
        </defs>
        <text x="{WIDTH/2}" y="35" font-family="Arial, Helvetica, sans-serif" font-size="24" font-weight="bold" text-anchor="middle" fill="#263238">End-to-End Viral Surveillance Pipeline</text>
    ''')

    positions = []
    for i, stage in enumerate(STAGES):
        if stage.get("split"):
            # Split into two boxes side-by-side for A/B tracks
            half_w = (BOX_W - 30) / 2
            y = START_Y + i * (BOX_H + GAP_Y)
            x_left = START_X
            x_right = START_X + half_w + 30

            parts.append(box_svg(x_left, y, half_w, BOX_H, COLORS[i], stage["title"] + " A", "AA frequencies + length", "features"))
            parts.append(box_svg(x_right, y, half_w, BOX_H, "#d84315", stage["title"] + " B", "ESM-2 embeddings\n(esm2_t12_35M)", "tree"))
            positions.append((i, (x_left + half_w / 2, y + BOX_H), (x_right + half_w / 2, y + BOX_H)))
        else:
            y = START_Y + i * (BOX_H + GAP_Y)
            x = START_X
            parts.append(box_svg(x, y, BOX_W, BOX_H, COLORS[i], stage["title"], stage["body"], stage["icon"]))
            positions.append((i, (x + BOX_W / 2, y + BOX_H),))

    # Draw arrows
    for i in range(len(positions) - 1):
        curr = positions[i]
        nxt = positions[i + 1]

        if len(curr) == 2:  # single source
            src_x, src_y = curr[1][0], curr[1][1]
            if len(nxt) == 2:  # single destination
                dst_x, dst_y = nxt[1][0], nxt[1][1] - BOX_H
                parts.append(arrow_svg(src_x, src_y + 2, dst_x, dst_y - 2))
            else:  # split destination
                dst_y = nxt[1][1] - BOX_H
                parts.append(arrow_svg(src_x, src_y + 2, nxt[1][0], dst_y - 2))
                parts.append(arrow_svg(src_x, src_y + 2, nxt[2][0], dst_y - 2))
        else:  # split source
            src_y = curr[1][1]
            dst_y = nxt[1][1] - BOX_H
            if len(nxt) == 2:
                parts.append(arrow_svg(curr[1][0], src_y + 2, nxt[1][0], dst_y - 2))
                parts.append(arrow_svg(curr[2][0], src_y + 2, nxt[1][0], dst_y - 2))
            else:
                parts.append(arrow_svg(curr[1][0], src_y + 2, nxt[1][0], dst_y - 2))
                parts.append(arrow_svg(curr[2][0], src_y + 2, nxt[2][0], dst_y - 2))

    parts.append("</svg>")
    return "\n".join(parts)


if __name__ == "__main__":
    svg_content = build_svg()
    OUT_PATH.write_text(svg_content, encoding="utf-8")
    print(f"Saved SVG flowchart: {OUT_PATH}")
