#!/usr/bin/env python3
"""Draw the release model card's SVG diagram of what changed against GSQ.

The pruned experts (layer 0 shown) and the upgraded tensors come from the
candidate build report; the byte figures are the plan's budget arithmetic.

    draw_qwen36_release_diagram.py reports/qwen36_gsq_rco_v2_candidate_gguf.json OUT.svg
"""
import json
import sys
from html import escape

report = json.load(open(sys.argv[1]))
out = sys.argv[2]
pruned0 = set(report["pruned_experts_per_layer"][0])
upgraded = {}
for name in report["upgraded_tensors"]:
    _, layer, kind, _ = name.split(".")
    upgraded.setdefault(kind.split("_")[1], set()).add(int(layer))
assert len(pruned0) == 24 and sum(len(v) for v in upgraded.values()) == 23

W, H = 960, 600
INK, MUTED, LINE = "#1f2937", "#5b6472", "#c4cad3"
GRAY, GRAYF = "#e6e9ee", "#f4f6f8"
RED, REDF = "#d1495b", "#fbe3e6"
BLUE, BLUEF = "#2f6db5", "#dde9f7"
ORANGE, ORANGEF = "#d9822b", "#fcebd9"
parts = []


def text(x, y, s, size=13, weight="normal", fill=INK, anchor="start"):
    parts.append(f'<text x="{x}" y="{y}" font-size="{size}" font-weight="{weight}" '
                 f'fill="{fill}" text-anchor="{anchor}">{escape(s)}</text>')


def box(x, y, w, h, fill, stroke, r=6, sw=1.2):
    parts.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" '
                 f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')


def arrow(x, y1, y2):
    parts.append(f'<line x1="{x}" y1="{y1}" x2="{x}" y2="{y2 - 6}" stroke="{MUTED}" stroke-width="1.5"/>'
                 f'<path d="M{x - 4},{y2 - 7} L{x + 4},{y2 - 7} L{x},{y2} Z" fill="{MUTED}"/>')


box(0, 0, W, H, "#ffffff", "#ffffff", r=0)
text(24, 36, "Qwen3.6-35B-A3B GSQ-RCO-E6-P24-U23", 20, "bold")
text(24, 58, "What changed compared with the 2-bit GSQ GGUF", 13, fill=MUTED)

# Left: the model, top to bottom.
L, LW = 24, 420
box(L, 78, LW, 40, ORANGEF, ORANGE)
text(L + 12, 103, "Token embedding", 13, "bold")
text(L + LW - 12, 103, "BF16 → Q6_K  (E6, CPU side)", 13, fill=ORANGE, anchor="end")
arrow(L + LW / 2, 118, 134)

box(L, 134, LW, 368, "#ffffff", LINE, r=8)
text(L + 12, 156, "Decoder layer × 40", 13, "bold")
text(L + LW - 12, 156, "10 × (3 linear-attention + 1 full-attention)", 12, fill=MUTED, anchor="end")
box(L + 12, 166, LW - 24, 30, GRAYF, LINE)
text(L + 24, 186, "Attention / linear attention", 12)
text(L + LW - 24, 186, "Q8_0, unchanged", 12, fill=MUTED, anchor="end")
box(L + 12, 202, LW - 24, 30, REDF, RED)
text(L + 24, 222, "Router", 12)
text(L + LW - 24, 222, "256 → 232 outputs", 12, fill=RED, anchor="end")
box(L + 12, 238, LW - 24, 30, GRAYF, LINE)
text(L + 24, 258, "Shared expert", 12)
text(L + LW - 24, 258, "Q8_0, unchanged", 12, fill=MUTED, anchor="end")

box(L + 12, 274, LW - 24, 218, GRAYF, LINE)
text(L + 24, 294, "Routed experts (8 active per token)", 12, "bold")
gx, gy, pitch, cell = L + 24, 304, 11, 9
for i in range(256):
    x, y = gx + (i % 16) * pitch, gy + (i // 16) * pitch
    if i in pruned0:
        box(x, y, cell, cell, REDF, RED, r=1.5, sw=1)
        parts.append(f'<path d="M{x + 2},{y + 2} L{x + cell - 2},{y + cell - 2} M{x + cell - 2},{y + 2} '
                     f'L{x + 2},{y + cell - 2}" stroke="{RED}" stroke-width="1.2"/>')
    else:
        box(x, y, cell, cell, "#9aa5b4", "#9aa5b4", r=1.5, sw=0.5)
tx = gx + 16 * pitch + 14
text(tx, 320, "256 experts per layer", 12)
text(tx, 340, "24 pruned (P24), 232 kept", 12, "bold", fill=RED)
text(tx, 358, "layer 0 shown; the 24 pruned", 11, fill=MUTED)
text(tx, 373, "experts differ per layer", 11, fill=MUTED)
text(tx, 403, "Expert weights: 3 tensors", 12)
text(tx, 421, "per layer (gate, up, down),", 11, fill=MUTED)
text(tx, 436, "120 in total", 11, fill=MUTED)
text(tx, 460, "Q2_0 (GSQ 2-bit);", 12)
text(tx, 478, "23 tensors → Q3_K (U23)", 12, "bold", fill=BLUE)
arrow(L + LW / 2, 502, 518)
box(L, 518, LW, 40, GRAYF, LINE)
text(L + 12, 543, "Output norm + LM head", 13, "bold")
text(L + LW - 12, 543, "Q8_0, unchanged", 13, fill=MUTED, anchor="end")

# Right: where the 23 upgrades are.
R = 480
text(R, 98, "Where the 23 Q3_K upgrades are", 15, "bold")
text(R, 118, "Routed-expert tensors by layer: ■ Q3_K   □ Q2_0", 12, fill=MUTED)
x0, pitch, cell = R + 64, 9.6, 8
for layer in range(0, 40, 5):
    text(x0 + layer * pitch + cell / 2, 146, str(layer), 10, fill=MUTED, anchor="middle")
text(x0 + 39 * pitch + cell / 2, 146, "39", 10, fill=MUTED, anchor="middle")
rows = [("attention", None), ("gate", "gate"), ("up", "up"), ("down", "down")]
for r, (label, kind) in enumerate(rows):
    y = 156 + r * 22
    text(x0 - 8, y + 12, label, 12, anchor="end")
    for layer in range(40):
        x = x0 + layer * pitch
        if kind is None:
            full = layer % 4 == 3
            box(x, y + 2, cell, 12, "#5b6472" if full else GRAY, "#5b6472" if full else LINE, r=1.5, sw=0.8)
        elif layer in upgraded[kind]:
            box(x, y, cell, 16, BLUE, BLUE, r=1.5, sw=0.8)
        else:
            box(x, y, cell, 16, "#ffffff", LINE, r=1.5, sw=0.8)
text(x0, 262, "attention row: ■ full attention  □ linear attention (Gated DeltaNet)", 11, fill=MUTED)
counts = {k: len(v) for k, v in upgraded.items()}
text(x0, 282, f"down {counts['down']}, gate {counts['gate']}, up {counts['up']}; "
              "mostly in the last 10 layers", 11, fill=MUTED)

# Right, lower: the byte budget.
text(R, 330, "Byte budget", 15, "bold")
rowsb = [
    ("Pruning frees (GPU side)", "−857 MB", RED),
    ("23 Q3_K upgrades cost", "+831 MB", BLUE),
    ("Net on the GPU side vs GSQ", "−27 MB", INK),
    ("Q6_K embedding (CPU side)", "−600 MB", ORANGE),
]
for i, (label, value, color) in enumerate(rowsb):
    y = 356 + i * 26
    text(R, y, label, 12)
    text(R + 400, y, value, 12, "bold", fill=color, anchor="end")
parts.append(f'<line x1="{R}" y1="{356 + 3 * 26 + 12}" x2="{R + 400}" y2="{356 + 3 * 26 + 12}" '
             f'stroke="{LINE}" stroke-width="1"/>')
text(R, 476, "File size", 12, "bold")
text(R + 400, 476, "11.59 GB  (GSQ 12.22 GB)", 12, "bold", anchor="end")
text(R, 504, "Everything not marked is byte-identical to the GSQ GGUF.", 11, fill=MUTED)
text(R, 520, "1 GB = 10⁹ bytes.", 11, fill=MUTED)

# Legend.
ly = 580
for i, (fill, stroke, label) in enumerate([(GRAYF, LINE, "unchanged"), (REDF, RED, "pruned"),
                                           (BLUEF, BLUE, "upgraded Q2_0 → Q3_K"),
                                           (ORANGEF, ORANGE, "re-quantized BF16 → Q6_K")]):
    x = 24 + i * 200
    box(x, ly - 11, 14, 14, fill, stroke, r=3)
    text(x + 22, ly, label, 12)

svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" viewBox="0 0 {W} {H}" '
       f'font-family="Helvetica, Arial, sans-serif" role="img" '
       f'aria-label="Diagram of the changes in Qwen3.6-35B-A3B GSQ-RCO-E6-P24-U23">'
       + "".join(parts) + "</svg>\n")
open(out, "w").write(svg)
print("wrote", out, len(svg), "bytes")
