"""Rebuild the paper's two-jet correlation figure with native LaTeX.

Run from a checkout: python examples/plot_joint_validation_latex.py
Requires NumPy, matplotlib for the original color palette, and pdflatex with pgfplots.
The command writes a standalone TeX source and the corresponding PDF.
"""
import argparse
from pathlib import Path
import shutil
import subprocess
import tempfile

import matplotlib
import numpy as np


def heatmap(data, model, reference, palette):
    edges = data["edges"]
    counts, sumw2 = data[model], data[model + "_w2"]
    fractions = counts / counts.sum()
    commands = []
    for i, j in zip(*np.nonzero(reference > 0)):
        left, right = edges[i:i + 2]
        bottom, top = edges[j:j + 2]
        ratio = fractions[i, j] / reference[i, j]
        effective = counts[i, j]**2 / sumw2[i, j] if sumw2[i, j] > 0 else 0.
        if j > i:
            raise ValueError("reference contains an unordered jet pair")
        sparse = effective < 10
        rgb = palette(np.clip((ratio - .6) / .8, 0., 1.))[:3]
        color = "eeeeee" if sparse else "".join(f"{round(255 * c):02x}" for c in rgb)
        cell = "diagonal" if i == j else "cell"
        commands.append(rf"\{cell}{{{color}}}{{{left:g}}}{{{right:g}}}{{{bottom:g}}}{{{top:g}}}{{{int(sparse)}}}")
        if (left, right, bottom, top) == (220, 400, 90, 140):
            if sparse:
                raise ValueError("the annotated cell is below the effective-statistics requirement")
            ink = "white" if np.dot(rgb, [.2126, .7152, .0722]) < .5 else "black"
            commands.append(rf"\node[text={ink},font=\small] at (axis cs:{np.sqrt(left * right):g},"
                            rf"{np.sqrt(bottom * top):g}) {{{ratio:.2f}}};")
    return "\n".join(commands)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path(__file__).resolve().parent / "data/zj_joint_validation.npz")
    parser.add_argument("--output", type=Path, default=Path("zj_joint_validation_latex.pdf"))
    args = parser.parse_args()
    pdf = args.output.with_suffix(".pdf").resolve()
    pdf.parent.mkdir(parents=True, exist_ok=True)
    palette = matplotlib.colormaps["RdBu_r"]
    template = Path(__file__).with_suffix(".tex").read_text()
    with np.load(args.data, allow_pickle=False) as data:
        fo = data["fo_per_run_fb"][..., 0].mean(axis=0)
        reference = fo / fo.sum()
        for key in ("marg", "joint"):
            template = template.replace("% CELLS " + key, heatmap(data, key, reference, palette))
    # The visible interval of a pgf shading is 25 to 75 bp.
    stops = []
    for position in [0., *np.linspace(25., 75., 11), 100.]:
        coord = np.clip((position - 25.) / 50., 0., 1.)
        r, g, b = palette(coord)[:3]
        stops.append(f"rgb({position:g}bp)=({r:.5f},{g:.5f},{b:.5f})")
    template = template.replace("% SHADING STOPS", ";\n".join(stops))
    source = pdf.with_suffix(".tex")
    source.write_text(template)
    with tempfile.TemporaryDirectory() as directory:
        process = subprocess.run(["pdflatex", "-halt-on-error", "-interaction=nonstopmode", str(source)],
                                 cwd=directory, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if process.returncode:
            raise RuntimeError(process.stdout[-3000:])
        shutil.copy2(Path(directory) / pdf.name, pdf)
    print(f"Wrote {pdf} and {source}.")


if __name__ == "__main__":
    main()
