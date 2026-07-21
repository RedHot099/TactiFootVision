from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle


OUT_DIR = Path(__file__).resolve().parent
PNG = OUT_DIR / "homography_improvement_slide.png"
PDF = OUT_DIR / "homography_improvement_slide.pdf"


def add_card(ax, x, y, w, h, title, before, after, delta, accent):
    card = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#D7DEE8",
        facecolor="#FFFFFF",
    )
    ax.add_patch(card)
    ax.text(x + 0.03, y + h - 0.08, title, fontsize=21, weight="bold", color="#102033")
    ax.text(x + 0.03, y + h - 0.20, "current", fontsize=13, color="#667085")
    ax.text(x + 0.03, y + h - 0.31, before, fontsize=31, weight="bold", color="#C2410C")
    ax.text(x + 0.03, y + 0.20, "oracle control", fontsize=13, color="#667085")
    ax.text(x + 0.03, y + 0.09, after, fontsize=31, weight="bold", color=accent)
    ax.text(x + w - 0.03, y + 0.11, delta, fontsize=19, weight="bold", color=accent, ha="right")


def add_bar(ax, x, y, w, h, label, current, oracle, suffix="%"):
    ax.text(x, y + h + 0.035, label, fontsize=17, weight="bold", color="#102033")
    ax.add_patch(Rectangle((x, y + h * 0.53), w, h * 0.22, color="#F1F5F9"))
    ax.add_patch(Rectangle((x, y + h * 0.53), w * current / 100.0, h * 0.22, color="#F97316"))
    ax.text(x + w + 0.02, y + h * 0.56, f"{current:.2f}{suffix}", fontsize=14, color="#9A3412")
    ax.add_patch(Rectangle((x, y + h * 0.08), w, h * 0.22, color="#F1F5F9"))
    ax.add_patch(Rectangle((x, y + h * 0.08), w * oracle / 100.0, h * 0.22, color="#059669"))
    ax.text(x + w + 0.02, y + h * 0.11, f"{oracle:.2f}{suffix}", fontsize=14, color="#047857")


def main():
    fig = plt.figure(figsize=(16, 9), dpi=120)
    ax = plt.axes([0, 0, 1, 1])
    ax.set_axis_off()
    fig.patch.set_facecolor("#F7F9FC")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    ax.add_patch(Rectangle((0, 0.91), 1, 0.09, color="#102033"))
    ax.text(
        0.04,
        0.948,
        "Homografia: z obecnego baseline'u do wiarygodnej projekcji image→pitch",
        fontsize=27,
        weight="bold",
        color="white",
        va="center",
    )
    ax.text(
        0.04,
        0.885,
        "SoccerNet-GSR valid · 58 sekwencji · 43,500 klatek · GT footpoints izolują błąd homografii",
        fontsize=16,
        color="#475467",
    )

    add_card(ax, 0.04, 0.56, 0.29, 0.25, "Mediana błędu", "93.7 m", "0.096 m", "973× mniej", "#059669")
    add_card(ax, 0.355, 0.56, 0.29, 0.25, "Success@2m", "0.041%", "99.35%", "+99.31 pp", "#059669")
    add_card(ax, 0.67, 0.56, 0.29, 0.25, "Dostępność", "23.3%", "98.1%", "+74.8 pp", "#059669")

    panel = FancyBboxPatch(
        (0.04, 0.18),
        0.58,
        0.30,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#D7DEE8",
        facecolor="#FFFFFF",
    )
    ax.add_patch(panel)
    ax.text(0.065, 0.435, "Co pokazuje eksperyment", fontsize=21, weight="bold", color="#102033")
    bullets = [
        "Obecny baseline interpretuje ludzkie keypointy YOLO-pose jako punkty boiska.",
        "Pipeline ewaluacji działa: oracle-control osiąga sub-metrowy p90 błędu.",
        "Następny krok: podmienić YOLO-pose na realny backend kalibracji: PnLCalib / Sportlight.",
    ]
    y = 0.38
    for item in bullets:
        ax.text(0.075, y, "•", fontsize=23, color="#059669", va="center")
        ax.text(0.095, y, item, fontsize=15.5, color="#344054", va="center")
        y -= 0.075

    panel2 = FancyBboxPatch(
        (0.66, 0.18),
        0.30,
        0.30,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.2,
        edgecolor="#D7DEE8",
        facecolor="#FFFFFF",
    )
    ax.add_patch(panel2)
    ax.text(0.685, 0.435, "Skala zmiany", fontsize=21, weight="bold", color="#102033")
    add_bar(ax, 0.685, 0.315, 0.20, 0.08, "Success@2m", 0.0411, 99.35)
    add_bar(ax, 0.685, 0.205, 0.20, 0.08, "Availability", 23.27, 98.10)
    ax.text(0.685, 0.185, "orange = current · green = oracle control", fontsize=11.5, color="#667085")

    ax.add_patch(Rectangle((0.04, 0.06), 0.92, 0.065, color="#EAF5F0"))
    ax.text(
        0.06,
        0.092,
        "Wniosek: current_yolopose_7pt zostaje tylko baseline'em historycznym; produkcyjnie podpinamy zewnętrzny backend kalibracji.",
        fontsize=17,
        weight="bold",
        color="#064E3B",
        va="center",
    )
    ax.text(
        0.96,
        0.025,
        "Źródło: results/experiments/homography_comparison_valid_current_oracle",
        fontsize=10.5,
        color="#667085",
        ha="right",
    )

    fig.savefig(PNG, dpi=120)
    fig.savefig(PDF)
    print(PNG)
    print(PDF)


if __name__ == "__main__":
    main()

