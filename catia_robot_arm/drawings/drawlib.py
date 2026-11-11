"""Small 2D drafting helper library (matplotlib + numpy only).

All coordinates handed to the low level functions are PAPER millimetres
(A3 landscape, 420 x 297).  The View class maps model millimetres to paper.
"""
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon as MPoly

W, H = 420.0, 297.0
INK = "#111111"
LW_OUT = 1.3
LW_THIN = 0.5
LW_FRAME = 1.8
FS = 7.0
FS_SMALL = 5.5
ARROW_L = 3.0
ARROW_HW = 0.9
GAP = 1.0
OVER = 2.0
CHAIN = (0, (26, 5, 3, 5))
HIDDEN = (0, (9, 5))
TXT_W = 1.5          # approx. mm per character at FS

MAIN_TITLE = "6-Axis Compact Robot Arm (LR Mate 200iD/4SC style) - approximate"
DATE = "2026-11-11"
NOTE = "Dimensions approximate - verify before manufacture"


def fmt(v):
    s = "%.2f" % v
    s = s.rstrip("0").rstrip(".")
    return s


def new_sheet():
    fig = plt.figure(figsize=(W / 25.4, H / 25.4))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    return fig, ax


# ---------------------------------------------------------------- primitives
def line(ax, p, q, lw=LW_THIN, ls="-", z=5, color=INK):
    ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=lw, ls=ls, zorder=z,
            solid_capstyle="butt", dash_capstyle="butt")


def polyline(ax, pts, closed=False, lw=LW_OUT, ls="-", z=5, fill=None):
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    if fill is not None:
        ax.add_patch(MPoly(pts, closed=True, fc=fill, ec="none", zorder=z))
    if closed:
        xs.append(xs[0])
        ys.append(ys[0])
    ax.plot(xs, ys, color=INK, lw=lw, ls=ls, zorder=z + 0.001,
            solid_joinstyle="round", dash_capstyle="butt")


def rect(ax, x0, y0, x1, y1, lw=LW_THIN, z=5):
    polyline(ax, [(x0, y0), (x1, y0), (x1, y1), (x0, y1)], closed=True, lw=lw, z=z)


def text(ax, x, y, s, fs=FS, ha="center", va="center", rot=0, weight="normal",
         family=None, z=9, style="normal"):
    kw = {}
    if family:
        kw["family"] = family
    ax.text(x, y, s, fontsize=fs, ha=ha, va=va, rotation=rot, color=INK,
            fontweight=weight, zorder=z, rotation_mode="anchor",
            fontstyle=style, **kw)


def arrow(ax, tip, d, L=ARROW_L, hw=ARROW_HW, z=6):
    """Filled arrow head with its tip at `tip`, pointing along unit vector d."""
    n = math.hypot(d[0], d[1])
    dx, dy = d[0] / n, d[1] / n
    bx, by = tip[0] - dx * L, tip[1] - dy * L
    px, py = -dy, dx
    ax.add_patch(MPoly([tip, (bx + px * hw, by + py * hw), (bx - px * hw, by - py * hw)],
                       closed=True, fc=INK, ec=INK, lw=0.2, zorder=z))


def circle_poly(cx, cy, r, n=96):
    return [(cx + r * math.cos(2 * math.pi * i / n), cy + r * math.sin(2 * math.pi * i / n))
            for i in range(n)]


def arc_poly(cx, cy, r, a1, a2, n=40):
    return [(cx + r * math.cos(a1 + (a2 - a1) * i / n), cy + r * math.sin(a1 + (a2 - a1) * i / n))
            for i in range(n + 1)]


# ---------------------------------------------------------------- dimensions
def dim_h(ax, x1, x2, yd, txt, y1=None, y2=None):
    """Horizontal dimension; x1,x2 paper X of features, yd paper Y of the dim line,
    y1,y2 paper Y of the features (for extension lines)."""
    if x1 > x2:
        x1, x2, y1, y2 = x2, x1, y2, y1
    for x, yf in ((x1, y1), (x2, y2)):
        if yf is not None:
            s = 1 if yd > yf else -1
            line(ax, (x, yf + s * GAP), (x, yd + s * OVER))
    L = x2 - x1
    tw = len(txt) * TXT_W + 1.0
    if L >= 2 * ARROW_L + 1.0:
        line(ax, (x1, yd), (x2, yd))
        arrow(ax, (x1, yd), (-1, 0))
        arrow(ax, (x2, yd), (1, 0))
        if tw <= L - 1:
            text(ax, (x1 + x2) / 2, yd + 1.6, txt, va="center")
        else:
            text(ax, x2 + 5 + 1, yd + 1.6, txt, ha="left")
    else:
        line(ax, (x1 - 5, yd), (x2 + 5, yd))
        arrow(ax, (x1, yd), (1, 0))
        arrow(ax, (x2, yd), (-1, 0))
        text(ax, x2 + 6, yd + 1.6, txt, ha="left")


def dim_v(ax, y1, y2, xd, txt, x1=None, x2=None):
    """Vertical dimension; y1,y2 paper Y of features, xd paper X of dim line."""
    if y1 > y2:
        y1, y2, x1, x2 = y2, y1, x2, x1
    for y, xf in ((y1, x1), (y2, x2)):
        if xf is not None:
            s = 1 if xd > xf else -1
            line(ax, (xf + s * GAP, y), (xd + s * OVER, y))
    L = y2 - y1
    tw = len(txt) * TXT_W + 1.0
    if L >= 2 * ARROW_L + 1.0:
        line(ax, (xd, y1), (xd, y2))
        arrow(ax, (xd, y1), (0, -1))
        arrow(ax, (xd, y2), (0, 1))
        if tw <= L - 1:
            text(ax, xd - 1.6, (y1 + y2) / 2, txt, rot=90)
        else:
            text(ax, xd - 1.6, y2 + 6, txt, rot=90, ha="left")
    else:
        line(ax, (xd, y1 - 5), (xd, y2 + 5))
        arrow(ax, (xd, y1), (0, 1))
        arrow(ax, (xd, y2), (0, -1))
        text(ax, xd - 1.6, y2 + 6, txt, rot=90, ha="left")


def dim_r(ax, c, r, ang_deg, txt, out=9):
    """Radius leader: arrow touches the arc from outside, text beyond."""
    a = math.radians(ang_deg)
    u = (math.cos(a), math.sin(a))
    P = (c[0] + r * u[0], c[1] + r * u[1])
    Q = (P[0] + out * u[0], P[1] + out * u[1])
    line(ax, Q, P)
    arrow(ax, P, (-u[0], -u[1]))
    ha = "left" if u[0] >= -0.2 else "right"
    text(ax, Q[0] + (0.8 if ha == "left" else -0.8), Q[1] + (0.6 if u[1] >= 0 else -0.6), txt, ha=ha)


def dim_d(ax, c, r, ang_deg, txt, out=8):
    """Diameter dimension across a circle along direction ang_deg."""
    a = math.radians(ang_deg)
    u = (math.cos(a), math.sin(a))
    P1 = (c[0] - r * u[0], c[1] - r * u[1])
    P2 = (c[0] + r * u[0], c[1] + r * u[1])
    E = (P2[0] + out * u[0], P2[1] + out * u[1])
    if 2 * r >= 2 * ARROW_L + 1:
        line(ax, P1, E)
        arrow(ax, P1, (-u[0], -u[1]))
        arrow(ax, P2, (u[0], u[1]))
    else:
        S = (P1[0] - 5 * u[0], P1[1] - 5 * u[1])
        line(ax, S, E)
        arrow(ax, P1, (u[0], u[1]))
        arrow(ax, P2, (-u[0], -u[1]))
    ha = "left" if u[0] >= -0.2 else "right"
    text(ax, E[0] + (0.8 if ha == "left" else -0.8), E[1] + (0.6 if u[1] >= 0 else -0.6), txt, ha=ha)


def leader(ax, tip, pt, txt, ha="left", fs=FS, arrow_head=True):
    line(ax, tip, pt)
    if arrow_head:
        d = (tip[0] - pt[0], tip[1] - pt[1])
        arrow(ax, tip, d)
    text(ax, pt[0] + (0.8 if ha == "left" else -0.8), pt[1], txt, ha=ha, fs=fs)


def cline(ax, p, q, z=4):
    line(ax, p, q, lw=LW_THIN, ls=CHAIN, z=z)


def ccross(ax, c, size=6.0, z=4):
    cline(ax, (c[0] - size, c[1]), (c[0] + size, c[1]), z)
    cline(ax, (c[0], c[1] - size), (c[0], c[1] + size), z)


# ---------------------------------------------------------------- tables
def table(ax, x0, ytop, colw, rowh, rows, header=True, fs=6.0, align=None):
    """rows: list of lists of str; ytop is the top edge; draws grid + text."""
    ncol = len(colw)
    nrow = len(rows)
    tw = sum(colw)
    for i in range(nrow + 1):
        line(ax, (x0, ytop - i * rowh), (x0 + tw, ytop - i * rowh),
             lw=LW_THIN * (1.6 if (i == 0 or i == nrow or (header and i == 1)) else 1))
    xs = [x0]
    for w in colw:
        xs.append(xs[-1] + w)
    for x in xs:
        line(ax, (x, ytop), (x, ytop - nrow * rowh), lw=LW_THIN)
    for i, r in enumerate(rows):
        for j, cell in enumerate(r):
            al = (align[j] if align else "left")
            tx = xs[j] + 1.2 if al == "left" else (xs[j] + xs[j + 1]) / 2
            text(ax, tx, ytop - (i + 0.5) * rowh, cell, fs=fs, ha=("left" if al == "left" else "center"),
                 weight=("bold" if (header and i == 0) else "normal"))


# ---------------------------------------------------------------- frame / title block
def projection_symbol(ax, x, y):
    """ISO first-angle symbol: truncated cone at left, its end view (circles) at right."""
    polyline(ax, [(x, y - 1.6), (x + 10, y - 3.2), (x + 10, y + 3.2), (x, y + 1.6)], closed=True, lw=0.5)
    cline(ax, (x - 2, y), (x + 12, y))
    xc = x + 19
    polyline(ax, circle_poly(xc, y, 3.2), closed=True, lw=0.5)
    polyline(ax, circle_poly(xc, y, 1.6), closed=True, lw=0.5)
    cline(ax, (xc - 5, y), (xc + 5, y))
    cline(ax, (xc, y - 5), (xc, y + 5))


def frame_and_title(ax, sheet_no, total, title, scale_txt):
    rect(ax, 20, 10, 410, 287, lw=LW_FRAME, z=3)
    # centering ticks
    for x in (215,):
        line(ax, (x, 287), (x, 291), lw=0.6)
        line(ax, (x, 10), (x, 6), lw=0.6)
    x0, x1, y0, y1 = 230, 410, 10, 50
    rect(ax, x0, y0, x1, y1, lw=1.3)
    for yy in (20, 32, 40):
        line(ax, (x0, yy), (x1, yy), lw=LW_THIN)
    text(ax, x0 + 3, 45, MAIN_TITLE, fs=8.5, ha="left", weight="bold")
    text(ax, x0 + 3, 36, title, fs=8, ha="left")
    cells = [(230, 268, "SCALE"), (268, 296, "UNITS"), (296, 340, "PROJECTION"),
             (340, 372, "SHEET"), (372, 410, "DATE")]
    for xa, xb, lab in cells:
        line(ax, (xa, 20), (xa, 32), lw=LW_THIN)
        text(ax, xa + 1.5, 30.2, lab, fs=4.5, ha="left")
    vals = {"SCALE": scale_txt, "UNITS": "mm", "SHEET": "%d/%d" % (sheet_no, total), "DATE": DATE}
    for xa, xb, lab in cells:
        if lab in vals:
            text(ax, (xa + xb) / 2, 23.6, vals[lab], fs=8)
    projection_symbol(ax, 303, 25.0)
    text(ax, 318 + 22, 23.4, "1st angle", fs=4.5, ha="left")
    text(ax, x0 + 3, 15, NOTE, fs=7, ha="left", style="italic")


def label(ax, x, y, s, fs=8):
    text(ax, x, y, s, fs=fs, ha="left", weight="bold")


# ---------------------------------------------------------------- View
class View:
    """Maps model mm (h, v) to paper mm with scale s; kind in front/top/side."""

    def __init__(self, ax, kind, ox, oy, s):
        self.ax, self.kind, self.ox, self.oy, self.s = ax, kind, ox, oy, s

    def x(self, h):
        return self.ox + h * self.s

    def y(self, v):
        return self.oy + v * self.s

    def P(self, h, v):
        return (self.x(h), self.y(v))

    # dimension wrappers: feature positions in model units, dim line in paper units
    def dh(self, h1, h2, ydim, txt=None, v1=None, v2=None):
        txt = txt if txt is not None else fmt(abs(h2 - h1))
        dim_h(self.ax, self.x(h1), self.x(h2), ydim, txt,
              None if v1 is None else self.y(v1), None if v2 is None else self.y(v2))

    def dv(self, v1, v2, xdim, txt=None, h1=None, h2=None):
        txt = txt if txt is not None else fmt(abs(v2 - v1))
        dim_v(self.ax, self.y(v1), self.y(v2), xdim, txt,
              None if h1 is None else self.x(h1), None if h2 is None else self.x(h2))

    def cc(self, h, v, size=6.0):
        ccross(self.ax, self.P(h, v), size)

    def cl(self, h1, v1, h2, v2):
        cline(self.ax, self.P(h1, v1), self.P(h2, v2))

    def poly(self, pts, closed=True, style="out", fill=None, z=5):
        pp = [self.P(*p) for p in pts]
        if style == "out":
            polyline(self.ax, pp, closed=closed, lw=LW_OUT, z=z, fill=fill)
        elif style == "thin":
            polyline(self.ax, pp, closed=closed, lw=LW_THIN, z=z)
        elif style == "hid":
            polyline(self.ax, pp, closed=closed, lw=LW_THIN, ls=HIDDEN, z=z)
        elif style == "cl":
            polyline(self.ax, pp, closed=closed, lw=LW_THIN, ls=CHAIN, z=z)

    def circ(self, h, v, r, style="out", z=5):
        self.poly(circle_poly(h, v, r), True, style, z=z)

    def label_pt(self, h, v, s, fs=FS, ha="center"):
        text(self.ax, self.x(h), self.y(v), s, fs=fs, ha=ha)


def scale_text(s):
    if s >= 1:
        return "%s:1" % fmt(s)
    return "1:%s" % fmt(1.0 / s)


def save_pages(figs, pdf_path, png_paths, dpi=150):
    from matplotlib.backends.backend_pdf import PdfPages
    with PdfPages(pdf_path) as pdf:
        for fig, png in zip(figs, png_paths):
            fig.savefig(png, dpi=dpi)
            pdf.savefig(fig)
            plt.close(fig)
