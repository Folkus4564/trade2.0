"""Generate the 8-sheet drawing set for the compact 6-axis robot (approximate).

Run:  python generate_drawings.py
Output (next to this script):
    robot_drawings.pdf            8 pages, A3 landscape
    sheet_01_GA.png ... sheet_08_P07_Flange.png   (150 dpi)

Geometry (all mm) mirrors macros/build_robot.CATScript and parameters.csv.
Only matplotlib + numpy are used.
"""
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from drawlib import *  # noqa: E402,F401,F403
import drawlib as dl  # noqa: E402

PI = math.pi
OUT_DIR = os.path.dirname(os.path.abspath(__file__))
TOTAL = 8

# ============================================================== geometry model
AX = {"XY": ("x", "y", "z"), "XZ": ("x", "z", "y"), "YZ": ("y", "z", "x")}
VAX = {"front": ("x", "z", "y"), "top": ("x", "y", "z"), "side": ("y", "z", "x")}
IDX = {"x": 0, "y": 1, "z": 2}


def rrect(cx, cy, w, h, r):
    pts = []
    for (ax_, ay_, a1) in ((cx + w / 2 - r, cy - h / 2 + r, -PI / 2),
                           (cx + w / 2 - r, cy + h / 2 - r, 0.0),
                           (cx - w / 2 + r, cy + h / 2 - r, PI / 2),
                           (cx - w / 2 + r, cy - h / 2 + r, PI)):
        pts += arc_poly(ax_, ay_, r, a1, a1 + PI / 2, 10)
    return pts


def rect_pts(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def stadium(cx, y1, y2, r):
    return arc_poly(cx, y2, r, 0, PI, 30) + arc_poly(cx, y1, r, PI, 2 * PI, 30)


def prism(plane, poly, lo, hi, holes=None, name=""):
    return dict(kind="prism", plane=plane, poly=[tuple(p) for p in poly], lo=lo, hi=hi,
                holes=holes or [], name=name)


def hole(polys, lo, hi):
    return dict(polys=polys, lo=lo, hi=hi)


def revx(pts, name=""):
    return dict(kind="revx", pts=pts, T=(0, 0, 0), name=name)


def shift(s, T):
    if s["kind"] == "revx":
        return dict(s, T=tuple(a + b for a, b in zip(s["T"], T)))
    ua, va, na = AX[s["plane"]]
    du, dv, dn = T[IDX[ua]], T[IDX[va]], T[IDX[na]]
    holes = [dict(polys=[[(u + du, v + dv) for u, v in p] for p in h["polys"]],
                  lo=h["lo"] + dn, hi=h["hi"] + dn) for h in s["holes"]]
    return dict(s, poly=[(u + du, v + dv) for u, v in s["poly"]], lo=s["lo"] + dn,
                hi=s["hi"] + dn, holes=holes)


def extent(s, axis):
    if s["kind"] == "revx":
        T = dict(zip("xyz", s["T"]))
        if axis == "x":
            xs = [p[0] for p in s["pts"]]
            return (min(xs) + T["x"], max(xs) + T["x"])
        r = max(p[1] for p in s["pts"])
        return (T[axis] - r, T[axis] + r)
    ua, va, na = AX[s["plane"]]
    if axis == na:
        return (s["lo"], s["hi"])
    i = 0 if axis == ua else 1
    vals = [p[i] for p in s["poly"]]
    return (min(vals), max(vals))


# ---- local-frame solids per part (see spec) ---------------------------------
bolt_holes = [circle_poly(15.75 * math.cos(math.radians(a)), 15.75 * math.sin(math.radians(a)), 2.75, 32)
              for a in (45, 135, 225, 315)]

PART_SOLIDS = {
    "P01": [
        prism("XY", rrect(0, 0, 190, 190, 12), 0, 15, name="Plate",
              holes=[hole([circle_poly(sx * 80, sy * 80, 5.5, 32) for sx in (-1, 1) for sy in (-1, 1)], 0, 15)]),
        prism("XY", rrect(0, 0, 150, 150, 20), 15, 165, name="Housing"),
        prism("XY", rect_pts(-105, -45, -75, 45), 15, 115, name="Connector"),
    ],
    "P02": [
        prism("XY", circle_poly(0, 0, 70), 0, 30, name="Disk"),
        prism("XZ", [(-60, 30), (60, 30), (60, 165)] + arc_poly(0, 165, 60, 0, PI, 40) + [(-60, 165)],
              -55, 55, name="Shoulder"),
        prism("XZ", circle_poly(0, 165, 50), -95, -55, name="J2 cover"),
    ],
    "P03": [
        prism("XZ", stadium(0, 0, 260, 50), -30, 30, name="Arm"),
        prism("XZ", circle_poly(0, 0, 35), 30, 35, name="Cap J2"),
        prism("XZ", circle_poly(0, 260, 35), 30, 35, name="Cap J3"),
    ],
    "P04": [
        prism("XZ", [(55, 0), (55, 65), (-55, 65), (-55, 0)] + arc_poly(0, 0, 55, PI, 2 * PI, 40),
              -55, 55, name="Elbow"),
    ],
    "P05": [
        revx([(0, 0), (0, 45), (200, 35), (200, 0)], name="Shaft"),
        prism("XZ", [(200, -35), (235, -35)] + arc_poly(235, 0, 35, -PI / 2, PI / 2, 30) + [(200, 35)],
              -40, 40, name="Yoke"),
    ],
    "P06": [
        prism("XZ", [(0, -30), (40, -30), (40, 30), (0, 30)] + arc_poly(0, 0, 30, PI / 2, 3 * PI / 2, 30),
              -24, 24, name="Wrist"),
        prism("YZ", circle_poly(0, 0, 25), 40, 55, name="Nose"),
    ],
    "P07": [
        prism("YZ", circle_poly(0, 0, 28), 0, 12, name="Flange"),
        prism("YZ", circle_poly(0, 0, 20), 12, 15, name="Tool disk",
              holes=[hole(bolt_holes, 7, 15), hole([circle_poly(0, 0, 10, 48)], 9, 15)]),
    ],
}

HOME = {"P01": (0, 0, 0), "P02": (0, 0, 165), "P03": (0, 85, 330), "P04": (0, 0, 590),
        "P05": (55, 0, 610), "P06": (290, 0, 610), "P07": (345, 0, 610)}
NAMES = {"P01": "Base", "P02": "Turret", "P03": "LowerArm", "P04": "ElbowHousing",
         "P05": "Forearm", "P06": "Wrist", "P07": "Flange"}
DESCR = {"P01": "Base pillar", "P02": "Turret / shoulder", "P03": "Lower arm", "P04": "Elbow housing",
         "P05": "Forearm", "P06": "Wrist", "P07": "Tool flange (ISO)"}


def project(s, vk, hidden=True):
    """Return [(kind, data, style)] for solid s in view vk."""
    ha, va, _ = VAX[vk]
    out = []
    if s["kind"] == "revx":
        T = dict(zip("xyz", s["T"]))
        pts = s["pts"]
        if vk in ("front", "top"):
            off = T["z"] if vk == "front" else T["y"]
            up = [(x + T["x"], r + off) for x, r in pts if r > 1e-9]
            lo = [(x + T["x"], -r + off) for x, r in reversed(pts) if r > 1e-9]
            out.append(("poly", up + lo, "out"))
        else:
            for r in sorted({p[1] for p in pts if p[1] > 1e-9}, reverse=True):
                out.append(("poly", circle_poly(T["y"], T["z"], r), "out"))
        return out
    ua, va_, na = AX[s["plane"]]
    if (ha, va) == (ua, va_):
        out.append(("poly", s["poly"], "out"))
        for h in s["holes"]:
            for p in h["polys"]:
                out.append(("poly", p, "out"))
    else:
        def rng(a, poly=None, lo=None, hi=None):
            if a == na:
                return (s["lo"] if lo is None else lo, s["hi"] if hi is None else hi)
            src = s["poly"] if poly is None else poly
            i = 0 if a == ua else 1
            vals = [p[i] for p in src]
            return (min(vals), max(vals))
        h0, h1 = rng(ha)
        v0, v1 = rng(va)
        out.append(("poly", rect_pts(h0, v0, h1, v1), "out"))
        if hidden:
            for h in s["holes"]:
                for p in h["polys"]:
                    a0, a1 = rng(ha, p, h["lo"], h["hi"])
                    b0, b1 = rng(va, p, h["lo"], h["hi"])
                    out.append(("poly", rect_pts(a0, b0, a1, b1), "hid"))
    return out


def draw_solids(V, solids, hidden=True):
    """Painter's algorithm: far solids first, white fill gives simple hidden-line removal."""
    na = VAX[V.kind][2]
    far_first = -1 if V.kind == "front" else 1
    ordered = sorted(solids, key=lambda s: far_first * sum(extent(s, na)) / 2.0)
    z = 2.0
    for s in ordered:
        for kind, pts, style in project(s, V.kind, hidden):
            if style == "hid":
                V.poly(pts, True, "hid", z=z + 0.005)
            else:
                V.poly(pts, True, "out", fill="white", z=z)
        z += 0.01


def placed(pids):
    out = []
    for pid in pids:
        out += [shift(s, HOME[pid]) for s in PART_SOLIDS[pid]]
    return out


def feat_block(ax, x, ytop, lines, fs=6.0):
    """Monospace feature recipe block."""
    text(ax, x, ytop, "CATIA FEATURE RECIPE (Part Design)", fs=7, ha="left", weight="bold")
    for i, ln in enumerate(lines):
        text(ax, x, ytop - 5.0 - i * 3.9, ln, fs=fs, ha="left", family="monospace")


# ============================================================== SHEET 1 : GA
def schematic(ax, ox, oy, s):
    def P(x, z):
        return (ox + x * s, oy + z * s)
    # ground hatch
    line(ax, P(-40, 0), P(40, 0), lw=1.2)
    for i in range(-4, 5):
        line(ax, (P(i * 10, 0)[0], P(0, 0)[1]), (P(i * 10, 0)[0] - 2.0, P(0, 0)[1] - 2.5), lw=0.4)
    # links (home pose)
    links = [(0, 0), (0, 330), (0, 590), (0, 610), (290, 610), (345, 610), (360, 610)]
    polyline(ax, [P(*p) for p in links], closed=False, lw=2.6, z=4)
    # axes
    cline(ax, P(0, -25), P(0, 345))                    # J1 (Z)
    cline(ax, P(-25, 610), P(395, 610))                # J4 / J6 (X)
    for (x, z, lab, dx, dy) in ((0, 330, "J2", -9, 0), (0, 590, "J3", -9, -1.5), (290, 610, "J5", 0, 7)):
        c = P(x, z)
        polyline(ax, circle_poly(c[0], c[1], 2.4), True, lw=0.9, z=7, fill="white")
        polyline(ax, circle_poly(c[0], c[1], 0.6), True, lw=0.9, z=8, fill=dl.INK)
        text(ax, c[0] + dx, c[1] + dy, lab, fs=7, weight="bold")
    text(ax, P(0, 0)[0] + 9, P(0, 150)[1], "J1 (Z)", fs=7, weight="bold")
    text(ax, P(100, 610)[0], P(100, 610)[1] + 5, "J4 (X)", fs=7, weight="bold")
    text(ax, P(345, 610)[0], P(345, 610)[1] - 6, "J6 (X)", fs=7, weight="bold")
    c = P(360, 610)
    polyline(ax, circle_poly(c[0], c[1], 1.6), True, lw=0.8, z=7, fill="white")
    text(ax, c[0] + 6, c[1] + 0.5, "TCP", fs=6.5, ha="left")
    text(ax, ox - 12, oy + 330 * s, "330", fs=6, ha="right")
    text(ax, ox - 12, oy + 460 * s, "260", fs=6, ha="right")
    text(ax, ox + 3, oy + 600 * s, "", fs=6)
    text(ax, ox + 145 * s, oy + 610 * s - 8, "290 (J3..J5)", fs=6, ha="center")


def sheet_GA():
    fig, ax = new_sheet()
    s = 0.2
    F = View(ax, "front", 150, 125, s)
    T = View(ax, "top", 150, 72, s)
    R = View(ax, "side", 62, 125, s)
    sol = placed(["P%02d" % i for i in range(1, 8)])
    for V in (F, T, R):
        draw_solids(V, sol, hidden=False)
    # view labels
    label(ax, 131, 281, "FRONT VIEW (XZ)")
    label(ax, 131, 99.5, "TOP VIEW (XY)", fs=7)
    label(ax, 36, 281, "RIGHT SIDE VIEW (YZ)")
    text(ax, 36, 276, "(1st angle: placed left of front view)", fs=5.5, ha="left")

    # joint axes
    F.cl(0, -15, 0, 672)
    for (x, z) in ((0, 330), (0, 590), (290, 610)):
        F.cc(x, z, 8)
    F.cl(-30, 610, 395, 610)
    T.cc(0, 0, 8)
    T.cl(0, -105, 0, 125)         # J2/J3 axis (along Y) at x=0
    T.cl(290, -55, 290, 55)       # J5
    T.cl(-30, 0, 395, 0)          # J4/J6
    R.cl(0, -15, 0, 672)          # J1
    R.cl(-100, 330, 125, 330)     # J2
    R.cl(-100, 590, 125, 590)     # J3
    R.cc(0, 610, 8)               # J4/J6 end-on
    R.cl(-60, 610, 60, 610)       # J5 (Y) at z=610

    # J labels (front)
    for (x, z, t, dx, dy) in ((0, 330, "J2", 7, 5), (0, 590, "J3", 7, -6), (290, 610, "J5", 0, 14),
                              (345, 610, "J6", 0, -14), (0, 668, "J1", 6, 0), (140, 610, "J4", 0, -12)):
        text(ax, F.x(x) + dx, F.y(z) + dy, t, fs=7, weight="bold")
    text(ax, F.x(360) + 4, F.y(610) - 1, "TCP", fs=6.5, ha="left")
    text(ax, F.x(360) + 4, F.y(610) - 5, "(360, 0, 610)", fs=5.5, ha="left")

    # ---- front view dimensions
    xb = F.x(-105)
    F.dv(0, 330, xb - 6, "330 (J2)", -95, -60)
    F.dv(0, 590, xb - 14, "590 (J3)", -95, -55)
    F.dv(0, 610, xb - 22, "610 (J4)", -95, -55)
    F.dv(0, 655, xb - 30, "655 overall", -95, -55)
    ytop = F.y(655)
    F.dh(0, 290, ytop + 8, "290 (J5)", 655, 640)
    F.dh(0, 360, ytop + 16, "360 (TCP)", 655, 630)
    F.dh(-95, 95, F.y(0) - 8, "190 (base)", 0, 0)
    F.dh(-105, 360, F.y(0) - 16, "465 overall", 15, 582)
    # ---- top view dims
    xr = F.x(360) + 10
    T.dv(-95, 95, xr + 4, "190", 95, 95)
    T.dv(-95, 115, xr + 14, "210 overall", 50, 50)
    # ---- side view dims
    R.dh(-95, 95, R.y(0) - 8, "190", 0, 0)
    R.dh(-95, 115, R.y(0) - 16, "210 overall", 0, 0)
    label(ax, 36, 111.5, "", fs=6)

    # notes
    text(ax, 22, 100, "NOTES", fs=7, ha="left", weight="bold")
    for i, t in enumerate(("1. Home pose: all joints at 0 deg.",
                           "2. World: Z up = J1 axis, +X = arm reach,",
                           "    J2/J3/J5 parallel to Y, J4/J6 along X.",
                           "3. Hub caps / fillets omitted (cosmetic).",
                           "4. Brand marks intentionally not shown.")):
        text(ax, 22, 95 - i * 4.2, t, fs=6, ha="left")

    # parts list
    rows = [["Item", "Part (CATPart)", "Home pos X,Y,Z (mm)"]]
    for i in range(1, 8):
        pid = "P%02d" % i
        rows.append([pid, "%s_%s - %s" % (pid, NAMES[pid], DESCR[pid]), "%d, %d, %d" % HOME[pid]])
    text(ax, 22, 71.5, "PARTS LIST", fs=7, ha="left", weight="bold")
    table(ax, 22, 69, [10, 58, 34], 5.2, rows, fs=5.6)

    # schematic
    label(ax, 262, 280, "KINEMATIC SCHEMATIC (home pose, XZ plane)", fs=7.5)
    schematic(ax, 280, 140, 0.2)
    text(ax, 318, 215, "Modified DH (approx, mm)", fs=6.5, ha="left", weight="bold")
    for i, t in enumerate(("d1 = 330  (J1->J2)", "a2 = 260  (J2->J3)", "a3 =  20  (J3 offset)",
                           "d4 = 290  (J3->J5)", "d6 =  70  (J5->flange face)", "reach ~ 550 (class)")):
        text(ax, 318, 210 - i * 4.4, t, fs=6, ha="left", family="monospace")

    # joint table
    label(ax, 262, 126, "JOINT TABLE", fs=7.5)
    rows = [["Axis", "Between", "Axis dir.", "Axis passes (x, z)", "Limits [deg]"],
            ["J1", "Base - Turret", "Z", "(0, 0)", "-170 / +170"],
            ["J2", "Turret - Lower arm", "Y", "(0, 330)", "-110 / +120"],
            ["J3", "Lower arm - Elbow", "Y", "(0, 590)", "-70 / +200"],
            ["J4", "Elbow - Forearm", "X", "(-, 610)", "-190 / +190"],
            ["J5", "Forearm - Wrist", "Y", "(290, 610)", "-125 / +125"],
            ["J6", "Wrist - Flange", "X", "(-, 610)", "-360 / +360"]]
    table(ax, 262, 123, [10, 40, 14, 36, 38], 6.0, rows, fs=6.2, align=["c", "left", "c", "c", "c"])
    text(ax, 262, 78, "Limits approximate (verify against official datasheet).", fs=5.8, ha="left",
         style="italic")
    frame_and_title(ax, 1, TOTAL, "SHEET 1 - GENERAL ARRANGEMENT (home pose)", "1:5")
    return fig


# ============================================================== part sheets
def new_part_sheet(n, pid, subtitle, s):
    fig, ax = new_sheet()
    frame_and_title(ax, n, TOTAL, "%s_%s - %s" % (pid, NAMES[pid], subtitle), scale_text(s))
    return fig, ax


def sheet_P01():
    s = 0.5
    fig, ax = new_part_sheet(2, "P01", "sketch profiles and extrusion", s)
    sol = PART_SOLIDS["P01"]
    A = View(ax, "top", 100, 190, s)
    draw_solids(A, sol)
    label(ax, 30, 273, "VIEW A - TOP (sketch plane XY: Sketch.1, .2 at z=0; Sketch.3, .4 at z=15)")
    A.cc(0, 0, 10)
    A.cl(-100, 0, 100, 0)
    A.cl(0, -100, 0, 100)
    for sx in (-1, 1):
        for sy in (-1, 1):
            A.cc(sx * 80, sy * 80, 5)
    yb = A.y(-95)
    A.dh(-95, 95, yb - 8, "190", -95, -95)
    A.dh(-80, 80, yb - 16, "160", -80, -80)
    A.dh(-75, 75, yb - 24, "150", -75, -75)
    xr = A.x(95)
    A.dv(-95, 95, xr + 8, "190", 95, 95)
    A.dv(-80, 80, xr + 16, "160", 80, 80)
    A.dv(-75, 75, xr + 24, "150", 75, 75)
    A.dv(-45, 45, A.x(-105) - 10, "90", -105, -105)
    dim_r(ax, A.P(83, 83), 12 * s, 45, "R12", out=8)
    dim_r(ax, A.P(55, 55), 20 * s, 70, "R20", out=26)
    hc = A.P(-80, 80)
    leader(ax, (hc[0] - 2.75 * s * 0.707, hc[1] + 2.75 * s * 0.707), (hc[0] - 14, hc[1] + 14), "4x Ø11", ha="right")
    text(ax, A.x(-105) - 1, A.y(45) + 3, "Pad.3", fs=5.5, ha="right")

    B = View(ax, "front", 318, 125, s)
    draw_solids(B, sol)
    label(ax, 250, 273, "VIEW B - FRONT (XZ): extrusion thickness")
    # sketch planes
    B.cl(-110, 15, 100, 15)
    B.cl(-110, 0, 100, 0)
    text(ax, B.x(0), B.y(0) - 30, "chain lines = sketch planes z=0 and z=15", fs=6)
    xr = B.x(95)
    B.dv(0, 15, xr + 8, "15", 95, 95)
    B.dv(15, 165, xr + 8, "150", 75, 75)
    B.dv(0, 165, xr + 18, "165", 95, 75)
    B.dv(15, 115, B.x(-105) - 10, "100", -105, -105)
    B.dh(-105, -75, B.y(115) + 8, "30", 115, 115)
    B.dh(-95, 95, B.y(0) - 8, "190", 0, 0)
    B.dh(-80, 80, B.y(0) - 16, "160 (holes)", 0, 0)
    text(ax, B.x(0), B.y(100), "Pad.2  L=150 (+Z)", fs=6)
    text(ax, B.x(0), B.y(7.5), "Pad.1  L=15", fs=5.2)
    leader(ax, (B.x(-80) - 2, B.y(7.5)), (B.x(-125), B.y(7.5)), "Pocket.1 4x D11, mirrored half 50", ha="right")
    text(ax, B.x(-105) - 14, B.y(65), "Pad.3 L=100 (+Z)", fs=6, ha="right")
    feat_block(ax, 24, 75, [
        "1 Sketch.1  plane z=0 (H=+X,V=+Y): rounded rect 190x190 R12      -> Pad.1  L=15 (+Z)    z 0..15",
        "2 Sketch.2  plane z=0 (H=+X,V=+Y): 4x circle D11 at (+-80,+-80)  -> Pocket.1 Mirrored extent half 50",
        "3 Sketch.3  plane z=15 (H=+X,V=+Y): rounded rect 150x150 R20     -> Pad.2  L=150 (+Z)   z 15..165",
        "4 Sketch.4  plane z=15 (H=+X,V=+Y): rect X-105..-75, Y-45..45   -> Pad.3  L=100 (+Z)   z 15..115",
        "Local frame = world frame, origin at floor centre. Part is FIXED in the assembly.",
    ])
    return fig


def sheet_P02():
    s = 2.0 / 3.0
    fig, ax = new_part_sheet(3, "P02", "sketch profiles and extrusion", s)
    sol = PART_SOLIDS["P02"]
    A = View(ax, "front", 85, 92, s)
    draw_solids(A, sol)
    label(ax, 30, 273, "VIEW A - FRONT (XZ): Sketch.2 shoulder, Sketch.3 cover")
    A.cc(0, 165, 10)
    A.cl(-75, 165, 75, 165)
    text(ax, A.x(0) + 2, A.y(165) - 4, "J2", fs=6.5, ha="left", weight="bold")
    A.cl(0, -10, 0, 232)
    A.dh(-70, 70, A.y(0) - 8, "140", 0, 0)
    A.dh(-60, 60, A.y(225) + 8, "120", 165, 165)
    xl = A.x(-70) - 8
    A.dv(0, 30, xl, "30", -70, -70)
    xr = A.x(60) + 8
    A.dv(30, 165, xr, "135", 60, 60)
    A.dv(0, 165, xr + 8, "165 (J2)", 60, 0)
    A.dv(0, 225, xr + 16, "225", 0, 0)
    dim_r(ax, A.P(0, 165), 60 * s, 40, "R60", out=14)
    dim_d(ax, A.P(0, 165), 50 * s, 160, "Ø100 (cover)", out=12)
    text(ax, A.x(0), A.y(95), "Pad.2 (mirrored half 55)", fs=5.8)
    text(ax, A.x(0), A.y(15), "Pad.1 L=30", fs=5.5)

    B = View(ax, "side", 232, 92, s)
    draw_solids(B, sol)
    label(ax, 170, 273, "VIEW B - SIDE (YZ)")
    B.cl(0, -10, 0, 232)
    B.cl(-100, 165, 75, 165)
    B.dh(-70, 70, B.y(0) - 8, "140", 0, 0)
    B.dh(-55, 55, B.y(225) + 8, "110 (2x55)", 225, 225)
    B.dv(0, 30, B.x(70) + 8, "30", 70, 70)
    B.dh(-95, -55, B.y(0) - 16, "40", 0, 0)
    ax_t = B.x(-75)
    text(ax, B.x(-75), B.y(215) + 12, "Pad.3 cover", fs=5.8)
    text(ax, B.x(-75), B.y(215) + 8, "mirrored half 20", fs=5.8)
    text(ax, B.x(-75), B.y(215) + 4, "on plane y=-75", fs=5.8)

    C = View(ax, "top", 340, 185, s)
    draw_solids(C, sol)
    label(ax, 292, 273, "VIEW C - TOP (XY)")
    C.cc(0, 0, 8)
    C.cl(-75, 0, 75, 0)
    C.cl(0, -100, 0, 75)
    dim_d(ax, C.P(0, 0), 70 * s, 215, "Ø140", out=6)
    C.dh(-60, 60, C.y(55) + 10, "120", 55, 55)
    C.dv(-55, 55, C.x(60) + 10, "110", 60, 60)
    C.dh(-50, 50, C.y(-95) - 8, "100", -95, -95)
    C.dv(-95, -55, C.x(-50) - 8, "40", -50, -50)
    C.dv(-95, 0, C.x(-72) - 8, "95", -50, 0)
    feat_block(ax, 24, 56, [
        "1 Sketch.1  plane z=0 (H=+X,V=+Y): circle D140 at (0,0)                  -> Pad.1  L=30 (+Z)   z 0..30",
        "2 Sketch.2  plane y=0 (H=+X,V=+Z): (-60,30)-(60,30)-(60,165), arc R60 c(0,165) over top to (-60,165), close",
        "                                                                          -> Pad.2  Mirrored extent half 55 (Y +-55)",
        "3 Sketch.3  plane y=-75 (H=+X,V=+Z): circle D100 at (0,165)              -> Pad.3  Mirrored extent half 20 (Y -95..-55)",
        "Local origin = (0,0,165) world (top of base).  J2 axis (parallel Y) at local z=165 -> world z=330.",
    ])
    return fig


def sheet_P03():
    s = 0.5
    fig, ax = new_part_sheet(4, "P03", "sketch profiles and extrusion", s)
    sol = PART_SOLIDS["P03"]
    A = View(ax, "front", 110, 98, s)
    draw_solids(A, sol)
    label(ax, 30, 273, "VIEW A - FRONT (XZ): Sketch.1 stadium")
    for z in (0, 260):
        A.cc(0, z, 8)
        A.circ(0, z, 35, "hid", z=9)
    A.cl(0, -60, 0, 320)
    text(ax, A.x(0) + 3, A.y(0) - 4, "J2", fs=6.5, ha="left", weight="bold")
    text(ax, A.x(0) + 3, A.y(260) + 4, "J3", fs=6.5, ha="left", weight="bold")
    A.dh(-50, 50, A.y(-50) - 8, "100", -50, -50)
    xl = A.x(-50) - 8
    A.dv(0, 260, xl, "260 (J2-J3)", 0, 0)
    A.dv(-50, 310, xl - 10, "360", 0, 0)
    dim_r(ax, A.P(0, 260), 50 * s, 45, "R50", out=14)
    dim_r(ax, A.P(0, 0), 50 * s, -45, "R50", out=14)
    dim_d(ax, A.P(0, 0), 35 * s, 20, "Ø70 cap (dashed)", out=40)
    dim_d(ax, A.P(0, 260), 35 * s, 25, "Ø70 cap", out=34)
    text(ax, A.x(0), A.y(130), "Pad.1 mirrored half 30", fs=6)

    B = View(ax, "side", 240, 98, s)
    draw_solids(B, sol)
    label(ax, 205, 273, "VIEW B - SIDE (YZ): thickness")
    B.cl(-45, 0, 50, 0)
    B.cl(-45, 260, 50, 260)
    B.cl(0, -60, 0, 320)
    B.dh(-30, 30, B.y(310) + 8, "60 (2x30)", 310, 310)
    B.dh(30, 35, B.y(310) + 18, "5", 35, 35)
    B.dv(-35, 35, B.x(35) + 14, "Ø70", 35, 35)
    leader(ax, (B.x(33), B.y(260)), (B.x(33) + 22, B.y(260) + 22), "Pad.2 cap L=5 (+Y)", ha="left")
    leader(ax, (B.x(33), B.y(0)), (B.x(33) + 22, B.y(0) - 22), "Pad.3 cap L=5 (+Y)", ha="left")
    text(ax, B.x(0), B.y(130), "Pad.1", fs=6)

    C = View(ax, "front", 345, 98, s)
    label(ax, 300, 273, "VIEW C - hub cap sketch axes", fs=8)
    text(ax, 318, 262, "Sketch.2 on plane y=30:", fs=6.5, ha="left")
    text(ax, 318, 257.5, "origin (0,30,0)  H=(0,0,1)  V=(1,0,0)", fs=6.5, ha="left")
    text(ax, 318, 253, "normal = H x V = Z x X = +Y  (pad goes outward)", fs=6.5, ha="left")
    text(ax, 318, 248, "circles D70 at (h,v)=(0,0) and (260,0)", fs=6.5, ha="left")
    feat_block(ax, 24, 66, [
        "1 Sketch.1  plane y=0 (H=+X,V=+Z): stadium - R50 at (0,0) and (0,260), lines x=+-50  -> Pad.1  Mirrored half 30",
        "2 Sketch.2  plane y=30 (H=+Z,V=+X; normal +Y): 2x circle D70 at (0,0),(260,0)         -> Pad.2  L=5 (+Y)",
        "Local origin = J2 axis point; world translation in assembly = (0, 85, 330). J3 axis at local z=260.",
        "Lower arm occupies world Y 55..115 (touches turret shoulder face y=55); caps are cosmetic.",
    ])
    return fig


def sheet_P04():
    s = 1.0
    fig, ax = new_part_sheet(5, "P04", "sketch profile and extrusion", s)
    sol = PART_SOLIDS["P04"]
    A = View(ax, "front", 100, 160, s)
    draw_solids(A, sol)
    label(ax, 30, 273, "VIEW A - FRONT (XZ): Sketch.1")
    A.cc(0, 0, 10)
    A.cl(-70, 0, 100, 0)
    A.cc(0, 20, 6)
    A.cl(-70, 20, 100, 20)
    text(ax, A.x(0) - 2, A.y(0) - 7, "J3", fs=6.5, weight="bold")
    text(ax, A.x(-65), A.y(20) + 2.5, "J4 axis (X)", fs=6.5, ha="left")
    A.dh(-55, 55, A.y(-55) - 8, "110", 0, 0)
    xr = A.x(55) + 8
    A.dv(0, 65, xr, "65", 55, 55)
    A.dv(0, 20, xr + 10, "20", 0, 55)
    dim_r(ax, A.P(0, 0), 55, 235, "R55", out=14)
    text(ax, A.x(0), A.y(45), "Pad.1", fs=7)
    text(ax, A.x(0), A.y(37), "Mirrored extent", fs=6)
    text(ax, A.x(0), A.y(30), "half 55", fs=6)
    B = View(ax, "side", 270, 160, s)
    draw_solids(B, sol)
    label(ax, 195, 273, "VIEW B - SIDE (YZ): thickness")
    B.cl(0, -70, 0, 80)
    B.cl(-70, 0, 70, 0)
    B.dh(-55, 55, B.y(65) + 8, "110 (2x55)", 65, 65)
    B.dv(0, 65, B.x(55) + 8, "65", 55, 55)
    B.dh(-55, 0, B.y(-55) - 8, "55", -55, -55)
    text(ax, B.x(0), B.y(32), "Pad.1", fs=7)
    feat_block(ax, 24, 90, [
        "1 Sketch.1  plane y=0 (H=+X,V=+Z), closed profile:",
        "     bottom semicircle R55 centre (0,0) from (-55,0) through (0,-55) to (55,0);",
        "     (55,0)-(55,65)-(-55,65)-(-55,0)                -> Pad.1  Mirrored extent, half 55 (Y +-55)",
        "Local origin = J3 axis (world (0,0,590)).  Front face x=55 carries the J4 axis at local z=20",
        "(world z=610).  Forearm P05 is placed at (55,0,610).",
    ])
    return fig


def sheet_P05():
    s = 0.75
    fig, ax = new_part_sheet(6, "P05", "sketch profiles and extrusion", s)
    sol = PART_SOLIDS["P05"]
    A = View(ax, "front", 45, 215, s)
    draw_solids(A, sol)
    label(ax, 30, 283, "VIEW A - FRONT (XZ): Sketch.1 shaft profile (revolve), Sketch.2 yoke, Sketch.3 slot")
    # pocket sketch outline (chain) + axes
    A.poly(rect_pts(200, -50, 280, 50), True, "cl", z=9)
    text(ax, A.x(240), A.y(-50) - 3, "slot sketch", fs=5.5)
    A.cl(-10, 0, 290, 0)
    text(ax, A.x(-10), A.y(0) + 2.5, "J4 / shaft axis", fs=5.5, ha="left")
    A.cc(235, 0, 8)
    A.cl(235, -55, 235, 55)
    text(ax, A.x(235) + 2, A.y(0) + 3, "J5", fs=6.5, ha="left", weight="bold")
    # dims above
    A.dh(0, 200, A.y(50) + 8, "200", 45, 35)
    A.dh(200, 280, A.y(50) + 8, "80", 50, 50)
    A.dh(0, 235, A.y(50) + 17, "235 (J5)", 45, 0)
    # below
    yb = A.y(-45) - 8
    A.dh(0, 200, yb, "200", -45, -35)
    A.dh(200, 235, yb, "35", -35, -35)
    A.dh(235, 270, yb, "35 (R)", -35, 0)
    # vertical
    A.dv(-45, 45, A.x(0) - 10, "Ø90 (2x45)", 0, 0)
    A.dv(-35, 35, A.x(200) - 8, "Ø70", None, None)
    xr = A.x(280) + 8
    A.dv(-50, 50, xr, "100", 280, 280)
    A.dv(-35, 35, xr + 8, "70", 235, 235)
    dim_r(ax, A.P(235, 0), 35 * s, -45, "R35", out=12)
    text(ax, A.x(100), A.y(-12), "Shaft  (revolve 360 deg about X)", fs=6)
    text(ax, A.x(100), A.y(-18), "sketch = upper half, centre line V=0", fs=6)

    B = View(ax, "top", 45, 118, s)
    draw_solids(B, sol)
    label(ax, 30, 163, "VIEW B - TOP (XY): extrusion thickness of yoke / slot")
    for yy in (-25, 25):
        B.poly([(200, yy), (270, yy)], False, "out", z=9)
    B.cl(-10, 0, 290, 0)
    B.dh(0, 200, B.y(-45) - 8, "200", -45, -35)
    B.dh(200, 270, B.y(-45) - 8, "70", -40, -40)
    xr = B.x(270) + 8
    B.dv(25, 40, xr, "15", 270, 270)
    B.dv(-25, 25, xr, "50", 270, 270)
    B.dv(-40, -25, xr, "15", 270, 270)
    B.dv(-40, 40, xr + 10, "80 (2x40)", 270, 270)
    B.dv(-45, 45, B.x(0) - 10, "Ø90", 0, 0)
    text(ax, B.x(235), B.y(0) + 3.5, "Pocket.1: mirrored half 25", fs=5.8, ha="center")
    text(ax, B.x(235), B.y(52) + 0, "Pad.1 yoke: mirrored half 40", fs=5.8)

    C = View(ax, "side", 335, 200, 1.0)
    draw_solids(C, sol)
    label(ax, 292, 273, "VIEW C - END (YZ)  1:1", fs=7.5)
    C.cc(0, 0, 10)
    for yy in (-25, 25):
        C.poly([(yy, -35), (yy, 35)], False, "out", z=9)
    C.dh(-40, 40, C.y(45) + 18, "80", 35, 35)
    C.dh(-25, 25, C.y(45) + 8, "50", 35, 35)
    C.dv(-45, 45, C.x(-45) - 8, "Ø90", -45, -45)
    text(ax, C.x(0), C.y(-52), "yoke Z +-35 (70)", fs=6)
    feat_block(ax, 24, 58, [
        "1 Sketch.1  plane y=0 (H=+X,V=+Z): (0,0)-(0,45)-(200,35)-(200,0)-(0,0); construction line (0,0)-(200,0) = CenterLine",
        "                                                              -> Shaft.1  360 deg about X  (cone D90 -> D70, x 0..200)",
        "2 Sketch.2  plane y=0: (200,-35)-(235,-35), arc R35 c(235,0) to (235,35), (235,35)-(200,35), close -> Pad.1 mirrored half 40",
        "3 Sketch.3  plane y=0: rect X 200..280, Z -50..50                          -> Pocket.1 mirrored half 25 (prongs 15 thick)",
        "Local origin = J4 axis at elbow front face (world (55,0,610)).  J5 axis at local x=235 (world x=290).",
    ])
    return fig


def sheet_P06():
    s = 1.2
    fig, ax = new_part_sheet(7, "P06", "sketch profiles and extrusion", s)
    sol = PART_SOLIDS["P06"]
    A = View(ax, "front", 85, 232, s)
    draw_solids(A, sol)
    label(ax, 30, 283, "VIEW A - FRONT (XZ): Sketch.1 wrist profile")
    A.cc(0, 0, 8)
    A.cl(-40, 0, 62, 0)
    A.cl(0, -35, 0, 35)
    text(ax, A.x(0) + 2, A.y(0) + 3, "J5 (Y)", fs=6.5, ha="left", weight="bold")
    text(ax, A.x(58), A.y(0) + 2.5, "J6 axis (X)", fs=5.8, ha="left")
    A.dh(0, 40, A.y(-30) - 8, "40", -30, -30)
    A.dh(-30, 40, A.y(-30) - 17, "70", 0, -30)
    xr = A.x(40) + 8
    A.dv(-30, 30, xr, "60 (2x30)", 40, 40)
    dim_r(ax, A.P(0, 0), 30 * s, 160, "R30", out=18)
    text(ax, A.x(10), A.y(0) - 8, "Pad.1  mirrored half 24", fs=5.6)

    B = View(ax, "top", 85, 135, s)
    draw_solids(B, sol)
    label(ax, 30, 172, "VIEW B - TOP (XY): thickness and nose")
    B.cl(-40, 0, 62, 0)
    B.cl(0, -32, 0, 32)
    B.dv(-24, 24, B.x(-30) - 8, "48 (2x24)", -30, -30)
    B.dh(40, 55, B.y(-25) - 8, "15 (Pad.2)", -25, -25)
    B.dh(-30, 40, B.y(-25) - 17, "70", -24, -24)
    B.dv(-25, 25, B.x(55) + 8, "Ø50", 55, 55)
    text(ax, B.x(5), B.y(0) + 6, "Pad.1", fs=6)

    C = View(ax, "side", 320, 225, s)
    draw_solids(C, sol)
    label(ax, 275, 283, "VIEW C - END (YZ): nose sketch", fs=8)
    C.cc(0, 0, 10)
    C.cl(-34, 0, 34, 0)
    C.cl(0, -34, 0, 34)
    dim_d(ax, C.P(0, 0), 25 * s, 30, "Ø50 (Sketch.2)", out=14)
    C.dh(-24, 24, C.y(30) + 8, "48", 30, 30)
    C.dv(-30, 30, C.x(-25) - 8, "60", -25, -25)
    text(ax, 275, 150, "Sketch.2 on plane x=40:", fs=6.5, ha="left")
    text(ax, 275, 146, "H=+Y, V=+Z (normal +X)", fs=6.5, ha="left")
    text(ax, 275, 142, "circle D50 at (0,0)", fs=6.5, ha="left")
    feat_block(ax, 24, 66, [
        "1 Sketch.1  plane y=0 (H=+X,V=+Z): (0,-30)-(40,-30)-(40,30)-(0,30), arc R30 c(0,0) through (-30,0) back to (0,-30)",
        "                                                                    -> Pad.1  Mirrored extent half 24 (Y +-24)",
        "2 Sketch.2  plane x=40 (H=+Y,V=+Z, normal +X): circle D50 at (0,0)  -> Pad.2  L=15 (+X), x 40..55",
        "Local origin = J5 axis (world (290,0,610)); wrist fits in the 50 mm forearm slot (24 < 25).  J6 axis along +X at z=0.",
    ])
    return fig


def sheet_P07():
    s = 2.5
    fig, ax = new_part_sheet(8, "P07", "sketch profiles and extrusion", s)
    sol = PART_SOLIDS["P07"]
    A = View(ax, "side", 120, 172, s)
    draw_solids(A, sol)
    label(ax, 30, 273, "VIEW A - END VIEW (YZ), looking along -X: Sketch.1,.2 (disks), Sketch.3,.4 (pockets)")
    A.cc(0, 0, 12)
    A.cl(-32, 0, 32, 0)
    A.cl(0, -32, 0, 32)
    A.circ(0, 0, 15.75, "cl", z=9)
    dim_d(ax, A.P(0, 0), 28 * s, 100, "Ø56 (Sketch.1)", out=10)
    dim_d(ax, A.P(0, 0), 20 * s, 215, "Ø40 (Sketch.2)", out=14)
    dim_d(ax, A.P(0, 0), 10 * s, 300, "Ø20 bore (Sketch.4)", out=50)
    # PCD
    dim_d(ax, A.P(0, 0), 15.75 * s, 340, "PCD 31.5", out=36)
    # 45 deg bolt hole
    hx, hy = 15.75 * math.cos(math.radians(45)), 15.75 * math.sin(math.radians(45))
    leader(ax, A.P(hx + 2.75, hy), (A.x(hx + 2.75) + 12, A.y(hy) + 14), "4x Ø5.5 (Sketch.3) at 45/135/225/315 deg", ha="left")
    # angle marker
    a_pts = [(A.x(0) + 14 * math.cos(math.radians(t)), A.y(0) + 14 * math.sin(math.radians(t))) for t in range(0, 46, 3)]
    polyline(ax, a_pts, closed=False, lw=0.5)
    text(ax, A.x(0) + 16, A.y(0) + 4.5, "45°", fs=6, ha="left")
    line(ax, A.P(0, 0), (A.x(0) + 40, A.y(0)), lw=0.3)

    B = View(ax, "front", 310, 175, s)
    draw_solids(B, sol)
    label(ax, 255, 273, "VIEW B - FRONT (XZ) 2.5:1: thickness")
    B.cl(-4, 0, 22, 0)
    text(ax, B.x(22), B.y(0) + 2.5, "J6 axis", fs=5.8, ha="left")
    # TCP / faces
    line(ax, B.P(15, 24), B.P(15, -24), lw=0.3, ls=(0, (6, 3)))
    B.dh(0, 12, B.y(28) + 8, "12 (Pad.1)", 28, 28)
    B.dh(12, 15, B.y(28) + 16, "3 (Pad.2)", 20, 20)
    B.dh(0, 15, B.y(28) + 24, "15 total", 28, 20)
    yb = B.y(-28) - 8
    B.dh(7, 15, yb, "8 (Pocket.1)", -11.4, -11.4)
    B.dh(9, 15, yb - 8, "6 (Pocket.2)", -10, -10)
    B.dv(-28, 28, B.x(0) - 8, "Ø56", 0, 0)
    B.dv(-20, 20, B.x(15) + 10, "Ø40", 15, 15)
    text(ax, 255, 78, "Pockets are Mirrored extent about plane x=15:", fs=6.5, ha="left")
    text(ax, 255, 74, "bolt holes half 8 (x 7..15 effective), bore half 6 (x 9..15).", fs=6.5, ha="left")
    text(ax, 255, 70, "TCP face x=15; J5 face to TCP: 55 (nose) + 15 = 70 mm.", fs=6.5, ha="left")
    feat_block(ax, 24, 90, [
        "1 Sketch.1  plane x=0 (H=+Y,V=+Z, normal +X): circle D56 at (0,0)                -> Pad.1  L=12 (+X)   x 0..12",
        "2 Sketch.2  plane x=12 (H=+Y,V=+Z):          circle D40 at (0,0)                -> Pad.2  L=3  (+X)   x 12..15",
        "3 Sketch.3  plane x=15 (H=+Y,V=+Z):          4x circle D5.5, PCD 31.5, 45/135/225/315",
        "                                                                                 -> Pocket.1 Mirrored extent half 8",
        "4 Sketch.4  plane x=15 (H=+Y,V=+Z):          circle D20 at (0,0) (centre bore)  -> Pocket.2 Mirrored extent half 6",
        "Hole centres (Y,Z): (+-11.137, +-11.137).  Local origin = J6 axis at x=55 from J5 (world (345,0,610)); TCP = (360,0,610).",
    ])
    return fig


def main():
    makers = [("sheet_01_GA", sheet_GA), ("sheet_02_P01_Base", sheet_P01), ("sheet_03_P02_Turret", sheet_P02),
              ("sheet_04_P03_LowerArm", sheet_P03), ("sheet_05_P04_ElbowHousing", sheet_P04),
              ("sheet_06_P05_Forearm", sheet_P05), ("sheet_07_P06_Wrist", sheet_P06),
              ("sheet_08_P07_Flange", sheet_P07)]
    figs, pngs = [], []
    for name, fn in makers:
        figs.append(fn())
        pngs.append(os.path.join(OUT_DIR, name + ".png"))
    dl.save_pages(figs, os.path.join(OUT_DIR, "robot_drawings.pdf"), pngs, dpi=150)
    print("wrote %d sheets + robot_drawings.pdf in %s" % (len(figs), OUT_DIR))


if __name__ == "__main__":
    main()
