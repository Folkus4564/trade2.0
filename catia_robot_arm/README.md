# 6-Axis Compact Robot Arm (LR Mate 200iD/4SC style) - approximate - CATIA V5 package

Everything needed to draw and build a compact white 6-axis industrial robot (tall square base pillar,
side-mounted lower arm, rounded elbow housing, horizontal forearm, small wrist, ISO tool flange) in CATIA V5.
No manufacturer logo/branding is used. **All dimensions are approximate** (estimated from a photo and the
~550 mm reach class) - verify against official CAD/datasheet before any real use.

## 1. What is included

| File | Purpose |
|------|---------|
| `drawings/robot_drawings.pdf` | 8 sheets, A3 landscape: Sheet 1 general arrangement (front/top/side, kinematic schematic, joint table, parts list); Sheets 2-8 one per part P01..P07 (dimensioned sketches + extrusion view + feature recipe) |
| `drawings/sheet_01_GA.png` ... `sheet_08_P07_Flange.png` | Same sheets as 150 dpi PNG |
| `drawings/generate_drawings.py`, `drawings/drawlib.py` | Regenerate drawings (`python generate_drawings.py`; matplotlib + numpy only) |
| `macros/build_robot.CATScript` | CATIA macro: builds P01..P07 CATParts + `RobotArm.CATProduct` |
| `parameters.csv` | Every driving dimension (name, value_mm, part, description) |
| `README.md` | This guide |

Parts: P01 Base, P02 Turret, P03 LowerArm, P04 ElbowHousing, P05 Forearm, P06 Wrist, P07 Flange.

## 2. Coordinate system and conventions

* Units mm. World: **Z up = J1 axis**, **+X = arm reach direction**, J2/J3/J5 axes parallel to **Y**, J4/J6 along **X**.
* Each part is modelled in a local frame; the assembly only translates (identity rotation) at home pose (all joints 0).
* Sketch axes are always defined explicitly (H = horizontal, V = vertical); pad direction = sketch normal = H x V.
* "Mirrored extent" features: the CATIA length field is **half** of the total thickness (see note in 6.3).

| Part | Local origin | Home translation (world) | Axis through origin |
|------|--------------|--------------------------|---------------------|
| P01 Base | floor centre | (0, 0, 0) | J1 (Z) |
| P02 Turret | top of base | (0, 0, 165) | J1; J2 at local z=165 |
| P03 LowerArm | J2 axis point | (0, 85, 330) | J2 (Y); J3 at local z=260 |
| P04 ElbowHousing | J3 axis point | (0, 0, 590) | J3 (Y); J4 at local z=20 |
| P05 Forearm | J4 axis at elbow front face | (55, 0, 610) | J4 (X); J5 at local x=235 |
| P06 Wrist | J5 axis point | (290, 0, 610) | J5 (Y); J6 at local x=55 |
| P07 Flange | J6 axis, 55 from J5 | (345, 0, 610) | J6 (X); face at local x=15 |

TCP (flange face centre) = (360, 0, 610). Overall height 655, base 190 x 190, overall length -105..360, overall width Y -95..115.

## 3. Assumptions / approximations

* Reach ~550 mm class; J2 height 330, upper arm 260, J3 offset 20, J3-J5 290, J5-flange 70 (spec values).
* Shapes simplified to extruded profiles: no fillets, no cable pockets, hub caps cosmetic, shaft is a straight cone D90 -> D70.
* Joint limits are approximate; mechanical interferences between parts (e.g. forearm shaft vs elbow arc) are not resolved.
* The macro and instructions are **untested in CATIA** (CATIA was not available when written).

## 4. Option A - build everything with the macro

1. Open `macros/build_robot.CATScript` in a text editor; edit `Const SAVE_DIR = "C:\RobotArm"` if desired (folder is created if missing).
2. In CATIA V5 start with no document open (or any), `Tools > Macro > Macros...`.
3. Click **Macro libraries...**, **Add existing library**, choose the `macros` folder (or pick `Files` and the .CATScript file), select it.
4. Select macro `CATMain`, click **Run**. Seven part documents open/save/close one after another, then `RobotArm.CATProduct` is created and saved.
5. Open `SAVE_DIR\RobotArm.CATProduct` and check against `drawings/sheet_01_GA.png`.

Troubleshooting (the macro stops with an error dialog showing the failing line):

| Symptom | Fix |
|---------|-----|
| Error on `SetAbsoluteAxisData` or sketch looks mirrored/rotated | Delete the sketch, create it manually per section 5 with the listed plane and axes (use `Sketch positioning` dialog: origin + H/V directions) |
| Error on `sk.CenterLine = ...` / Shaft fails | Draw the shaft sketch manually, pick the axis line, **Insert > Shape > Shaft**; or sketch half profile + axis in Part Design |
| Pad goes the wrong way (non-mirrored) | In the pad dialog click **Reverse Direction** (affects P01 plate/housing/connector, P03 caps, P06 nose, P07 disks) |
| Mirrored feature is twice/half as thick | Change the length: in this CATIA release the mirrored length is the total (or half) - set so total matches column "total thickness" in `parameters.csv` |
| Arc is wrong side / error on arc end points | Arc angles are counter-clockwise start->end; redraw that arc with the 3-point arc tool through the listed points |
| `AddComponentsFromFiles` / `Move.Apply` fails | Insert components with **Insert > Existing Component** and position with **Manipulation** or constraints (section 5.2) |
| Macro stops half way | Fix the failing feature manually, save the part, comment out finished `BuildPxx` calls in `CATMain` and rerun |

Tip: to switch a single part to manual, comment out its `BuildPxx` line in `CATMain`, save your manual part as `SAVE_DIR\P0x_Name.CATPart`, and the assembly step still picks it up.

## 5. Option B - manual build

### 5.1 Part Design (per part; Start > Mechanical Design > Part Design)

For every sketch: select the plane (create offset planes with **Plane** > *Equation* or *Offset from plane*), enter **Sketcher**, set the axes if needed (`Sketch positioning`), draw the profile, add constraints (dimensions from the tables), exit, then apply the feature.

**P01_Base**

| # | Plane / axes (H, V) | Sketch geometry + constraints | Feature |
|---|---------------------|-------------------------------|---------|
| 1 | XY, z=0 (X, Y) | Rectangle 190 x 190 centred on origin, 4 corner fillets R12 | Pad 15 (+Z) |
| 2 | XY, z=0 | 4 circles D11 centred (+-80, +-80) | Pocket, Mirrored extent, length 50 (through the plate) |
| 3 | z=15 plane (X, Y) | Rectangle 150 x 150 centred, corner R20 | Pad 150 (+Z) |
| 4 | z=15 plane (X, Y) | Rectangle X -105..-75, Y -45..45 | Pad 100 (+Z) |

**P02_Turret** (origin = world (0,0,165))

| # | Plane / axes | Sketch | Feature |
|---|--------------|--------|---------|
| 1 | XY, z=0 (X, Y) | Circle D140 at origin | Pad 30 (+Z) |
| 2 | XZ (H=X, V=Z) | (-60,30)-(60,30)-(60,165), arc R60 centre (0,165) over the top to (-60,165), close | Pad Mirrored extent, half 55 (Y +-55) |
| 3 | plane y=-75 (H=X, V=Z) | Circle D100 centre (0,165) | Pad Mirrored extent, half 20 (Y -95..-55) |

**P03_LowerArm** (origin = J2)

| # | Plane / axes | Sketch | Feature |
|---|--------------|--------|---------|
| 1 | XZ (H=X, V=Z) | Stadium: arcs R50 at (0,0) and (0,260), vertical lines x=+-50 | Pad Mirrored extent, half 30 |
| 2 | plane y=30, H=Z, V=X (normal +Y) | Circles D70 at (h,v)=(0,0) and (260,0) | Pad 5 (outward, +Y) |

**P04_ElbowHousing** (origin = J3)

| # | Plane / axes | Sketch | Feature |
|---|--------------|--------|---------|
| 1 | XZ (H=X, V=Z) | Semicircle R55 centre (0,0) below (through (0,-55)); lines (55,0)-(55,65)-(-55,65)-(-55,0) | Pad Mirrored extent, half 55 |

**P05_Forearm** (origin = J4 axis at elbow front face)

| # | Plane / axes | Sketch | Feature |
|---|--------------|--------|---------|
| 1 | XZ (H=X, V=Z) | Closed (0,0)-(0,45)-(200,35)-(200,0)-(0,0) + construction axis line on V=0 (0,0)-(200,0) | **Shaft** 360 deg about the axis |
| 2 | XZ | (200,-35)-(235,-35), arc R35 centre (235,0) to (235,35), (235,35)-(200,35), close | Pad Mirrored extent, half 40 |
| 3 | XZ | Rectangle X 200..280, Z -50..50 | Pocket Mirrored extent, half 25 (two 15 mm prongs) |

**P06_Wrist** (origin = J5)

| # | Plane / axes | Sketch | Feature |
|---|--------------|--------|---------|
| 1 | XZ (H=X, V=Z) | (0,-30)-(40,-30)-(40,30)-(0,30), arc R30 centre (0,0) through (-30,0) back to (0,-30) | Pad Mirrored extent, half 24 |
| 2 | plane x=40, H=Y, V=Z (normal +X) | Circle D50 at (0,0) | Pad 15 (x 40..55) |

**P07_Flange** (origin = J6 axis, 55 from J5)

| # | Plane / axes | Sketch | Feature |
|---|--------------|--------|---------|
| 1 | YZ (H=Y, V=Z), x=0 | Circle D56 | Pad 12 (+X) |
| 2 | plane x=12 (H=Y, V=Z) | Circle D40 | Pad 3 (face at x=15) |
| 3 | plane x=15 | 4 circles D5.5 on PCD 31.5 at 45/135/225/315 deg (centres (+-11.137, +-11.137)) | Pocket Mirrored extent, half 8 |
| 4 | plane x=15 | Circle D20 at origin | Pocket Mirrored extent, half 6 |

After each part: **File > Save As** `P0x_Name.CATPart`.

### 5.2 Assembly Design (Start > Mechanical Design > Assembly Design)

1. New Product `RobotArm`. **Insert > Existing Component** all 7 parts.
2. **Fix** P01 (Constraints toolbar, Fix component).
3. For each joint add: (a) **Coincidence** between the joint axes (select the cylindrical/centre axis or the shared origin axes of both parts), (b) **Contact** or **Offset** between the faces listed below. Alternatively place directly with **Manipulation > Snap/Move** using the translations from section 2.

| Joint | Parts | Coincidence (axes) | Contact / offset (faces) | Offset |
|-------|-------|--------------------|--------------------------|--------|
| J1 | P01 - P02 | Z axis of P01 with Z axis of P02 | P01 housing top (z=165) with P02 disk bottom | contact, 0 |
| J2 | P02 - P03 | Y-parallel axis through P02 (0,*,330) with P03 origin axis | P02 shoulder +Y face (y=55) with P03 face (y=-30 local) | contact, 0 (P03 at y = 85) |
| J3 | P03 - P04 | P03 axis at local z=260 with P04 origin axis (Y) | P03 -Y face (y=55 world) with P04 +Y face (y=55) | contact, 0 |
| J4 | P04 - P05 | P05 X-axis with the J4 axis line (z=20 above P04 origin) | P05 shaft start face with P04 front face x=55 | contact, 0 |
| J5 | P05 - P06 | P06 Y-axis through origin with P05 yoke centre (x=235) | Yoke prong inner faces (y=+-25) with wrist faces (y=+-24) | offset 1 each side (or centre) |
| J6 | P06 - P07 | P07 X-axis with P06 nose axis | P06 nose end face (x=55 local) with P07 face x=0 | contact, 0 |

Check: P07 face centre at world (360, 0, 610) = TCP.

### 5.3 DMU Kinematics (Start > Digital Mockup > DMU Kinematics)

1. New Mechanism `Robot`. Fix P01 (Fixed Part).
2. Create **6 Revolute joints** (Insert > Joint > Revolute); in each select the two **line axes** and the two **planes** (same pairs as the assembly table). Tick **Angle driven**:

| Joint | Part 1 - Part 2 | Axis | Angle limits |
|-------|-----------------|------|--------------|
| J1 | P01 - P02 | Z | -170 / +170 |
| J2 | P02 - P03 | Y | -110 / +120 |
| J3 | P03 - P04 | Y | -70 / +200 |
| J4 | P04 - P05 | X | -190 / +190 |
| J5 | P05 - P06 | Y | -125 / +125 |
| J6 | P06 - P07 | X | -360 / +360 |

3. Enter the limits in the joint dialog (**Angle driven** > *Angle limits* checkboxes). Keep joints at 0 for the home pose.
4. The status must show "Degrees of freedom 0" and "Mechanism can be simulated with commands".
5. **Simulation > With commands**: drag the sliders per joint; **Insert > Simulation > Insert** to record, **Compile Simulation** to play. Optionally add a **Sensor** on the TCP point.
6. (Approx. modified DH: d1=330, a2=260, a3=20, d4=290, d6=70.)

### 5.4 Drafting (Start > Mechanical Design > Drafting)

1. **New drawing**: Standard ISO, Sheet A3 ISO, Landscape, scale 1:5 (assembly) or per part; **Projection method = First angle**.
2. Open the product window, **Front View** from a plane (XZ): **Insert > Views > Projections > Front View**, then **Top View** and **Left View** (Left view = right-side view in first angle) and optionally **Isometric View**.
3. **Generate dimensions**: *Dimensioning > Generation > Generate Dimensions*; remove duplicates; add overall dimensions (base 190, J2 330, J3 590, J4 610, J5 X 290, TCP X 360, height 655) with the Dimension tool.
4. Centre lines/axes: *Generation > Create Axis / Centerline*.
5. **Title block**: *Insert > Drawing > Frame and Title Block* (ISO), text: title "6-Axis Compact Robot Arm (LR Mate 200iD/4SC style) - approximate", scale, units mm, sheet n/8, date 2026-11-11, note "Dimensions approximate - verify before manufacture".
6. Per part: new sheet, same steps, show the sketch dimensions from the tables above.

### 5.5 Optional finishing

* **Edge Fillet** (Dress-Up): R2 on housing/turret vertical edges, R3 on arm edges, R5 on elbow/yoke outer edges for the smooth look.
* **Apply Material** (Render) or *Properties > Graphic*: white colour (RGB 245/245/245), small reflective specular; use black for the flange and base plate bolts.
* Optionally add chamfers on the flange (0.5 x 45 deg) and a grey ring (R2 edge) at the J2/J3 hub caps.

## 6. Notes

1. `parameters.csv` lists every driving dimension; changing a value there means editing it in the macro/part too (no formula link is created).
2. Features marked "Mirrored extent half X" correspond to a total thickness of 2X.
3. Mirrored extent length semantics: some CATIA releases interpret the Pad length as the total thickness when *Mirrored extent* is ticked. The macro sets half. If your result is twice/half as thick, edit the length in the Pad/Pocket dialog.
4. Run `python drawings/generate_drawings.py` to regenerate the PDF/PNGs after changing geometry.
