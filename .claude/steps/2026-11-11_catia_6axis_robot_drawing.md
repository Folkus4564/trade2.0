# CATIA V5 package: 6-axis compact robot (LR Mate 200iD/4SC style)

## Goal
From a product photo, produce (1) engineering drawings and (2) everything needed to
build the robot in CATIA V5: per-part dimensioned sketches, a CATScript macro that
auto-builds all 7 parts + the assembly, a step-by-step manual guide, and DMU
Kinematics joint definitions.

Dimensions are APPROXIMATE (estimated from photo + public reach class, ~550 mm reach,
J2 height 330, upper arm 260, J3 offset 20, J3->J5 290, J5->flange 70). Verify against
the manufacturer's official CAD/datasheet before any real use.

## Output folder: `catia_robot_arm/`
- `README.md`                 - step-by-step CATIA guide (manual build, assembly, kinematics, drafting)
- `parameters.csv`            - every driving dimension (name, value_mm, part, description)
- `macros/build_robot.CATScript` - builds P01..P07 + RobotArm.CATProduct, saves to a folder
- `drawings/generate_drawings.py` - matplotlib, produces the PDF + PNGs below
- `drawings/robot_drawings.pdf` - Sheet 1 GA (front/top/side + kinematic schematic + title block), Sheets 2-8 one per part with dimensioned sketch views
- `drawings/sheet_XX_*.png`

## Conventions
- mm. World: Z up = J1 axis, +X forward (arm reach direction), J2/J3/J5 axes parallel to Y, J4/J6 along X.
- Every sketch in the macro uses `sketch.SetAbsoluteAxisData(Array(ox,oy,oz, hx,hy,hz, vx,vy,vz))`
  so H/V directions are explicit (no CATIA plane-orientation ambiguity).
- Offset planes made with `HybridShapeFactory.AddNewPlaneEquation(a,b,c,d)` (unambiguous).
- Pads that must go one way only use sketches whose normal (H x V) points the right way.
  Otherwise use `IsSymmetric = True` ("Mirrored extent": FirstLimit length = HALF thickness).
- Closed profiles: create Point2D endpoints, lines/arcs with StartPoint/EndPoint/CenterPoint set.

## Part geometry (local frames)

P01_Base  (local = world, origin floor centre)
- Plate: rounded rect 190x190, corner R12, sketch XY (H=X,V=Y), pad 15 (+Z)
- 4x hole D11 at (+-80,+-80): pocket, symmetric, half-depth 50
- Housing: rounded rect 150x150 R20 on plane z=15, pad 150 (z 15..165)
- Rear connector box: rect X -105..-75, Y -45..45 on plane z=15, pad 100

P02_Turret  (local origin = world (0,0,165))
- Disk D140, sketch XY, pad 30 (z 0..30)
- Shoulder: XZ profile (H=X,V=Z): (-60,30)->(60,30)->(60,165), arc R60 c(0,165) over top ->(-60,165)->close; symmetric pad half 55 (Y +-55). J2 at local z=165.
- J2 motor cover: circle D100 at (x0,z165) on plane y=-75, symmetric pad half 20 (Y -95..-55)

P03_LowerArm  (local origin = J2; world (0,85,330))
- Stadium XZ: R50 at (0,0) and (0,260), lines x=+-50; symmetric pad half 30 (Y +-30)
- Cosmetic hub caps: circle D70 at (0,0) and (0,260) on plane y=30, pad 5 (outer side, +Y)  -- use H=Z? choose axes so normal is +Y: H=(0,0,1),V=(1,0,0)  => ZxX=+Y

P04_ElbowHousing  (local origin = J3; world (0,0,590))
- XZ profile: bottom semicircle R55 c(0,0) from (55,0) to (-55,0) through (0,-55); (55,0)->(55,65)->(-55,65)->(-55,0). Symmetric pad half 55.

P05_Forearm  (local origin on J4 axis at elbow front face; world (55,0,610))
- Shaft about X: XZ sketch (H=X,V=Z) closed (0,0)->(0,45)->(200,35)->(200,0)->(0,0), centerline = line on V=0; AddNewShaft 360 deg
- Yoke: XZ profile (200,-35)->(235,-35), arc R35 c(235,0) to (235,35)->(200,35)->close; symmetric pad half 40
- Yoke slot: rect X 200..280, Z -50..50, symmetric pocket half 25 (leaves two 15 mm prongs). J5 at local x=235.

P06_Wrist  (local origin = J5; world (290,0,610))
- XZ profile: (0,-30)->(40,-30)->(40,30)->(0,30), arc R30 c(0,0) through (-30,0) back to (0,-30); symmetric pad half 24
- Nose: circle D50 on plane x=40 (H=Y,V=Z => normal +X), pad 15 (x 40..55)

P07_Flange  (local origin = J6 axis at x=55 from J5; world (345,0,610))
- Disk D56 on YZ (H=Y,V=Z), pad 12 (x 0..12); tool disk D40 on plane x=12, pad 3 (face at x=15 => J5-face = 70)
- 4x D5.5 on PCD 31.5 at 45/135/225/315 deg, centre bore D20: pockets from plane x=15, symmetric half 8 (bolts) / 6 (bore)

## Assembly (home pose, all joints 0)  -- Move.Apply 12-value matrix, identity rotation
P01 (0,0,0) P02 (0,0,165) P03 (0,85,330) P04 (0,0,590) P05 (55,0,610) P06 (290,0,610) P07 (345,0,610)
TCP (flange face centre) = (360, 0, 610).

## Kinematics (DMU)  -- revolute joints, approx limits (verify)
J1 Base-Turret axis Z +-170 | J2 Turret-LowerArm axis Y -110/+120 | J3 LowerArm-Elbow axis Y -70/+200
J4 Elbow-Forearm axis X +-190 | J5 Forearm-Wrist axis Y +-125 | J6 Wrist-Flange axis X +-360
DH (modified, approx): d1=330, a2=260, a3=20, d4=290, d6=70.

## Verification
- Run generate_drawings.py -> PDF with 8 sheets, views match the numbers above.
- Static check of the CATScript (balanced Sub/End Sub, every pad references existing sketch).
- CATIA itself is not available in the container: the macro is untested in CATIA; README tells the user how to run it and what to do if a line errors.

## Completed
2026-11-11 - commit 2e59e75. Drawings rendered and visually checked; macro statically checked, not run in CATIA.
