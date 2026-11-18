---
name: "blockbench-sculpted-style"
description: "Use when designing, critiquing or generating Blockbench/Minecraft-style 3D models (dragons, Monster Hunter-style monsters, environment creatures, gathering points, props, vehicles, scenes) in the sculpted cuboid, hand-painted pixel-texture style, including SyncCraft Block-DSL assets. Pass it to every subagent."
---

# Sculpted Blockbench style

This skill defines one look and one way of working.

**The look:** large, deliberate cuboid masses; real block horns and spikes; cut-out planes for thin edges; calm, banded, vibrant pixel paint. It is the look of top Blockbench creature artists, sculpture made of boxes. It is **not** voxel art, **not** a scatter of shards, and **not** vanilla Minecraft minimalism.

Models that fail this style rarely fail at *seeing* the reference. They fail at **construction**: one form gets broken into many small, randomly angled pieces, the face becomes a long flat box, horns become sheets or sticks, and the paint gets noisy. Most rules below exist to stop exactly that.

## 0. How to use this skill

- **Read it in full** before the first reference review, the first construction step and the first critique of every pass. Read it again whenever a pass is rejected. Re-read Section 1 before every build or edit call.
- **Open the approved style anchors** (2.3) every time you read the skill. If you cannot open them, say so in the pass log; never describe an image you did not see.
- **Subagents:** give every subagent this skill, the anchor images and the brief in Section 12. A subagent that has not read both may not build, review or paint.
- **Start clean.** Do not reuse models, scripts, textures or notes from earlier projects as templates. Reuse only this skill, the anchors and SyncCraft.
- **Keep a project log** (`PROJECT_GOAL.md`) holding:
  - the non-negotiables;
  - the roster;
  - each model's current pass;
  - the owner's latest corrections, word for word.

  Re-read it at the start of every pass so instructions survive context compression.

## 1. Non-negotiables

1. **Construct, don't scatter.** Every box is a named form (skull, brow, cheek, thigh, horn segment 2, plate 3 in row B). Before adding a box, name its form and the view in which it changes the silhouette. If you can't, don't add it.
2. **Big first.** Primary masses keep most of the volume: at every stage the 10 largest boxes hold at least half of all box volume. Detail never replaces a mass.
3. **No fake curves or spheres.** Never stack or slice many small boxes to imitate a sphere, dome, smooth curve, long slope or tapering spike. Use the remap table (3.2).
4. **Real block horns and spikes.** A horn is 2-4 thick box segments; a spike is one box. Never a flat plane for a horn, never a staircase of cubes.
5. **Every rotation has a reason:** the pose, the form's own slope, or a row's shared lean. No random jitter. Mirrored parts get mirrored angles.
6. **Everything connects.** Each element touches or overlaps a neighbour. Overlap is good; floating parts are bugs. Cross-bone seams use `blend_with`.
7. **Build symmetric.** Build only the left side and centre; the right comes from mirroring. The rest pose is symmetric (Section 4). Attitude lives in animation, not in the model.
8. **The head passes first.** Build and approve the head alone before the body (6.6). The jaw is a separate bone on anything player-sized or larger.
9. **Place eyes, don't drop them.** Eyes sit on the side of the skull at the brow-snout junction, under the brow, in the upper half of the head (6.3).
10. **Epic for monsters, simple only when small.** Creatures as large as the player or larger get deliberate hero-level form. Only creatures smaller than the player (fish, birds, insects, small critters) may be simplified.
11. **Calm, clean paint.**
    - 85-90% of each surface is calm mid-tone.
    - Banded light, no per-pixel noise.
    - One accent colour.

    "Dirty" means too much contrast in too many places (Section 10).
12. **Minimum passes:** 3 white-model, 2 detail, 2 texture, plus 1 rig-and-motion check. More passes are welcome. More passes never permit more fragments.
13. **Punch back early.** A model that fails a gate in 5.4 returns to white pass 1. Never patch a broken base with detail.
14. **Count is an outcome.** Budgets are ceilings, not targets. Never add elements to reach a number or raise a score.
15. **Audit, then look.** A passing SyncCraft audit is necessary, never sufficient. Every pass, compare renders against zoomed reference crops.

## 2. The look

### 2.1 What it is
- **Form**
  - A few large cuboids, rotated as whole masses to make slopes and gesture.
  - They overlap at joints so the body reads as one continuous form.
  - Silhouette first: each part reads as a dark shape from front, side, three-quarter and top.
- **Hierarchy, about 70 / 25 / 5 by volume:**
  - primary masses;
  - secondary forms that change the outline (brow, cheek, jaw, shoulder, knee, horn);
  - tertiary accents (claws, teeth, fangs, spike tips).
- **Pushed proportions:** heads, horns, claws and the signature feature are bigger than life, and everything tapers thick to thin.
- **Thin things are planes with cut-out alpha:** membranes, fins, frills, fur tufts, manes, leaves, cloth.
- **Paint**
  - Hand-painted pixel art at one texel density.
  - 3-5 banded values per colour, with hue-shifted ramps (warm lights, cool shadows).
  - A 1 px light top edge and a 1 px dark contact edge.
  - A limited vibrant palette: dominant, supporting and one accent.
  - Glow is a small, focal material.

### 2.2 What it is not
- **Shard armour:** dozens of small slabs at random angles covering the body. This is the most common failure (see 2.5).
- **Voxel volumes:** staircase spheres, domes and slopes.
- **Horns as stick chains or sheets;** spikes as cube staircases.
- **A long flat face:** the head as one long box with features painted on, or eyes in the wrong place.
- **Toy monsters:** few boxes, straight limbs, flat colours, everything axis-aligned on a creature that should be epic.
- **Dirty paint:** noise, many high-contrast specks, black outlines, grey shadows.
- **Repulsive rendering of creepy creatures:** slime, veins, gore, eye clusters (see 9.5).

### 2.3 Style anchors (owner-approved)
Open these every time you read the skill. Community art you find yourself becomes an anchor only after the owner approves it; the owner has rejected agent-picked references before.
- The owner's image set (5 dragons, 4 scenes), described in 2.4.
- **Primary anchors:** x.com/zyhfisme/status/1733516648437719544 and x.com/zyhfisme/status/1733044355840213081.
- x.com/AnzSamaEr (including status 1965148236198674683).
- ArtStation `g0BAae` and `obdg2m`; Pinterest pins 822188475757578039 and 978618194031219174; Sketchfab "stygian-zinogre"; YouTube `y6y96xnMplM`.
- blockbench.net/gallery. **Ignore the fox piece.**
- **Cobblemon assets** (gitlab.com/cable-mc/cobblemon-assets, CC-BY-NC). Use them for construction discipline: clean large forms, strict symmetry, separate jaw and eye elements ready for animation. Their creatures are simpler than this style's monsters, so borrow the discipline, not the detail level.

If a URL cannot be opened in your environment, use the owner's saved copies in the project's `style_anchors/` folder. If there are none, ask.

### 2.4 What the owner's set shows
1. **Rotated, stacked cuboids that read as flowing curves.** You can count the boxes, but the outline reads as one flowing form.
2. **Silhouette edges drawn in pixels:** torn membranes, fringes, drips and tatters on cut-out planes. Hard box edges alone look unfinished.
3. **Banded cel shading** in 3-5 steps per colour; shadows toward violet or blue, lights toward cyan, yellow or cream.
4. **Two body colours plus one hard-part colour plus at most one glow:** red and cream, blue and navy, tan and sand, with ivory horns.
5. **Segmented undersides are geometry:** wide overlapping plates in the underside colour.
6. **Round machine parts are octagons** (wheels, hubs, pots). Round *organic* parts are single boxes with painted light.
7. **Saturation is controlled:** the strongest colour and value sit at the focal point.

Per image:
- **Red and cream dragon:** thick spar chains, a dithered cream-to-tan membrane per bay with a ragged trailing edge, pale antler-like horns that change direction at each segment, cream toes.
- **Black and blue elder dragon:** dense detail that is still *organised*. A row of rounded scale boxes armours the wing's leading edge; cyan crystal horns are the only saturated accent; a pale chest panel.
- **Blue ice dragon:** a head about a third of the body mass, a heavy block jaw with a row of yellow teeth, one red eye set deep under the brow, crystal prisms on rock bases.
- **Pink chibi dragon:** about 25 boxes, a head nearly half the body. Proof that the style scales down without losing banded paint or a clear silhouette.
- **Tan club-tailed wyvern:** thick wing arms, sand membranes, a row of curved dorsal spikes, a studded mace tail, grey segmented ram horns.
- **Plated beast and quill beast:** built from overlapping slabs in *coherent rows* that all lean the same way. That is density, not scatter.
- **Scenes:** warm emissive lights against cool ambient; props that lean slightly; chamfered hard surfaces; weathering that follows gravity.

### 2.5 Exemplars
- **Good: "Saffron Squallwing" test flight,** which the owner judged amazing.
  - 151 elements, 50 bones, about 2.5 boxes per bone.
  - The 10 largest boxes hold 55% of the volume; the median element is about 6 x 3.5 x 6 units on a 170-unit span.
  - Partitioned membranes, 4-segment block horns, a banded texture with chevrons on the dorsal parts only and calm flanks.
  - **Its one rule break:** the head was turned 8 degrees and the tail curved sideways in the rest pose, which broke mirror symmetry. That attitude belongs in a display clip.
- **Bad: SyncCraft v9 Zinogre** (repo commit `ba969f3`).
  - 262 elements, 6.4 boxes per bone; the 10 largest boxes hold only 48% of the volume.
  - The median element is about 2 x 3 x 5 units on a 94-unit body.
  - The white render is shard armour: the head is a pile of plates with no readable skull, snout, jaw or eye; the horns are thin bars; the tail is a stack of tilted slabs; mirror parity is broken.

## 3. Construction grammar

### 3.1 Build order inside any part
1. Host mass.
2. Secondary forms that change the outline.
3. Rows of plates following the flow lines.
4. Tips: claws, teeth, spike points.
5. Planes for thin edges.
6. Paint.

Each step must improve the silhouette or the focal area. If it doesn't, undo it.

### 3.2 Remap table: real shapes into box language
| Real shape | Build it as | Never |
|---|---|---|
| Sphere, egg, round belly, dome, skull cap | 1 box, optionally plus 1-2 chamfer boxes on the edges that show in silhouette; painted light does the roundness | stacked rings, concentric rotated boxes, `prism(n)` with n > 8, slices |
| Long slope (snout ridge, back line, wing root) | 1 box rotated to the slope; 2 overlapping boxes if the angle changes | a staircase of offset boxes |
| Smooth curve (neck, tail, horn arc) | 3-5 seated segments with a steady angle step (10-25 degrees) and steady taper; each segment at least 1.5x as long as it is thick | 8+ thin slices; segments shorter than their thickness |
| Spike, thorn, claw, fang | 1 box, pointed by its size, texture or cut-out; 2 boxes (base plus tip) only if it is longer than 4x its base and bends | 3+ shrinking cubes; a flat plane |
| Horn | 2-4 thick box segments (7.1) | a plane; thin sticks; a cube staircase |
| Many thin spikes (quills, fur spikes, crystal fringe) | cut-out planes (`edge: spikes`), 2-3 layers | dozens of thin boxes |
| Membrane, fin, frill, ear fin | plane(s) with alpha cut-out, one panel per bone | thick slabs; one plane spanning moving bones |
| Fur, mane | 2-3 layers of cut-out planes or tilted slabs in rows | a fuzzy block; scattered cubes |
| Armour plates, scales | rows of slabs with the same lean, overlapping 30-50%; small scales are painted | random shards |
| Tentacle, trunk, whip tail | 4-7 seated segments, each 0.8-0.85 of the previous, gentle steady curl | many short slices; uniform pipes |
| Wheel, hub, cylinder (hard surface) | `octagon` (2 boxes) | `prism(12)` on organic parts |
| Eye | painted stamp on the skull's side face; a shallow box only for bulging eyes (frogs, insects, chameleons) | a cube stuck on the surface |

### 3.3 Rotation discipline
- **Allowed sources:**
  - (a) the pose, through bone rest angles;
  - (b) the form's own slope, for example a snout pitched down 5-15 degrees;
  - (c) a row's shared lean: all plates in a row lean back about 15 degrees, changing by at most 5 degrees along the row, in a steady progression.
- **Not allowed:**
  - per-box random angles;
  - "hand-made wobble" on structural parts;
  - rotation added to raise a style score.
- Comment every non-zero `rotate` with its reason (`// slope`, `// row lean`, `// pose`).
- Prefer clean angles: 5, 7.5, 10, 15, 22.5, 30, 45.

### 3.4 Size floors
- **On creatures larger than the player,** no box is thinner than 1.5 units in two dimensions. Exceptions: tips (claw, tooth, fang, horn tip, spike tip) and eyes. Finer detail goes into texture or a cut-out plane.
- **Minimum add-on size:** a plate or add-on is at least one eighth the size of the part it sits on. Anything smaller is painted.

### 3.5 Connection
- Every element touches or overlaps a neighbour; nothing floats.
- Seat chain segments (neck, spine, tail, horn) 25-30% into the parent: `attach(..., seat=0.28)`.
- **Overlap where forms flow together,** but each box must still show a real part of itself. SyncCraft rejects a box that is more than about 65% buried.
- **Cross-bone seams use `blend_with`.** For a seam into a mirrored subtree, list both names, because `mirror_x` does not rename `blend_with` targets.
- **Joint test:** rotate each joint 30 degrees both ways. Overlap must cover the gap.

### 3.6 Construction check (run every pass)
**Guards:**
- An average of at most 3.5 boxes per bone outside the head. Any bone with more than 8 boxes needs a written reason.
- The 10 largest boxes hold at least 50% of box volume.
- Mirror parity is OK.

These are starting thresholds, measured on one good model and several bad ones; tighten them with your own approved models. Passing them does not make a model good. Failing them almost always means shard armour or broken symmetry.

```python
# usage: python construct_check.py <SyncCraft repo> <model.blockdsl>
import sys, collections, numpy as np
sys.path.insert(0, sys.argv[1] + "/src")
from synccraft.dsl.parser import parse_program
from synccraft.kinematics.se3_chain import compile_geometry
s = compile_geometry(parse_program(open(sys.argv[2]).read()))
boxes = [b for b in s.boxes if min(b.size) > 0]
vol = np.array(sorted((float(np.prod(b.size)) for b in boxes), reverse=True))
per_bone = collections.Counter(b.bone_name for b in boxes)
key = lambda b, sx: (round(sx * float(b.center[0]), 1), round(float(b.center[1]), 1), round(float(b.center[2]), 1))
left = sorted(key(b, -1) for b in s.boxes if b.center[0] < -0.05)
right = sorted(key(b, 1) for b in s.boxes if b.center[0] > 0.05)
print(f"elements {len(s.boxes)}  boxes/bone {len(boxes)/len(per_bone):.1f}  top-10 volume {vol[:10].sum()/vol.sum():.0%}")
print("busiest bones", per_bone.most_common(4))
print("mirror parity", "OK" if left == right else f"BROKEN ({len(left)} left vs {len(right)} right)")
```

## 4. Symmetry and the rest pose
- **Build only the left side (-X) and the centreline.** Generate the right side with `mirror_x(name)` or `mirror(axis=x)` on bones and macros, and `mirror: true` on stamps. Never hand-place a right-side part.
- **Centreline parts:** centre at x = 0 exactly, symmetric sizes, no yaw or roll.
- **The rest (bind) pose is symmetric about the X = 0 plane.**
  - Spine, neck, head and tail use pitch only.
  - Left and right limbs mirror each other.
  - Gesture comes from pitch: an S-curved neck, a lifted tail, a lowered head.
- **Attitude goes in a display pose or animation clip,** never the rest pose. That means a turned head, a tail swing, a raised paw or one wing higher.
- **Real asymmetric anatomy** (a broken horn, a one-sided molt) is declared in the trait sheet. Build it as a clearly named overlay *after* symmetry is verified.
- **Verify three ways:**
  1. the construction check (mirror parity);
  2. `analyze_morphology(program).symmetry_pairs`;
  3. front and top renders beside a mirrored copy.

## 5. Pipeline and pass gates

### 5.1 Reference board (before any box)
- **At least 10 images per model, filed by category:**
  - front, side, back, top;
  - three-quarter front and back;
  - head close-ups (side and front);
  - limb or foot close-up;
  - wing or tail close-up;
  - official render;
  - gameplay screenshot;
  - approved community Blockbench work.
- **Label missing views.** If images cannot be viewed in your environment, say so and ask for them. Never invent what a picture shows.
- **Part map.** For each region (head, neck and torso, forelimbs or wings, hind limbs, tail, surface), write 2-5 lines in head-lengths: shape, proportion, angles, counts of horns, spikes and toes, and colours. Mark each trait keep, exaggerate, merge, paint or drop. Geometry is for whatever changes the silhouette; texture is for everything else.
- **Zoomed crops** of the head, horns, limbs and tail tip are what every regional check compares against.

### 5.2 Passes
| Pass | Goal | Must produce |
|---|---|---|
| W1 Blockout | 12-25 primary masses, gesture, proportions in head-lengths; head gate (6.6) | white renders: front, side, top, three-quarter front and back, side and front silhouettes |
| W2 Secondary | forms that change the outline: brow, cheek, jaw, shoulder and haunch, knees and elbows, horn segments, wing fingers, feet | the same, plus a region sheet: head side, three-quarter and front; torso side; limb side; wing top; tail top |
| W3 Refine | fix every regional critique; recheck proportions with numbers; delete or merge anything that does not change the silhouette | the same |
| D1 Detail | plate rows, claws, teeth, spike rows, membranes, fur planes, on focal areas first | the same |
| D2 Detail | tilts, joint overlaps, connection and symmetry; remove anything noisy | the same |
| T1 Texture | base colours per material, banded light, features (eyes, mouth, markings) | textured renders plus the atlas |
| T2 Texture | calm the surfaces, sharpen focal contrast, place glow, squint test | the same |
| A1 Rig and motion | pivots at joints, 30-degree test at every joint, symmetric rest pose, idle plus walk (`gait`), swept-motion audit | clip previews or key stills |

The white model is a draft. It must survive W2 and W3 before any detail.

### 5.3 Every pass
1. Re-read this skill, open the anchors, and open this model's reference board.
2. Build or edit.
3. Run the SyncCraft audit and the construction check (3.6).
4. Render the view set and the region sheet.
5. For each region, write: "reference shows ...; model shows ...; change ...".
6. Log in `PROJECT_GOAL.md`: what changed, the check results and the open issues.

### 5.4 Punch back to W1 when any of these is true
- The silhouette does not read as the subject from the side and front.
- The head fails its gate or its proportions (6.1).
- Mirror parity is broken without a declared reason.
- The construction guards fail (3.6).
- A forbidden construction from 3.2 is present.
- The owner or a reviewer says it does not look like the reference or the style.

Detail and texture passes may not be used to fix base problems.

## 6. Heads (most failures happen here)

### 6.1 Proportions
- **Head length to head height** (back of skull to nose tip, against top of cranium to bottom of the closed jaw):

  | Head type | Length : height |
  |---|---|
  | Wolf-like and fanged heads | 1.3-1.7 |
  | Flying wyverns and dragons | 1.6-2.1 |
  | Brutes | 1.2-1.6 |
  | Long-snouted leviathans and crocodilians | up to 2.6 |

  Above the range, the face is too long: **shorten the snout first.**
- **Muzzle:**
  - length 0.6-1.0x the cranium length;
  - width 0.6-0.8x the cranium width;
  - its top 10-25% lower than the brow.
- **Widest at the cheeks or the back of the skull, never at the snout.** SyncCraft flags a snout wider than 0.88 of the skull as a "doghouse" head.
- **Size:** a large monster's head is 15-25% of its body length without the tail. Never shrink the head to make room for detail.

### 6.2 Depth: no flat faces
A face is flat when the front and three-quarter views show one plane. Build at least three depth steps:
1. **Brow ridge** per side, overhanging the eye by 0.5-1 unit and tilted 10-20 degrees (inward for predators). Its shadow gives the expression.
2. **Cheek (zygomatic) box** per side behind the eye, widening the head.
3. **Muzzle**, stepped down and narrower, pitched down 5-15 degrees, with a nasal ridge and a nose.

Then the **jaw**: a separate bone, narrower than the upper jaw, deep at the back, its top meeting the tooth line. Add a chin if the reference has one.

**Tests:**
- *Front view:* the head outline is a wedge or shield (wide at the cheeks and top, narrower at the muzzle), never a rectangle.
- *Side view:* the profile shows the brow step, the muzzle top, the nose, a clear mouth line and the jaw angle.

### 6.3 Eyes
- **Where:**
  - on the side faces of the cranium, where it meets the muzzle (the front quarter to third of the cranium);
  - just under the brow;
  - about 55-65% of the way up from the bottom of the jaw.
- **Facing:** predators' eyes angle forward 15-30 degrees and show in the three-quarter front view and partly in the front view, under the brow. Herbivores' eyes sit more to the side.
- **How:** a painted stamp, mirrored.
  - A dark socket of about 4 x 3 px at 32 px per block.
  - Iris 2 x 2 or 2 x 1.
  - A 1 px pupil or slit and a 1 px glint.
  - Glowing eyes bleed 1-2 px into the socket.
  - Only bulging eyes get a shallow box.
- **Wrong:**
  - on the top face or on the muzzle;
  - centred at mid-height on the side of the skull;
  - left and right at different heights;
  - a cube stuck on the surface;
  - a cluster of equal eyes.
- **Many-eyed creatures:** two main readable eyes; the rest are smaller painted pairs.

### 6.4 Horns, crests and ears on the head
- Roots sit at the back-top of the cranium or on the brow, overlapping the skull. Directions follow the reference; construction follows 7.1.
- Crest plates fan from the back of the skull over the first neck segment. This is what joins head to neck.

### 6.5 Mouth
- **Teeth:** 2-4 fangs as boxes. The rest are a painted row with dark gaps over a dark gum line.
- **Rest pose:** the mouth is slightly open if the reference shows it.
- **The jaw stays.** Only creatures smaller than the player that never open their mouth may merge it into the head.

### 6.6 Head gate (end of W1 and every later pass)
Render the head alone at 9 px per unit or more: side, front and three-quarter front. Compare with the head close-ups.

**It passes only when:**
- the proportions are in range (6.1);
- there are at least three depth steps;
- the eyes are placed per 6.3;
- the horns read as real blocks;
- the jaw is present and meets the upper jaw;
- a viewer could name the species from the head alone.

Otherwise stop and rebuild the head before anything else.

## 7. Horns, spikes and plates

### 7.1 Horns
- **Segments:** 2-4. Use 5 only for a signature horn longer than the head.
- **Root thickness:** at least one fifth of the cranium width, and at least 2 x 2 units on large creatures.
- **Taper:** each segment 70-85% of the previous one, and at least twice as long as it is thick.
- **Curve:** bend 10-25 degrees per segment in one consistent direction, so the curve reads from both front and side. Seat each segment 25% into the previous one.
- **Pairs and secondaries:** pairs are mirrored. Secondary horns are shorter and a different length from the main pair.
- **Tip:** the last segment may be a small box. Paint the light tip on the last segment only, so the horn doesn't look striped.

### 7.2 Spikes
- **One box each** (3.2).
- **Rows of 5-12:** graded, largest at the focal point, leaning back 10-30 degrees with the row.
- **Clusters of thin spikes** are cut-out planes.

### 7.3 Plates and armour
- **Rows:** plates follow the body's flow lines, and each covers 30-50% of the next. A row shares one lean.
- **Size:** largest over the shoulders, smaller toward head and tail.
- **Underside scutes:** wide (nearly the body's width), sunk so about 1 unit protrudes, parallel to the body.
- **Only plates that change the silhouette are geometry.** The rest are paint.

## 8. Body parts

### 8.0 Choose the process by body type
| Type | Wing or front limb | Legs | Build order |
|---|---|---|---|
| Six-limbed dragon | wings rooted on their own shoulder mass above the forelegs | four legs: fore pair straighter, hind pair zig-zag | torso masses, legs, then wings so they clear the legs |
| Wyvern (wing-walker) | the wing arm is the foreleg: thick humerus and forearm, a wrist claw that touches the ground, fingers folded back along the forearm | two heavy hind legs with a large thigh, crouched | hind legs and tail first for balance, then wing arms |
| Upright elder or biped | wings on the back; long arms with clawed hands | long digitigrade legs, three long toes | spine curve, legs, arms, wings |
| Feathered or bird-like | arm hidden under three rows of feather planes | thin tarsus, big thigh fluff, splayed toes | body, then feather rows from inner to outer |
| Insect or many-legged | thin plane wings with a vein texture, or wing cases | femur up and out, tibia down, small foot | body segments, then legs in mirrored pairs |
| Machine | panels, rotors or none | pistons and linkages (8.7) | chassis, joints, then limbs |

**Starting proportions in head-lengths (head : neck : torso : tail):**

| Body plan | Ratio | Notes |
|---|---|---|
| Western dragon | 1 : 2 : 3 : 4 | chest deep enough for two shoulder girdles |
| Wyvern | 1 : 1.5 : 2.5 : 3.5 | big chest, small hips |
| Drake or fanged beast | 1 : 0.8 : 3 : 2.5 | heavy forequarters |
| Brute | 1.2 : 0.6 : 2.5 : 2.5 | thick counterweight tail |
| Eastern dragon | 1 : body 10-14 in one long S | the body is the neck |
| Serpent or leviathan | 1 : body 8-12 | the read comes from the curve and the fin rhythm |
| Elder or boss | add 20% to the head and the signature feature | |

### 8.1 Neck
- 2-5 segments in an S-curve made with pitch. Each segment is slightly smaller than the last toward the head.
- The base is nearly as deep as the front of the chest.
- Add throat plates and a dorsal row that continues from the back.

### 8.2 Torso
- **Three masses:** chest (deepest), waist (narrowest) and hips, pitched a few degrees relative to each other so the back line arches.
- **Shoulder and haunch boxes** on the flanks; limbs attach to these.
- A smaller box on the chest for withers or a hump, so the top edge isn't a straight rectangle.
- **Connection check:** in side view the outline from skull to tail tip is one flowing line, never a row of crates.

### 8.3 Limbs
- **Choose the stance from the reference:** pillar (heavy), cursorial zig-zag (most dragons' hind legs), crouched biped (wyverns) or sprawled (lizards, insects).
- **Segments:** thigh or shoulder mass, then upper segment, lower segment and foot.
  - Thickness steps down about 30% per segment.
  - The foot widens again, to about 1.3x the lower leg.
- **Knee or elbow box, calf bulge and heel block.** These three boxes per leg turn a post into a leg.
- **Toes:** 3 boxes, splayed 5-15 degrees. Claws are single boxes pitched down 15-20 degrees, the middle one longest.
- **Grounding:** toes rest on the ground (y = 0). Splay legs at the knee, with the opposite roll on the foot, not at the hip.

### 8.4 Wings
- **Skeleton:** root mass, upper arm, forearm, wrist, then 3-4 fingers.
  - Spars 2-4 units thick, tapering only in their last third.
  - Knuckles on the two longest fingers.
  - A thumb claw at the wrist.
- **Membrane partition:** one cut-out panel per bone, parented to the spar on its leading side.
  - Cut each panel in alpha to its bay polygon: wrist, tip of the leading finger, tip of the next finger.
  - Scallop and tear only the trailing edge; at most two holes in the outer panels.
- **Panel heights** (in each spar's frame):
  - inside the panel's own spar's thickness, outside the neighbour's;
  - consecutive panels differ by at least 0.125 (use `layer: k`).
- **Folding:** fold with yaw at the elbow, wrist and fingers; flap with roll at the shoulder. Test the folded pose from above before texturing.
- **Span:** each wing about one body length (nose to hip).
- **Poses:**
  - *raised display* for turnarounds: upper arm up and back, forearm up and forward, fingers fanned down and out;
  - *folded at rest*: fingers closed and pointing to the tail, the wing held against the flank;
  - *wing-walk*: wrist on the ground, panels stacked along the forearm.

  The folded and wing-walk poses are mirrored on both sides.
- **Feathered wings:** arm boxes under 3 rows of feather planes, with 3-5 separate primaries at the tip.

### 8.5 Tail
- 5-7 segments tapering to 20-30% of the base thickness. The base is nearly as thick as the hips.
- Continue the gesture with pitch, and lift or curl the tip.
- Carry the dorsal row down the tail, shrinking.
- End with **one** signature feature: a blade, club, fin, tuft or stinger.

### 8.6 Arms and hands
- **Chain:** shoulder, upper arm, forearm, palm, then three fingers and a thumb. Each finger is one box plus a claw.
- **Rest pose:** elbows bent and fingers half curled. A straight arm with flat fingers looks like a toy.

### 8.7 Machines, vehicles and buildings
- **Volumes:** start from two or three big volumes, then chamfer the corners that show in silhouette with rotated boxes.
- **Greebles:** every one must look like it does something: hinge, vent, lamp, rail, pipe.
- **Joints and wheels:**
  - joints are octagon hubs with a painted bolt, larger than the limb they join;
  - wheels are an octagon tyre with tread pixels, a rim and a hub;
  - a piston is a rod sliding in a thicker sleeve;
  - a hose is 4-7 seated segments, not dozens.
- **Decals tell the story:** numbers, hazard stripes, patched panels.
- **Weathering follows gravity and use:** streaks under seams, wear on edges, a top-surface layer (snow, moss, dust).

### 8.8 Cut-out planes
- Paint the edge shape in alpha with a stepped pixel outline; no straight plane borders should show.
- Planes are always double-sided, and never coplanar with a box face or another plane (use `layer`).
- **Fringes:** run along the back of the neck, the jaw line, the elbows and the tail, with spikes 1-3 px wide pointing away from the direction of travel.
- **Hanging things** (moss, cloth, roots, icicles): drips of uneven length.

## 9. Creature classes, scale and categories

### 9.1 Scale tiers
| Subject | Treatment | Element ceiling (guide) |
|---|---|---|
| Elder dragons and boss monsters, any size | hero or elder, epic and deliberate | 200-450 |
| Large monsters (bigger than the player) | hero | 150-350 |
| Player-sized monsters and mounts | standard to hero | 80-220 |
| Smaller than the player (fish, birds, insects, critters) | simplified, but still clean and banded | 15-80 |
| Gathering points and props | prop | 8-60 |

**Texel density:** 32 px per block for hero and elder creatures, 16 for standard and small ones. Keep one density per asset.

**Community scale anchor:** a popular animated community fire dragon is about 415 cubes, 8 blocks tall, at 16 px per block. That is a practical ceiling for a hero creature that still animates smoothly, not a target.

### 9.2 Monster Hunter classes as body plans
| Class | Body plan and construction notes | Examples |
|---|---|---|
| Flying wyvern | 2 legs, wing-arms; deep chest, small hips; wings 1.3-1.5x nose-to-tail span; tail tip is the signature | Rathalos, Rathian |
| Elder dragon | often six limbs (4 legs + wings) or a unique plan; horns that sweep into the dorsal line; one strong signature | Fatalis, Kushala Daora, Velkhana, Valstrax, Alatreon, Kirin |
| Fanged wyvern | wolf-like quadruped; heavy forequarters; short deep muzzle (6.1); fur as cut-out rows; one glow accent | Zinogre |
| Leviathan | long body in an S made with pitch, low limbs, fins and frills as partitioned planes | Lagiacrus, Mizutsune |
| Brute wyvern | biped, oversized head and jaw, thick counterweight tail | Anjanath, Deviljho |
| Bird wyvern | light biped, beak or crest; often player-sized | Yian Kut-Ku |
| Fanged beast | mammal-like, fur rows, expressive face | Rajang |
| Temnoceran | spider-like; see 9.5 | Nerscylla |
| Cephalopod | tentacled; see 9.5 | Nu Udra, Xu Wu |

Check each game's current roster on the web, and file each monster under its class from an official source.

### 9.3 Environment creatures
- **Smaller than the player** (fish, birds, small herbivores, insects): simplified per 9.1. The silhouette, symmetry, banded paint and a readable eye still apply.
- **Herbivores larger than the player** follow the standard tier.

### 9.4 Gathering points
- **Rocks and ore outcrops:** 3-7 large rotated boxes, never voxel boulders.
- **Crystals:** 3-7 single-box prisms of mixed height.
- **Plants and mushrooms:** cut-out planes or chunky caps (one box each).
- **Bones:** a few large box bones.
- **Marker:** a small focal glow or contrasting accent that shows the point is interactable.

### 9.5 Unsettling creatures (spiders, cephalopods, parasites): faithful, not repulsive
- **Keep the identity:** body plan, limb count, signature shapes.
- **Clean the presentation:**
  - Limbs and tentacles are thicker, tapered 3-4 segment chains with a steady rhythm. No tangles, no hair-thin legs.
  - A slightly larger body and head relative to the legs.
  - Two clear main eyes with highlights; any extra eyes small and painted.
  - Mouthparts simplified into 2-3 shapes; no gore, no detailed mouth interior.
  - A graphic, saturated palette instead of flesh tones; no slime, veins or wet noise.
- **The aim:** not cute, not disgusting. Imposing and readable.

## 10. Texture

### 10.1 Workflow
1. Flat base colour per material from the palette.
2. Banded light: top faces lightest, sides middle, undersides darkest. A 1 px light top edge and a 1 px dark contact edge.
3. Material strokes in clusters, on focal zones and edges only.
4. Features: eyes, mouth, markings, glow with a 1-3 px bleed.
5. Squint test at 25% size.

### 10.2 Clean, not dirty
- **Calm area:** 85-90% of each surface is calm mid-tone.
- **Scales and grain:** at most one value step of contrast, in regular offset rows.
- **Markings:** large graphic shapes that follow the form, such as dorsal chevrons or stripes along the long axis.
- **Colour budget:** at most 5 colours per material; 3 + 1 per creature (body, underside, hard parts, one accent).
- **Never:** per-pixel noise, black outlines, grey shadows, or stripes reset on every segment (candy-cane horns).
- **Glow:** at most 4% of the surface, near the focal point.
- **Paint follows geometry:** eyes go where 6.3 says, teeth on the tooth line, highlights on top faces.
- **Second texture pass:** it removes as much as it adds.

### 10.3 Paint by region
- **Head:** darker around the mouth and eye sockets; lighter on the brow ridge and snout top. The underjaw is in the underside colour, with a lip line one shade darker.
- **Torso:** countershaded (dark back, mid flank, light belly), with one soft band at each transition. Patterns follow the long axis and thin toward the belly.
- **Limbs:** darker toward the feet, a lighter kneecap highlight. Bands are at most one dark row in five.
- **Claws, horns, teeth:** a light top edge and a dark base. The lighter tip goes on the last segment only.
- **Membranes:** the underside colour as a stepped gradient from spar to trailing edge.

### 10.4 Presentation
- **Turnarounds:** a dark neutral backdrop.
- **Hero shots:** a simple scene, a key light with a coloured rim, a little bloom on emissive parts, and a cast shadow.

## 11. SyncCraft and Block-DSL v4.1 (the harness)

### 11.1 Specifications
Read `docs/BLOCK_DSL_V3_2_SPECIFICATION.md` (base constructs) and `docs/BLOCK_DSL_V4_SPECIFICATION.md` (v4.0 / v4.1 additions). Use SyncCraft for every model. The audits are your error correction: the SE(3) chain, SAT collisions, connectivity and grounding.

### 11.2 Meta and layout
- `@meta { dsl_version: "4.1", audit_profile: sculpted, size_step: 0.5, units_per_block: 16, texel_density: 32 }`.
- `@layout { budget max_boxes = ..., envelope grounded = true }`. Budgets are ceilings.

### 11.3 Constructs
- `attach(..., seat=0.28)` for chains.
- `mirror_x(name)` / `mirror(axis=x)` for symmetry; `mirror: true` on stamps.
- `chain <name> attach(...) segments=N taper=... seat=... bend=[...]` for necks, tails and horns, with N within the remap limits.
- `plane` with `layer: k` (0.25 units per step) and `edge: ragged|spikes|drip|scallop|triangle_bay`.
- `blend_with: [...]` for cross-bone seams.
- `octagon` for hard-surface round parts. `prism(n)` never on organic parts.
- **Plane `size` order:**
  - `normal: x` is `[height, length]`;
  - `normal: y` is `[width, length]`;
  - `normal: z` is `[width, height]`.

  A wrong order still passes the audit.

### 11.4 Surface
- `@style { shade: banded(steps=4, direction=down, cluster=diagonal), glow_bleed: 2 }`.
- Coats: `scale_rows`, `banded_limb`, `plain`, `striped`, `spotted`.
- `@stamps` for eyes and teeth.
- Paint by hand or by script on the exported UV layout when the bake can't express the look, and record that you did.

### 11.5 Checks
- `audit_scene(scene, audit_profile="sculpted")`: collisions, coplanar faces, contact and burial, budget, grounding.
- The construction check (3.6).
- `analyze_morphology(program)`: limbs, posture, `symmetry_pairs`.
- `audit_behavior_clip(...)`: joint limits and swept motion.
- `evaluate_sculpted_style_anchor(program, scene, atlas_rgba=...)`.

### 11.6 Stage files and renders
`scripts/inspect_mh_sculpt.py <id> --stage 1|2|3 --dsl-root <your dsl dir>` renders stage files laid out as `dsl/<id>/stage1_blockout.blockdsl` (W passes), `stage2_secondary.blockdsl` (D passes) and `<id>.blockdsl` (final). Keep every pass as `passes/<id>_<pass>.blockdsl`.

### 11.7 Scores are diagnostics, not targets
The style evaluator rewards a high share of rotated boxes, and the dataset tests demand at least 260 elements. Chasing either produces shard armour. Treat both as diagnostics, never as targets.

### 11.8 Gotchas
- **Re-ground** after every pose change, so the lowest toe sits at y = 0.
- **The audit tests a plane's whole rectangle,** not its alpha shape.
- **Every size** is a multiple of `size_step`.

## 12. Subagent brief (paste into every subagent call)

```
You are working on <model> in the sculpted Blockbench style.
Before any visual reasoning or construction, and again at the start of every pass:
  1. read SKILL.md (blockbench-sculpted-style) in full;
  2. open every image in style_anchors/;
  3. open this model's reference board and its zoomed crops;
  4. read PROJECT_GOAL.md (owner corrections included).
Your pass: <W1|W2|W3|D1|D2|T1|T2|A1>. Inputs: <files>. Outputs: <files>, renders, region sheet, pass log.
Hard rules - Section 1 of the skill, all 15, apply without exception.
Use Block-DSL v4.1 through SyncCraft; run audit_scene and the construction check (3.6);
render the view set and region sheet; for each region write "reference shows / model shows / change".
If any punch-back test (5.4) fails, stop and report "PUNCH BACK TO W1: <reason>". Do not patch with detail.
Report: changes, check results, region comparisons, open issues. Say plainly if you could not see an image.
```

## 13. Review rubric (every pass, by a reviewer who did not build it)
Score each item 0, 1 or 2:
1. silhouette
2. head (6.6)
3. eyes
4. horns and spikes as real blocks
5. construction (no shards, no fake curves)
6. connection
7. symmetry
8. proportions against the reference
9. volume hierarchy
10. calm, clean paint
11. vibrancy and focal point
12. fidelity to the reference's identity traits
13. animation readiness

**Pass:** no zeros and at least 20 of 26. Any zero on items 1, 2, 5 or 7 means punch back to W1.

## 14. Lessons from test flights
- **Measure proportions before detailing.** On a standard dragon the head is at least a third of the torso length, and the upper leg at least half the torso's depth.
- **Check every joint close up.** A jaw that floats 1.5 units below the snout can still pass the audit.
- **Colour the lower jaw like the underside.** A cream jaw under a tooth row reads at once.
- **Seat segments about 28% into their parents.** Shingle the joints and fan crest plates from the skull over the neck; the row of boxes becomes one form.
- **Partitioned wings fold cleanly.** Size each panel to its bay polygon and regenerate it whenever a bone changes.
- **The Squallwing run.**
  - What moved it most: a wedge snout (down 5 degrees, nasal ridge down 12), jaw open 14 degrees, horns swept back 52 degrees and splayed 34, dorsal fins graded largest at the withers, a forked tail with flat streamers.
  - Texture: low-contrast brick scale rows, dorsal-only chevrons, an eye in a dark socket under the brow.
  - Its mistake: attitude in the rest pose.
- **Repo v9 diagnosis.** Visual reasoning was fine, but construction shattered every form. Most of the owner's complaints trace to that one cause: dirty, ugly, a flat long face, eyes in odd places, sheet horns, broken symmetry.

## 15. References, originality and licensing
- **Monster Hunter monsters are Capcom's IP.** Fan reconstructions are for non-commercial research only. Tag them like the repo's `research_nc`, keep them out of commercial weights, and never copy textures from screenshots.
- **For commercial work,** design originals:
  1. one concept line (habitat + temperament + role);
  2. 2-3 animal anchors for anatomy;
  3. one shape language (blades: fast and aggressive; slabs: heavy and ancient; curves: aquatic or young);
  4. one signature feature at 1.5x, plus at most two supporting features;
  5. a 3 + 1 palette;
  6. a plausibility pass: muscles, centre of mass, how the wings fold, how it eats.

  Change the body plan, shape language and signature feature, not just the colours.

  **Theme motifs:**

  | Theme | Forms | Surface | Glow |
  |---|---|---|---|
  | Fire | cracked plates, swept horns, vents | charcoal and deep red, bone horns | orange in cracks and throat |
  | Ice | crystal clusters, blades, fur ruffs | pale blue and white, navy shadow | cyan in crystals and eyes |
  | Storm | fin crests, forked horns, whip tail | slate blue with gold | electric blue along the dorsal row |
  | Poison | sacs, frills, barbels | olive, teal, yellow belly | acid green |
  | Earth | slab armour, club tail, thick horns | ochre and grey | amber in seams, or none |
  | Water | fins, whiskers, smooth body | deep blue to pale belly | bioluminescent rows |
  | Shadow | torn membranes, thin limbs | near-black purple | magenta or violet |
- **Credit** community artists and Cobblemon (CC-BY-NC) where their work informed a model.

## 16. Deliverables per asset
1. The reference board and part map.
2. Pass files W1-W3, D1-D2, T1-T2 and A1, each with renders, a region sheet, check results and a pass log.
3. The final Block-DSL source, the texture atlas, the `.bbmodel` export, the audit report and the rubric score.
4. A short note on what was simplified or exaggerated, and why.