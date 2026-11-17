---
name: "blockbench-sculpted-style"
description: "Use when designing, critiquing or generating Blockbench/Minecraft-style 3D models (creatures, dragons, vehicles, props, scenes) in the sculpted cuboid, hand-painted pixel-texture style, including Block-DSL test and dataset assets."
---

# Sculpted Blockbench style

The target look is the one used by high-end Blockbench artists for large creatures and cosy dioramas. It is built from cuboids and flat planes, but it is **not voxel art** and **not vanilla Minecraft minimalism**. Think of it as sculpture with boxes: capture what makes the subject recognisable, simplify the rest, and paint light and material into a pixel texture.

## 1. Style traits (the definition of done)

**Form**
- Cuboids are freely rotated and overlapped to make slopes, tapers and curves. Nothing important sits on a rigid grid.
- Silhouette first. Every part must read as a black shape from front, side, three-quarter and top.
- Clear size hierarchy: a few big masses, a handful of medium forms, small accents. Roughly 70 / 25 / 5 by volume.
- Proportions are pushed: larger heads, hands, horns, claws and props than real life; thick-to-thin tapers on limbs, tails, horns.
- Thin things are flat planes or half-unit slabs: leaves, feathers, membranes, pages, cloth, fins, hair tufts. Tilt and layer them; never leave them coplanar.
- Deliberate imperfection: books lean, planks are slightly skewed, spikes vary in size and angle. Perfect symmetry and equal spacing look machine-made.
- Curves are 3+ segments with a steady angle step (10-25 degrees each), tapering as they go.
- Dense, purposeful clusters of small props or spikes around focal areas; calm, larger surfaces elsewhere.

**Texture**
- Hand-painted pixel art at one consistent texel density per asset (16 px per block for chibi and standard creatures; 32 for hero and elder tiers).
- Shading is banded: 3-5 distinct value steps per colour, placed as bands and stepped diagonal clusters, not as smooth blends or noise.
- Transparent cut-out edges on planes are a core tool, not an extra.
- Light is painted in: top faces lightest, sides mid, undersides darkest, plus a 1 px lighter top edge and 1 px darker bottom or contact edge on most boxes.
- Hue-shifted ramps of 4-6 steps: highlights drift warm (toward yellow), shadows drift cool (toward blue or purple) and slightly more saturated. Never shade by adding black or white only.
- Clusters and strokes, not noise: wood grain as broken lines, scales as offset rows, stone as a few chips, fabric as folds.
- Vibrant, limited palette: one dominant hue family, one supporting, one accent near its complement (for example warm orange glow against violet and teal). Saturation is highest in mid-tones.
- Glow is a material: emissive eyes, embers, windows, crystals, with their colour bleeding 1-3 px onto neighbours.
- Features are placed by hand: eyes with a highlight pixel, nostrils, teeth gaps, scars, markings, stitches.

**Not this style**
- Stair-stepped voxel volumes; axis-aligned box robots; uniform limb thickness.
- Flat single-colour faces, random per-pixel noise, grey shadows, pure black outlines everywhere.
- Smooth subdivided meshes or photo textures.
- Detail spread evenly over the whole model.

## 2. Workflow

Work like a sculptor: reference, analysis, big to small, with a visual check at every stage. Do not skip to detail.

### A. Gather reference
Search for and collect, per subject: official art or renders, in-game or turnaround views (front, side, back, top, three-quarter), community fan models, and any block or low-poly interpretations. Prefer at least five images covering different angles. Note which traits appear in every source; those are identity traits.

### B. Write a trait sheet (reference document, with the images)
Go through the body in order and write two to five short lines per region. Use a head-length or block as the unit.

1. **Whole**: gesture line (the single curve from nose to tail tip), stance, centre of mass, overall length:height:width, the three things anyone would recognise.
2. **Head**: skull block shape, snout length and taper, jaw, eye position and size, horns or crests (count, direction, curve), teeth, frills, ears.
3. **Neck and torso**: neck length and S-curve, chest depth versus waist, shoulder and hip mass, back line, dorsal features, belly plates.
4. **Wings or arms**: attachment point, segment lengths as ratios, fold pattern, finger count, membrane or feather layering, span versus body length.
5. **Limbs**: segment count and angles (plantigrade, digitigrade, sprawled), thigh mass, foot shape, claw count and size.
6. **Tail**: length versus body, taper, curve, tip feature.
7. **Surface**: materials per region, palette (name 3-5 colours with roles), pattern type, glow, wear.

For each trait mark one of: **keep** (identity), **exaggerate** (make it 1.2-1.5x), **merge** (several small things become one form), **paint** (texture only, no geometry), **drop**.
Rule of thumb: geometry for anything that changes the silhouette; texture for everything else.

### C. Blockout (primary masses)
8-20 boxes total. One box per major mass: skull, snout, neck segments, chest, belly, hips, each limb segment, tail segments, wing spars. Set rotations now, including the gesture curve. Check silhouettes from four views against the trait sheet. Fix proportions here; later stages cannot rescue them.

### D. Secondary forms
Add what changes the outline at medium scale: brow, jaw, cheek, shoulder and haunch masses, knees and elbows, horn segments, dorsal row, wing fingers, feet. Taper by stepping size down each segment. Break long straight edges with a second, slightly rotated box.

### E. Tertiary detail and planes
Claws, teeth, spikes, plates, membranes, tufts, straps, props. Vary size and angle within a row (largest near the focal point, shrinking away). Layer planes at different heights and tilts. Stop when the focal areas are rich and the rest is calm.

### F. Texture
1. Flat base colours per material from the palette.
2. Painted light: top, side, underside values; edge highlight and contact shadow lines.
3. Material strokes in clusters.
4. Features: eyes, mouth, markings, glow and its colour bleed.
5. Squint test: the model should still read at a quarter size.

### G. Rig readiness
Pivots at anatomical joints; overlapping boxes at joints act as miters so bends do not open gaps; nothing essential breaks when a limb rotates 30 degrees.

### Visual checks (after every stage C-F)
- **Regional**: render or view each region close up (head, torso, wing, limb, tail). Compare with its trait-sheet lines. Name one thing that is wrong before moving on.
- **Whole**: front, side, three-quarter front, three-quarter back, top. Check gesture, balance, hierarchy, and that feet meet the ground.
- **Silhouette**: view as a flat dark shape. If the subject is not identifiable, go back to C.
Record each check with its image in the working document.

## What the owner's reference set shows

These notes come from nine images the owner supplied: five dragons (a red-and-cream western dragon, a black-and-blue upright elder dragon, a chunky blue ice dragon, a pink chibi dragon, a tan club-tailed wyvern) and four scenes (an artist's studio, a harvest robot in an orchard, a snow expedition truck, an airship among cliff houses). Where a rule elsewhere in this skill disagrees with these notes, these notes win.

### Shared signature
1. **Cuboids stay cuboids, but they are rotated and stacked into organic flow.** No stair-stepped voxel volumes and no smooth meshes on creatures. You can count the boxes, yet the outline reads as a curve.
2. **The silhouette edge is drawn in pixels.** Thin planes with transparent, hand-cut edges do the work geometry cannot: torn wing membranes with holes, spiky fur or spine fringes along the neck and limbs, dripping moss, cloth tatters, leaf clusters. This is the single most distinctive trait. Hard box edges alone look unfinished.
3. **Banded cel shading.** Three to five clearly separate value steps per colour, laid as bands and stepped diagonal clusters that follow the form. Shadows shift toward purple or blue, highlights toward cyan, yellow or cream. Almost no soft noise.
4. **Two-colour body scheme plus one accent.** Each dragon is a body colour and an underside or membrane colour (red and cream, blue and navy, black and violet-blue, tan and sand), with one hard-part colour for horns, claws and crystals and at most one glow accent.
5. **Segmented undersides are geometry.** Throat, chest and belly are a column of separate overlapping plate boxes, each a little wider or narrower than its neighbour, in the underside colour.
6. **Round things are octagons.** Wheels, joints, hubs, balloons, tree crowns and pots are built from two or three boxes rotated 45 degrees, or a low faceted prism. Never a single square, never a smooth cylinder.
7. **Saturated but controlled.** Large areas are mid-saturation; the strongest colour and brightest value sit at the focal point (eyes, crystals, lamps, fruit, headlights).
8. **Presentation is part of the piece.** Dark neutral backdrop for turnarounds; for hero shots a simple scene, a key light with coloured rim, soft bloom on emissive parts, and a cast shadow.

### What each dragon teaches
- **Red and cream western dragon (clean, mid detail):** wing spars are thick box chains in the body colour: upper arm, forearm, and three long fingers radiating from the wrist. The membrane is one large plane per bay, painted as a dithered gradient from cream at the spar to deeper tan at the body, with a ragged pixel-cut trailing edge and a few holes near the tip. Horns are pale, long, and change direction at each segment like antlers, with shorter secondary horns at the cheek. Legs carry horizontal bands of alternating body tones to suggest scale rows. Toes and claws are cream. The tail's cream underside continues the belly column to the tip.
- **Black and blue elder dragon (maximum detail):** hundreds of small boxes. Upright stance on long digitigrade legs with three long toes. The wing's leading edge is armoured with a row of overlapping rounded scale boxes. Membranes are large planes with a smooth pixel gradient from deep blue to violet and heavily torn edges with interior holes. Bright cyan crystal horns and shards are the only saturated accent and read from any distance. A pale chest panel breaks the dark mass. Long thin clawed fingers are separate boxes.
- **Blue ice dragon (chunky, expressive):** the head is about one third of the body mass, with a heavy block jaw, a row of large yellow teeth, and a single red eye pixel set deep under the brow. Scales are painted as stepped diagonal clusters in four blues from cyan highlight to purple shadow. Back crystals are tall white-to-cyan prisms rising from small brown rock bases. Neck and limb edges carry a fringe of pixel spikes. Belly plates are dark navy segments. Claws are long, thin and brown.
- **Pink chibi dragon (minimal):** about 25 boxes. Head nearly half the body, stub legs, tiny wings, three short horns, one large dark eye with a highlight. Pastel palette, soft glow. Proof that the style scales down: fewer boxes, same banded shading and same clear silhouette.
- **Tan club-tailed wyvern (dynamic pose):** crouched, wing-arms planted, tail raised. Thick wing arms, sand membranes with ragged edges, a row of curved yellow dorsal spikes made as small pixel-cut planes, a mace tail studded with white spikes, grey segmented ram horns, a glowing eye. Shows posing, rim light and a strong signature feature.

### What the scenes teach
- **Studio interior:** warm emissive light sources against cool violet ambient; dense prop clusters (books, bottles, candles) with slight tilts; plants as layered leaf planes; a clear focal object lit brightest.
- **Harvest robot:** hard-surface forms get chamfers from extra rotated boxes; joints and wheels are octagons with a painted bolt; hoses are chains of many short segments following a curve; foliage is large rotated cubes with leaf-cluster textures and cut-out edges; palette is three hues (yellow, green, sky blue) at high saturation.
- **Expedition truck:** silhouette greebles from thin boxes (rails, ladders, antennas, lamps); painted weathering as vertical pixel streaks and chipped edges; pixel-font lettering and hazard stripes as decals; emissive white lamps; a second material layered on top surfaces (snow caps as white slabs). Muted base, small orange and blue accents.
- **Airship and cliff houses:** large faceted volumes with bold diagonal pixel patterns; a teal, cream and terracotta triad; hanging cloth and moss as drippy cut-out planes; steep roofs as rotated slabs; rock faces painted with angular highlight shapes rather than noise.

### Detail tiers (pick one before starting)
| Tier | Boxes | Texel density | Head share | Use |
|---|---|---|---|---|
| Chibi | 15-40 | 16 px per block | 40-50% | pets, icons, small mobs |
| Standard | 80-200 | 16 px per block | 20-30% | most mobs and mounts |
| Hero | 200-450 | 16-32 px per block | 15-25% | bosses, showcase creatures |
| Elder | 450-900 | 32 px per block | 10-18% | flagship monsters |

### Cut-out planes: how to use them
- Paint the edge shape in the texture's alpha, with a jagged or stepped pixel outline; avoid straight plane borders showing.
- Membranes: one plane per wing bay, edge scalloped between spars, 2-6 holes clustered near the outer trailing edge, more damage on older or fiercer creatures.
- Fringes: narrow planes along the back of the neck, jaw line, elbows and tail, spikes 1-3 px wide, pointing away from the direction of travel.
- Hanging and growing things: moss, cloth, roots, icicles as vertical planes with dripping edges of uneven length.
- Always double-sided, and never coplanar with a box face.

### Vehicles, machines and buildings
- Start from two or three big volumes, then chamfer corners with rotated boxes.
- Add function-driven detail: every greeble should look like it does something (hinge, vent, lamp, rail, pipe).
- Wheels: octagon rim, darker octagon tyre with tread pixels, hub with a bolt pattern.
- Decals carry story: numbers, names, warning stripes, patched panels.
- Weathering follows gravity and use: streaks under seams, wear on edges and steps, dirt near the ground, a top-surface layer (snow, moss, dust).

## Body-part guide for entities

Sizes are in head-lengths (HL) unless stated. Box counts are for a large creature; halve them for small animals. Each part lists structure first, then what to paint.

### Head (spend the most here)
- **Skull**: one main block, slightly wider at the back than the front. Add a brow box per side that overhangs the eye by half a unit; the shadow under it gives expression. Tilt brows inward 5-10 degrees for a predator, outward for a gentle look.
- **Snout**: one or two boxes, narrower and lower than the skull, stepping down toward the nose. Rotate the top box down 5-15 degrees so the profile slopes. Add a nose or nostril bump at the tip.
- **Jaw**: separate bone hinged at the back-bottom of the skull. Make it narrower than the upper jaw so upper teeth overlap it. Add a chin box; a jaw with depth reads stronger than a flat plate.
- **Teeth**: do not model every tooth. Two to four large fangs as geometry (canines, tusks); the rest as a painted row with dark gaps. Fangs angle slightly outward and back.
- **Eyes**: usually painted, set in the shadowed zone under the brow, 2x1 or 2x2 px with one highlight pixel and a dark upper lid line. Glowing eyes bleed 1-2 px of colour onto the socket. Model an eye bump only for frogs, chameleons, crocodiles and insects.
- **Cheeks and jaw muscles**: one box per side behind the mouth corner widens the head and anchors frills or tufts.
- **Ears, frills, fins**: planes or half-unit slabs on their own bones, angled back and outward 20-45 degrees, never flat against the head.
- **Horns**: 3-5 tapering segments, each rotated 10-25 degrees from the last, so the curve is visible in side and front views. Base overlaps the skull. Pair a main horn with one or two smaller ones of a different length; identical horns look generated.
- **Paint**: darker around the mouth and eye sockets, lighter on the brow ridge and snout top, a lighter underjaw, lip line one shade darker.

### Neck
- 2-5 segments. Give it an S-curve: first segments rise, last segment and head pitch back down. A straight neck is the most common cause of a toy-like look.
- Each segment slightly smaller than the last toward the head; the base is nearly as deep as the chest front.
- Throat plates or a lighter underside strip, and a dorsal row (spikes, mane, fin) that continues from the back.

### Torso
- Three masses, not one box: **chest** (deepest and widest), **waist** (narrowest), **hips**. Rotate chest and hips a few degrees relative to each other so the back line arches or dips.
- Add shoulder and haunch masses as separate boxes on the flanks; limbs attach to these, not to a flat side. They are the main cue that the creature has muscles.
- Belly: a lighter, slightly narrower strip of plates below. Back: a dorsal ridge that peaks over the shoulders.
- Break the top edge. Stack a smaller rotated box on the chest for withers or a hump so the silhouette is not a rectangle.
- **Paint**: countershade (dark back, mid flank, light belly) with a soft band at each transition; pattern (stripes, spots, scale rows) follows the body's long axis and thins toward the belly.

### Limbs
- Three segments plus foot, each with a different length, thickness and angle. Zig-zag in side view: digitigrade hind leg goes forward-back-forward; a fore leg goes back-forward.
- **Upper segment** is the thickest and partly merges into the shoulder or haunch mass. **Lower segment** is 60-75% of its thickness. Add a small box for the knee or elbow point.
- **Feet** are larger than realistic: about 1.3x the lower leg's width. Toes as 2-4 separate boxes that splay 5-15 degrees; claws as tapered boxes pitched down, with the middle claw longest.
- Feet rest flat on the ground plane; heels may lift on digitigrade legs but toes do not float.
- Forelimbs that grasp: the hand is a palm box plus 3 fingers and a thumb, each a single box with a claw.
- **Paint**: darker toward the feet, lighter kneecap highlight, claws with a light top edge and dark base.

### Wings
- **Membrane wings**: shoulder, forearm, wrist, then 3-4 finger spars fanned from the wrist. Spars are thin tapered boxes; membranes are planes between them, each at its own height so they layer. The trailing edge should scallop: let each membrane end shorter between spars.
- **Feathered wings**: arm boxes hidden under 3 rows of plane feathers (coverts, secondaries, primaries), each row longer and darker, with 3-5 separate primaries splayed at the tip.
- Folded pose: forearm folds back over the upper arm, fingers gather and point to the tail; keep it for ground poses to avoid a wall of membrane.
- Span for a believable flyer is 2-3x body length, wing root is at least as thick as the upper foreleg, and a thumb claw at the wrist gives scale.
- **Paint**: spars take the body colour and are clearly thicker than a pure skeleton would be. The membrane takes the underside colour as a stepped or dithered pixel gradient running from spar to trailing edge. Cut the trailing edge ragged in the texture's alpha and add holes near the outer tip; see "Cut-out planes".

### Tail
- 5-9 segments tapering to 20-30% of the base thickness. Continue the body's gesture curve; lift or curl the tip.
- Base is nearly as thick as the hips so the body flows into it.
- Carry the dorsal row down it, shrinking, then end with one signature feature (blade, club, fin, tuft, stinger, crystal cluster).
- Length 1-1.5x the torso for balance on bipeds and long-necked forms.

### Surface add-ons
- **Spikes and plates**: rows of 5-12, largest at the shoulders or skull, shrinking both ways; alternate two sizes; lean them back 10-30 degrees.
- **Fur and manes**: clumps as tilted slabs or planes in 2-3 layers, tips lighter; never a single fuzzy block.
- **Armour and scales**: large plates as geometry only where they overlap an edge; everything else as painted scale rows.
- **Crystals, embers, growths**: clusters of 3-7 rotated prisms of mixed height, emissive, with glow bleed painted on the surface beneath.

## Connected bodies: overlap, seating and shingles

A creature must read as one continuous form. Boxes that merely touch face to face look like a train of crates. The references solve this with deliberate overlap, and the two densest ones (a tan-and-teal plated beast and a quill-covered beast) are built almost entirely from overlapping slabs.

### Seat every segment
- Push each neck, torso and tail segment 25-30% of its length into its parent. In Block-DSL set the attach anchor to about 0.72 of the face distance instead of 1.
- Seat limbs one to two units into the shoulder or haunch mass, and the head into the last neck segment.
- After seating, re-check ground contact and total length; proportions shrink a little.

### Cover every joint
- Lay a pair of plates over the top corners of each segment, tilted outward 15-20 degrees and pitched 8 degrees, reaching one to three units past the joint. They hide the seam and turn when the joint turns.
- From the back of the skull, fan three crest plates over the first neck segment (one centre, two angled out 20-25 degrees). This is what connects head to neck.
- Use the same idea at shoulder, hip and tail root.

### Shingle for armour and fur
- Lay plates like roof tiles: each covers 30-50% of the next, all leaning the same way (back and outward), in rows that follow the body's flow lines.
- Vary length and tilt slightly within a row; make plates largest over the shoulders and smaller toward head and tail.
- Dense tier: several rows deep, hundreds of slabs; standard tier: one row at joints plus a dorsal row.
- Quills and spines are the same technique with thin, long slabs fanned 10-15 degrees apart from a shared root area.
- Alternate two or three tones of the body colour plate by plate so each tile reads.

### Limits
- Every box must still show a real part of itself. If a plate is more than about two thirds buried, shrink or move it.
- In Block-DSL the sculpted profile accepts overlap only inside one bone or between a bone and its direct parent. A plate that reaches a grandparent or a sibling chain is rejected; shorten it or parent it differently. Two concentric boxes as an octagon are also rejected.
- Underside plates should be wide (nearly the body's width) and sunk so only about one unit protrudes; narrow protruding plates read as planks.

### Connection check
Render the side view and look only at the outline of the back and belly from skull to tail tip. It should be one flowing line with small steps, never a row of rectangles with notches between them. Then view each joint close up.

## Wings and limbs in depth

Blockbench and Minecraft animate rigid pieces only: bones rotate, move and scale; nothing bends. Every wing and limb must therefore be designed as a set of rigid parts that still look right when each part rotates. Decide the creature type first, because the process differs.

### Choose the process by type

| Type | Wing or front limb | Legs | Build order |
|---|---|---|---|
| Six-limbed dragon | separate wings rooted on a shoulder mass above the forelegs | four pillar legs, fore pair straighter, hind pair zig-zag | torso masses, legs, then wings last so they clear the legs |
| Wyvern (wing-walker) | the wing arm is the foreleg: thick humerus and forearm, wrist with a claw that touches the ground, fingers folded back along the forearm | two heavy hind legs with a large thigh, crouched | hind legs and tail first for balance, then wing arms posed as props |
| Upright elder or biped | wings on the back, often two layers; long arms with clawed hands | long digitigrade legs, three long toes | spine curve first, legs, arms, wings |
| Feathered or bird-like | arm hidden under three rows of feather planes | thin tarsus, big thigh fluff, splayed toes | body, feather rows inner to outer |
| Insect or many-legged | thin plane wings with a vein texture, or wing cases | femur up and out, tibia down, small foot | body segments, then legs in mirrored pairs |
| Machine | panels, rotors, or none | pistons and linkages; see below | chassis, joints, then limbs |

### Membrane wings: skeleton
- Four spar levels: **root mass** on the body, **upper arm**, **forearm**, then **fingers** from the wrist. The elbow is the high point when the wing is raised; the wrist is the outermost corner of the leading edge.
- Spars are thick: 2-4 units on a standard dragon, roughly the thickness of the lower leg. Thin sticks look weak and vanish at distance. Taper only the last third of each finger.
- Three fingers are enough at standard tier, four at hero tier. The first continues the leading edge; the last points back toward the hip. Spread 25-40 degrees between neighbours when open.
- Give the two longest fingers a knuckle (two bones each) so the tip can curl and the wing can fold tighter.
- Mark elbow and wrist with a slightly larger rotated box and add a thumb claw at the wrist.
- Open-wing size: each wing about one body length (nose to hip) from root to tip; total span two to three body lengths.

### Membrane wings: partition the membrane
A membrane cannot bend, so slice it along the bones.
- **One panel per bone.** Each finger bone carries the panel behind it; a two-bone finger carries two panels. The forearm and upper arm each carry one. A standard three-finger wing has seven panels per side.
- **Parent each panel to the spar on its leading side**, so the panel follows that spar when it rotates.
- **Shape each bay as a triangle fanning from the wrist**, cut in the texture's alpha: a panel is opaque only as deep as its distance from the wrist times the tangent of the gap to the next finger. The rectangle is the carrier; the alpha draws the sector.
- **Layer the panels.** Put neighbouring panels at slightly different heights (a quarter to half a unit) and let each overlap the next by 1-2 units. When fingers close, panels slide over one another like a hand fan and no gap opens. Panels at the same height flicker and are rejected by the audit.
- **Scallop and tear only the trailing edge.** Keep the edge along the next finger clean so the overlap hides it. Holes belong in the outer panels only, two at most per panel.
- **Test the fold before texturing.** Rotate the fingers and forearm to a folded pose and view from above: no gaps between panels, no panel poking through the body. In the second test flight this check passed visually and also revealed two spar collisions at the tightest fold, which set the joint limits.
- For large wings, split a long bay again across its length (inner and outer panel on the same bone) only if the wing must curl in animation.

### Membrane wings: poses
- **Raised display** (the reference red dragon): upper arm up and back, forearm up and forward, fingers fanned down and out. Best for turnarounds.
- **Folded at rest:** forearm back along the upper arm, fingers closed and pointing to the tail, wing held against the flank.
- **Wing-walk** (the reference tan wyvern): wrist on the ground, fingers folded up and back so the panels stack along the forearm.
- Roll the whole wing at the shoulder for flapping; fold with yaw at elbow, wrist and fingers. Keeping these two motions on different joints keeps the panels parallel.

### Legs: structure
- Read the reference's stance first: **pillar** (elephant, heavy dragons: nearly straight, wide feet), **cursorial zig-zag** (wolf, deer, most dragons' hind legs), **crouched biped** (wyvern, theropod), **sprawled** (lizard, crocodile, insects).
- Each leg is thigh or shoulder mass on the body, upper segment, lower segment, foot. Thickness steps down about 30% per segment, then the foot widens again.
- Add the shapes that make it a leg and not a post: a knee or elbow box, a calf bulge on the back of the lower segment, a heel block behind the foot. In the second test flight these three boxes per leg changed the read more than doubling the leg's thickness did.
- **Feet:** a pad in the body colour, three toes in the underside colour, and separate claws in the hard-part colour pitched down 15-20 degrees. Outer toes splay about 14 degrees. Reference dragons all show pale toes against a darker leg.
- Hind thighs are the biggest mass on the leg and overlap the hip by one to two units so the leg seems to grow out of the body.
- Paint horizontal bands sparingly on the upper and lower segments; one darker row in five.
- Before rigging, rotate each joint 30 degrees both ways and look: overlapping boxes at knee and ankle must cover the gap.

### Arms and hands
- Shoulder, upper arm, forearm, palm, then three fingers and a thumb, each one box plus a claw. Long thin fingers are separate boxes splayed at different angles (the reference elder dragon).
- Rest pose with elbows bent and fingers half curled; a straight arm with flat fingers looks like a toy.

### Machine limbs and wheels
- Joints are octagons: a hub box rotated 45 degrees, with a painted bolt, larger than the limb it joins. Note that Block-DSL's sculpted audit rejects two concentric boxes as an octagon; use one rotated box or offset the second.
- Limbs are beams with visible mechanics: a piston (thin rod sliding in a thicker sleeve), a parallel link, a cable or hose made of many short segments.
- Feet are plates or pads with toes as chamfered slabs. Wheels are an octagon tyre, a rim and a hub.
- Machines move by rigid rotation and sliding, which suits the format; design each moving part with a clear axis and leave a visible gap around it.

## Dragons and imaginary creatures

An invented creature is convincing when it borrows real anatomy and commits to one idea.

### Design recipe
1. **One-line concept**: element or habitat + temperament + role ("storm-charged alpine ambusher").
2. **Two or three animal anchors** for anatomy: for example wolf body, eagle wing, crocodile tail. Take proportions and joint layout from them.
3. **Shape language**: pick one and apply it to horns, spikes, plates and claws alike.
   - Triangles and blades: aggressive, fast.
   - Squares and slabs: heavy, armoured, ancient.
   - Curves and circles: gentle, aquatic, young.
4. **One signature feature** exaggerated 1.5x (a crown of horns, a hammer tail, glowing back crystals) and at most two supporting features. Everything else stays quiet.
5. **Palette of 3 + 1**: body colour, underside colour, hard-part colour (horn, claw, plate), plus one saturated accent that also glows.
6. **Plausibility pass**: where are the muscles that move each limb, where is the centre of mass, how does it fold its wings, how does it eat.

### Body plans and starting proportions

| Type | Limbs | Head : neck : torso : tail | Notes |
|---|---|---|---|
| Western dragon | 4 legs + 2 wings | 1 : 2 : 3 : 4 | Wings root above and behind the fore shoulders on their own mass; chest must be deep enough for two shoulder girdles |
| Wyvern | 2 legs + wing-arms | 1 : 1.5 : 2.5 : 3.5 | Walks on wing wrists or stands bird-like; big chest, small hips |
| Drake or fanged beast | 4 legs, no wings | 1 : 0.8 : 3 : 2.5 | Mammal-like stance, heavy forequarters, expressive head and mane |
| Brute | 2 heavy legs, small arms | 1.2 : 0.6 : 2.5 : 2.5 | Oversized head and jaw, thick tail as counterweight |
| Eastern dragon | 4 short legs | 1 : body 10-14 in one long S | Body is the neck; antlers, whiskers, mane, no wings |
| Serpent or leviathan | fins or none | 1 : body 8-12 | Read comes from the curve and the fin rhythm |
| Hydra | 4 legs, 3+ necks | necks 3 each | Vary neck heights and head poses; widest chest |
| Elder or boss | any | add 20% to head and signature feature | More layers of plates and a second glow colour |

### Element and theme motifs

| Theme | Forms | Surface and palette | Glow |
|---|---|---|---|
| Fire, magma | cracked plates, swept-back horns, ember vents | charcoal and deep red hide, bone horns | orange-yellow in cracks, throat, wing veins |
| Ice | crystal clusters, smooth blades, fur ruffs | pale blue and white, navy shadows | cyan in crystals and eyes |
| Storm | fin-like crests, forked horns, long whip tail | slate blue with gold plates | electric blue or white along the dorsal row |
| Poison, swamp | sacs, frills, drooping barbels, moss | olive and teal, sickly yellow belly | acid green in throat and sacs |
| Earth, stone | slab armour, club tail, short thick horns | ochre, grey, lichen patches | amber in seams, or none |
| Water, abyss | fins, trailing whiskers, smooth body | deep blue to pale belly, coral accents | bioluminescent spots in rows |
| Shadow, void | torn membranes, extra eyes, thin limbs | near-black purple with one hot accent | magenta or violet |

### Dragon head recipe
Skull, sloped two-part snout, deep jaw, strong brow, cheek box, two main horns of 4 segments, two shorter horns or frills, 2-4 visible fangs, a nose feature, and a dorsal crest starting between the horns. Around 35-60 boxes for a large dragon. View it alone from front, side and three-quarter before attaching it.

### Posing the bind pose
Do not model in a stiff T-pose. Give the rest pose a slight attitude: head lowered and turned a little, one fore foot ahead, tail curved, wings half open and raised. The rig can still animate from it, and every render looks alive.

### Common failures on invented creatures
- Wings attached to the same spot as the forelegs with no extra shoulder mass.
- Head too small; neck a straight pipe; tail a straight stick.
- Five different motifs fighting (spikes, fur, crystals, fins, armour). Keep one main, one supporting.
- Glow everywhere. Limit it to 3-5% of the surface, concentrated near the focal point.
- Symmetric, evenly spaced, same-size spikes down the whole back.
- A recognisable existing monster with the colours changed. Change the body plan, shape language and signature feature, not just the palette.

## 3. Spatial reasoning rules

- Measure in head-lengths first, convert to units second. Write ratios before sizes.
- Preferred angles: multiples of 7.5 or 11.25 degrees read as intentional; use 15, 22.5, 30, 45 for structure and small off-angles (3-8 degrees) for hand-made wobble.
- A slope is two overlapping boxes rotated toward each other, not a staircase.
- A taper is a chain of boxes each 70-85% of the previous cross-section.
- Overlap is allowed where one form flows into the next. Each box must still show a meaningful part of itself; never hide a box inside another.
- Keep at least a half-unit offset or a tilt between neighbouring planes and faces to avoid flicker.
- Left and right may differ slightly in pose (a raised paw, a tilted horn) once the base is symmetric.

## 4. Budgets

| Subject | Boxes and planes | Bones | Texture |
|---|---|---|---|
| Small prop or item | 10-40 | 1-5 | 32-64 px |
| Furniture, weapon, small animal | 30-90 | 5-20 | 64-128 px |
| Large creature | 120-350 | 40-110 | 128-256 px |
| Boss-scale dragon | 300-700 | 80-200 | 256-512 px |
| Room diorama | 300-1500 | few | several atlases |

Spend about half the budget on the head and the other focal feature.

## 5. Mapping to Block-DSL

- Use `audit_profile: sculpted` for this style and `size_step: 0.5`. Strict mode suits grid-aligned test assets only.
- Slopes and wobble: per-box `rotate` with `pivot`; curves: nested bones or `chain` with a `rest_euler` step and `taper`.
- Thin forms: `plane`, layered at different heights; fan by yaw so sheets stay parallel.
- Repeated detail with variation: macros with size and angle parameters, called with different values each time.
- Palettes: ramps with `hue_shift` set so lights go warm and shadows cool; lightness 0.6-0.9 because the bake darkens.
- Plain coats need `rd_feed=0.09, rd_kill=0.059`; lower feed gives blotches.
- Eyes, teeth, markings, glow bleed: stamps (or hand-painted `@textures` when the bake cannot express the look).
- Spinning parts carry no `joint_limits`. Leave a one-unit gap between surface plates and joint faces.
- After each stage run the audits and render front, side, top and both three-quarter views. Passing the audit is necessary, not sufficient: look at the render.

## Lessons from test flights

From the first full run (an original antlered forest dragon, 127 boxes):

- **Check proportions with numbers before detailing.** In the side silhouette measure head length against torso length and leg thickness against torso depth. On a standard-tier dragon the head should be at least a third of the torso length and the upper leg at least half the torso depth. The first pass had a head and legs about 30% too small and it was not obvious until rendered.
- **Look for floating parts in a close-up of every joint.** A jaw hung a unit and a half below the snout and still passed the audit. View the head alone from the side with the mouth at rest; the lower jaw's top must meet the tooth row.
- **Make the lower jaw the underside colour.** A cream jaw under a tooth row reads immediately; a body-coloured jaw with a small chin does not.
- **Underside plates must hug the body.** Plates that are rotated or stick out more than half a unit read as loose planks. Sink them one unit into the body and keep them parallel to it.
- **Bands on limbs: one dark row in five, not alternating halves.** Heavy alternating bands look like striped socks.
- **In Block-DSL a plane with `normal: x` takes `size: [height, length]`; with `normal: y` it is `[width, length]`; with `normal: z` it is `[width, height]`.** Getting this wrong still passes the audit.
- **Sibling planes on the centre line are coplanar.** Offset every second mane or fringe plane by a quarter unit sideways.
- **The repository's bake and renderer do not support this style's textures.** Paint onto the exported UV layout with a script or by hand, with alpha, and preview with a renderer that respects transparency. Record that the texture was produced outside the language.
- **Second run (wings and limbs):** partitioned seven-panel wings with triangular alpha bays folded cleanly in a top-view fold test; knee, calf and heel boxes fixed the stick-leg read. Still short of the references: wings about 70% of the size they should be, no continuous bay from the last finger to the body, underside plates on the chest still read as planks.
- **Third run (connection):** seating segments 28% into their parents, joint shingles and skull crest plates turned the row-of-boxes body into one form; wings enlarged to full size. Remaining gaps against the references: only one row of shingles (references use many), symmetric pose, rule-generated texture, no glow or scene.
- **Still weak after one run, so check these first next time:** torso reading as a row of boxes in side view (overlap and tilt the three masses more), dorsal fringe too short to change the silhouette (make it 15-20% of body depth), no asymmetry in the pose.

## 6. Self-critique before delivering

- Recognisable in silhouette from two angles?
- Is there a gesture curve, or is the spine a straight line?
- Do limbs and tail taper? Do horns and spikes vary?
- Is anything important still axis-aligned for no reason?
- Are there three clear value groups and a focal point with the strongest contrast and saturation?
- Do shadows shift hue? Any grey or noisy faces?
- Do the feet touch the ground and does the centre of mass sit over them?
- Does each kept identity trait from the sheet appear? Was anything copied one-to-one that should have been simplified?

## 7. Reference use and originality

Study existing designs to learn traits and proportions, then abstract. The goal is translation into this style, not a part-for-part reconstruction. For assets that go into a dataset or product, design original creatures that borrow *kinds* of traits (a storm-themed fanged wolf-dragon) rather than reproducing a named character from a game or another artist's model. Record the references studied and credit artists whose work informed the style.

## 8. Deliverables per asset

1. Trait sheet with reference images and keep/exaggerate/merge/paint/drop marks.
2. Stage renders (blockout, secondary, detail, textured) with the regional and whole checks.
3. The model source (Block-DSL or .bbmodel), texture, and audit result.
4. A short note on what was simplified and why.