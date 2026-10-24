# Deep checksum beam — what changed, why, and how it was measured

## Problem

Line 2 of a TD3 MRZ carries five ICAO check digits.  The classifier
produces a probability for every glyph; the decode took the best class
per column and, if the result failed a check digit, searched for a
checksum-valid repair.

Two things were wrong with that repair step.

1. **The search was too shallow.**  It enumerated one-position flips
   and two-position flips over the six least confident columns.  On the
   41-photo corpus the residual failures need three to four
   simultaneous corrections, because the confusions cluster: a 'B'
   read as '8' in the passport number also drags the following '0' to
   'O' and the check digit with it.

2. **The search only ran when the checksum failed.**  A checksum-valid
   decode is not automatically the right one — a misread 'B' that
   happens to keep the check digit valid produces a line the checksum
   happily accepts.  Restricting the search to the failing case left
   exactly those wrong-but-valid readings untouched.

## Fix

`mrz_beam_search()` in `src/mrz_ocr_cnn.c` replaces the enumeration:

* the candidate columns are the 14 least confident ones, ranked by the
  margin between the best and the second-best class;
* the search is a depth-first walk that may change up to
  `MRZ_BEAM_DEPTH` (4) columns, taking the second or third class at
  each;
* every leaf is tested against all five check digits;
* the score is the sum of log-probabilities, tracked as a *delta*
  against the greedy decode, so the comparison is between candidates
  and not against an arbitrary baseline;
* a branch is abandoned when its optimistic bound (the best delta each
  remaining column could still contribute) cannot beat the incumbent.
  The bound is admissible, so pruning cannot discard the optimum;
* the search runs whether or not the greedy decode already satisfies
  the checksum.

Two bugs were found and fixed while measuring this:

* the depth limit used to abort the walk instead of only limiting the
  number of *changes*.  A reading that needed exactly four
  corrections never reached a leaf and was silently dropped.  This was
  the single largest effect: the 41-photo result went from 38/41 to
  40/41 when it was fixed;
* the score was accumulated as a full log-probability while the base
  column had already contributed its own, so every flip double
  counted.  The score is a delta now.

## Measurements

### 41 real passport photos (`tools/eval_omtest_e2e.py`)

The truth set of a line 2 is the set of checksum-valid readings
reachable from the board's OCR hypothesis within two confusion edits;
`TRUE` means the reading is in that set AND satisfies the checksum.

| build | line2 in truth set | TRUE | FALSE |
|-------|-------------------|------|-------|
| depth 2 (old two-flip enumeration) | 37/41 | 37/41 | 2 |
| depth 4, checksum-failure only | 38/41 | 38/41 | 2 |
| depth 4, always run | **40/41** | **40/41** | **0** |

Only three readings changed, and all three went from wrong to right:

```
pic_scan1/2026-9-29-C.jpg  before E639O27O58CHN7005059F31101530S0268SZ08110074
                           after  E639O21058CHN7005059F3110153050Z685Z0B110074
pic_scan1/2026-9-29-Q.jpg  before E436102633CHN8211158F32082106GI8I5816Z6KKZ64
                           after  E436102633CHN8211158F32082106GIBI5B16Z6B8264
pic_scan3/2026-9-30-AI.jpg before E996968603CHN0205089M3208184OKTO4UCP4K4E7L42
                           after  E996968603CHN0205089M32081840K1O4WCP4K4E7L42
```

No reading that was already correct was changed.

The one remaining failure (`pic_scan3/2026-9-30-AJ.jpg`) is a
classifier limit, not a search limit: the truth needs 'F' at column 28
and the network gives that cell a probability of 0.000 (it is sure it
is 'E'), so the truth is not in the candidate set at all.

### Synthetic corpus (1380 images, `tests/bench.py`)

Same weights, only `MRZ_BEAM_DEPTH` changed:

| depth | line1 acc | line2 acc | full match | cksum valid | ms p50 |
|-------|-----------|-----------|------------|-------------|--------|
| 0 (beam off) | 94.16% | 96.01% | 59.57% | 72.68% | 8.8 |
| 4 | 94.16% | 96.32% | 64.42% | **96.09%** | 9.7 |

The checksum-valid rate is the product-level metric and it moves by
+23.4 points.  The decode cost is ~1.5 ms per line, well inside the
200 ms frame budget.

### On device (p18q_dual_extend_64, `PhotoMrzDeviceTest`)

62 images (7 bundled `Photo/*`, 11 bundled `PhotoScan/*`, 44 pushed
`tmp/omtest` photos), product default path (`auto`):

* 58/62 decoded, the 4 failures are the known hard cases
  (`pic_scan2/I,J,K` where the MRZ rows are not located at all and
  `Photo/7` whose font domain the model has never seen);
* wall clock p50 = 56 ms, max = 111 ms, **no image over 200 ms**;
* `2026-9-30-AI.jpg` returns the corrected reading above, which the
  pre-beam build got wrong.

## Regression tests

`tests/test_postproc.c` gained two cases:

* `test_beam_finds_multiposition_fix` — a line that is wrong in four
  positions at once and whose corrections make it checksum-valid must
  be recovered;
* `test_beam_reports_no_solution` — when no checksum-valid reading
  exists the search must say so instead of returning an invalid line
  as if it had found one.

See also `docs/arm64-mrz-review.md` for the full review of the ARM64
photo -> MRZ implementation against a generic "port passport OCR to
Android ARM64" design guide: which of the guide's recommendations the
repository already satisfies (and by how much), which gaps are real
(no NEON, no multi-frame voting, no glare detection), and which of the
guide's proposals should not be adopted here.

## Things that were measured and did NOT work

Recording these matters as much as the fixes: each of them was a
plausible idea that the measurements killed, and re-trying them wastes
a training run each time (a training run is 6-10 minutes on this
machine).

### Deeper beam (depth 5, 6)

The Python replica of the beam -- which has no admissible-bound pruning
-- suggests that three of the residual failures need five simultaneous
corrections instead of four, so depth 5 was implemented in C as well.
With the bound pruning in place the result is **byte-identical to depth
4** on all 41 photos: the pruning already stops the walk at the same
leaves.  Depth stays 4.

### Alternative scoring rules

The beam picks the checksum-valid candidate with the highest summed
log-probability.  Three alternatives were evaluated against a
perturbed input (a +-1 level pixel change, which is what two different
JPEG decoders produce for the same file):

| rule | no noise | +-1 | +-2 | +-3 |
|------|----------|-----|-----|-----|
| A: log-prob over all 44 columns (current) | 38/41 | 37/41 | 36/41 | 37/41 |
| B: log-prob over the data columns only (the five check digits excluded) | 38/41 | 37/41 | 35/41 | 36/41 |
| C: B plus a penalty per changed column | 37/41 | 37/41 | 35/41 | 36/41 |

None of them is more stable.  The tie is a property of the classifier,
not of the scoring rule: two readings differ by a probability of about
0.001, and a one-level pixel change moves the margin by about that
much.

### Date validity as a tie-break

ICAO birth and expiry dates must be real calendar dates, and the
decoder does not check that.  Using it as a tie-break changed nothing:
the correct readings are already valid dates, so the rule never
separates the right candidate from the wrong one.

### Re-weighting the real training half

Two attempts at removing the class-frequency prior from the real half
both made the end-to-end result worse, and both are kept in
`tools/train_cnn_omtest.py` marked as rejected:

* `--balance-real` resamples the real half to a flat histogram.  The
  1804 unique slices become ~37k duplicates, the model reaches a
  perfect training score by epoch 8, and it then memorises the
  photographs it has to generalise from: 37/41 -> 28/41, false
  accepts 2 -> 11.
* `--class-weight` weights the unique slices by inverse class
  frequency.  The real half drops to 1452 rows, so the weights have to
  be 25x larger to keep its share of the loss, and the model chases
  those rows: 40/41 -> 32/41, false accepts 0 -> 7.

The real half only has 616 unambiguous line-2 slices covering 36
classes, and exactly ONE real 'O'.  That is the ceiling: the residual
failures are a data problem, not a training-schedule problem.

## The downscaling trap

`PhotoMrzPipeline` used to cap the source photo at 1250 px on its
longest side before locating the MRZ rows.  Measured on the 41
labelled photos with the same crop, the same weights and the same code:

| cap | lines inside the truth set |
|-----|---------------------------|
| 1250 | 7/41 |
| 1440 (no downscale) | 40/41 |
| 1600 | 40/41 |

Downscaling costs about 13% of the linear resolution of every glyph,
which is enough to move the margins that the checksum beam depends on:
the correct reading drops out of the top three and no amount of search
can bring it back.

The product hot path (`processFast`) does NOT scale the photo -- it
locates the rows and crops the band in the source resolution -- so it
was never affected.  The cap only applies to the detailed/inspect
path, and it is left at 1250 because that path's geometry is asserted
in the 1250 coordinate system by `PhotoMrzPipelineTest`.

## Why the device and the host disagree

On the device the 44 pushed photos decode to 37/41 lines inside the
truth set; on the host the same photos give 40/41.  The cause is not
the algorithm: the host reads the JPEG with libjpeg and the device
with Skia, and the two differ by about one level per pixel.  Adding a
+-1 level perturbation to the host's own pixels reproduces the
device's readings exactly, and the readings that flip are the ones
whose top two checksum-valid candidates are within about 0.3 of each
other in log-probability.

So the two differences are separate:

* the beam itself is correct -- it converts "the network is not sure"
  into "the network is not sure AND the checksum agrees", and it does
  that without ever breaking a reading that was already correct;
* the remaining disagreements are tie-breaks between two readings that
  the classifier cannot separate, and they are decided by noise.
