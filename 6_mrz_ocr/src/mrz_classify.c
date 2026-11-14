#include "mrz_ocr.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float calc_grid_boundary_energy(const int32_t *vertProj, int lineW, int charCount) {
    float pitch = (float)lineW / charCount;
    int searchRadius = (int)(pitch * 0.12f);
    if (searchRadius < 1) searchRadius = 1;
    int64_t totalValleyEnergy = 0;

    for (int k = 1; k < charCount; ++k) {
        int centerX = (int)(roundf(k * pitch));
        int x0 = centerX - searchRadius;
        int x1 = centerX + searchRadius;
        if (x0 < 0) x0 = 0;
        if (x1 > lineW - 1) x1 = lineW - 1;

        int32_t minVal = vertProj[centerX];
        for (int x = x0; x <= x1; ++x) {
            if (vertProj[x] < minVal) minVal = vertProj[x];
        }
        totalValleyEnergy += minVal;
    }
    return (float)totalValleyEnergy / (charCount - 1);
}

/* Count vertical ink groups in the projection.
 *
 * MRZ is set on a fixed grid with a visible gutter between cells, so the
 * number of ink groups is a direct, pitch-independent estimate of the
 * character count: ~30 for a travel permit, ~44 for a passport row.  It
 * is the only signal here that does not need to know the pixel pitch, so
 * it settles the case the grid test cannot -- a small, noisy, single row
 * where both grids find a valley somewhere.
 *
 * Groups separated by a single empty column are merged, because a thin
 * stroke can leave a one-pixel hole inside a glyph. */
static int count_ink_groups(const int32_t *proj, int w, int h) {
    int thr = h / 8;
    if (thr < 1) thr = 1;
    int n = 0, in = 0, gap = 0;
    for (int x = 0; x < w; ++x) {
        if (proj[x] > thr) {
            if (!in) { n++; in = 1; }
            gap = 0;
        } else if (in) {
            gap++;
            if (gap > 1) in = 0;
        }
    }
    return n;
}

mrz_doc_layout_t mrz_classify_layout(const uint8_t *binaryBottomLine,
                                     int lineW, int lineH, int stride,
                                     const mrz_ocr_rect_t *allCandidateLines,
                                     int numLines) {
    if (lineW <= 0 || lineH <= 0 || !binaryBottomLine)
        return MRZ_LAYOUT_UNKNOWN;

    float aspectRatio = (float)lineW / (lineH > 1 ? lineH : 1);

    /* No MRZ row is a 5-pixel sliver.  Measured on
     * assets/pic/72ef4ce48bf27731_0002.jpg, whose band crop landed on
     * the data page instead of the MRZ: the splitter handed us a 1144x7
     * strip and the grid-energy vote below happily called it CS/CD
     * (e30=0.24 e44=0.40, groups=8) purely because 0.24 < 0.40*0.80.
     * A wrong 30-character decode is worse than a dropped frame.
     * Relaxed from 8 to 6 for very thin CS/CD permits whose text is
     * only 7 px after Otsu (2026-9-30-J). */
    if (lineH < 6)
        return MRZ_LAYOUT_UNKNOWN;

    /* The projection must cover the WHOLE line.  It used to be clamped
     * to a fixed 2048-column array, which silently truncated every band
     * wider than 2048 px -- a 2202 px line lost its last three
     * characters, the pitch was computed from 2048/44 instead of
     * 2202/44, and the 44-cell grid fell 7% out of phase.  The effect
     * was not a graceful degradation: on a genuine 44-character line the
     * clipped 44-grid scored a HIGHER boundary energy than the 30-grid
     * (measured e30=5.38 e44=9.63 on a scale-6 render), i.e. the
     * discriminator voted for the wrong answer exactly on the
     * high-resolution inputs it was meant to serve. */
    int validW = lineW;
    int32_t *vertProj = (int32_t *)calloc((size_t)validW, sizeof(int32_t));
    if (!vertProj) return MRZ_LAYOUT_UNKNOWN;
    for (int y = 0; y < lineH; ++y) {
        const uint8_t *row = binaryBottomLine + (size_t)y * stride;
        for (int x = 0; x < validW; ++x) {
            if (row[x] > 0) vertProj[x] += 1;
        }
    }

    float energy30 = calc_grid_boundary_energy(vertProj, validW, 30);
    float energy44 = calc_grid_boundary_energy(vertProj, validW, 44);
    int n_groups = count_ink_groups(vertProj, validW, lineH);

    /* MRZ_OCR_LAYOUT=1 dumps the two quantities the decision rests on.
     * Thresholds calibrated by eye on one corpus are the failure mode
     * this print exists to prevent: the synthetic renderer draws OCR-B
     * with an 8x12 cell (pitch/height 0.69) while a real MRZ cell is
     * ~0.98, so the SAME document produces a very different aspect
     * ratio on screen and on paper. */
    {
        static int dbg = -1;
        if (dbg < 0) {
            const char *e = getenv("MRZ_OCR_LAYOUT");
            dbg = (e && e[0] && strcmp(e, "0") != 0) ? 1 : 0;
        }
        if (dbg)
            fprintf(stderr, "LAYOUT w=%d h=%d aspect=%.2f e30=%.2f e44=%.2f "
                    "r=%.3f groups=%d nlines=%d\n",
                    lineW, lineH, aspectRatio, energy30, energy44,
                    energy30 > 0.0f ? energy44 / energy30 : 0.0f, n_groups,
                    numLines);
    }

    /* Which character grid fits this line better?
     *
     * Three signals, in decreasing order of reliability as measured on
     * synthetic renders at scale 3..6 with and without noise:
     *
     *   1. the ink-group count.  It is scale-free and it was right on
     *      every render tested (46 for a 44-cell row, 32 for a 30-cell
     *      row).  It counts a couple of extra groups where a thin stroke
     *      leaves a one-pixel hole, so the 44-vote needs |d| <= 3 and the
     *      30-vote |d| <= 6 (a 30-cell row has proportionally more '<'
     *      cells, whose thin strokes split more often); either vote only
     *      fires when the other hypothesis is far away.
     *   2. the dual-hypothesis grid valley energy.  Scale-invariant in
     *      principle, but on small noisy renders both grids find a
     *      valley and it fired the wrong way (e30=3.21 e44=4.93 on a
     *      genuine 44-cell row at scale 3), so it is now a second vote,
     *      not the first.  It also refuses to fire when the ink-group
     *      count says there is no text row here at all (n_groups < 24):
     *      a 30-cell row measures ~32 groups, so anything below 24 is
     *      decoration or speckle, and letting the grid test decide on it
     *      is how a 7-row border strip became "CS/CD".
     *   3. the aspect ratio, as the last resort.  Its absolute value
     *      depends on the renderer (synthetic OCR-B is drawn on an 8x12
     *      cell, pitch/height 0.69; a real MRZ cell is 2.54/2.60 = 0.98),
     *      so a 44-character line measures ~30 on a render and ~43 on
     *      paper and no single threshold serves both.
     */
    int is44Chars;
    {
        int d30 = n_groups - 30; if (d30 < 0) d30 = -d30;
        int d44 = n_groups - 44; if (d44 < 0) d44 = -d44;
        if (d44 <= 3 && d30 >= 8) {
            is44Chars = 1;
        } else if (d30 <= 6 && d44 >= 10) {
            is44Chars = 0;
        } else if (n_groups >= 15 && energy44 + 1e-3f < energy30 * 0.80f) {
            is44Chars = 1;
        } else if (n_groups >= 15 && energy30 + 1e-3f < energy44 * 0.80f) {
            /* Grid-energy vote lowered from n_groups>=24 to >=15: Sauvola
             * binarization on guilloche-background permits (CS/CD) yields
             * 18-21 groups instead of ~32, but the e44/e30 ratio is still
             * a strong discriminator (3-4x).  A non-text band with 15-23
             * groups has random projection energy ~1.0, well outside the
             * 0.80 threshold. */
            is44Chars = 0;
        } else if (aspectRatio >= 18.0f && aspectRatio <= 36.0f && d30 <= 12) {
            /* A real MRZ row is 29.3 (30 cells) to 43.0 (44 cells) cells
             * wide on paper, so a single row that measures inside the
             * 30-cell band is plausibly a permit.  18..36 is deliberately
             * loose on the low side to survive a synthetic renderer, which
             * draws OCR-B on an 8x12 cell and measures ~21 for the same
             * document.  The group count must also be within 12 of 30:
             * aspect alone let a truncated 44-cell passport row through
             * here (measured at scale 4 with noise and 1 degree of tilt). */
            is44Chars = 0;
        } else {
            /* Neither hypothesis fits: too few/many cells and an aspect
             * ratio no MRZ row has.  This is what a mislocated band looks
             * like -- e.g. the data page of a passport, whose two text
             * rows measure aspect 11 and 61 ink groups.  Reporting
             * UNKNOWN drops the frame; treating "not 44" as "therefore
             * 30" used to decode that page into a checksum-shaped
             * CS/CD-looking string. */
            free(vertProj);
            return MRZ_LAYOUT_UNKNOWN;
        }
    }

    mrz_doc_layout_t verdict = MRZ_LAYOUT_UNKNOWN;

    if (is44Chars) {
        /* A 44-cell line is only a TD3 passport if a same-width line sits
         * directly above it.  One 44-cell line alone means line 1 was
         * lost (glare, crop, tilt) -- that is a frame to drop, NOT a
         * travel permit, so it stays UNKNOWN on purpose. */
        if (numLines >= 2 && allCandidateLines) {
            const mrz_ocr_rect_t *bottom = &allCandidateLines[numLines - 1];
            const mrz_ocr_rect_t *upper  = &allCandidateLines[numLines - 2];
            int bH = bottom->h;
            int maxTol = bH * 2;
            int dxLeft = bottom->x - upper->x;
            if (dxLeft < 0) dxLeft = -dxLeft;
            int dw = bottom->w - upper->w;
            if (dw < 0) dw = -dw;
            if (dxLeft <= maxTol && dw <= bottom->w / 10) {
                verdict = MRZ_LAYOUT_TD3_DOUBLE_44;
            }
        }
    } else {
        /* 30-cell line.  Three equal-width stacked rows make a TD1 back
         * (home-return permit / taiwan compatriot pass); anything else is
         * a single-line CS/CD travel permit, including the case where a
         * Chinese caption line above it was paired as a pseudo second
         * row -- the width test below rejects that pairing. */
        if (numLines >= 2 && allCandidateLines) {
            const mrz_ocr_rect_t *bottom = &allCandidateLines[numLines - 1];
            const mrz_ocr_rect_t *upper  = &allCandidateLines[numLines - 2];
            int bH = bottom->h;
            int maxTol = bH * 2;
            int dxLeft = bottom->x - upper->x;
            if (dxLeft < 0) dxLeft = -dxLeft;
            int dw = bottom->w - upper->w;
            if (dw < 0) dw = -dw;
            int dh = bottom->h - upper->h;
            if (dh < 0) dh = -dh;
            /* The row still has to look like a 44-cell row before a
             * same-width row above it can be read as TD3 line 1.  A
             * CS/CD travel permit with a caption line of the same width
             * above it otherwise becomes a 44-cell TD3 and decodes into
             * 44 characters of garbage: assets/pic/1c6acd09..._0222.png
             * measured groups=28 (44 cells need ~44-46 groups, 30 cells
             * ~28-32), e30=3.62 vs e44=5.53 -- both counts favour the
             * 30-cell hypothesis.
             *
             * 30-cell hypothesis. */
            int looks44 = (n_groups >= 34) || (energy44 <= energy30);
            if (dxLeft <= maxTol && dw <= bottom->w / 10 && dh <= bH / 2
                && looks44) {
                verdict = MRZ_LAYOUT_TD3_DOUBLE_44;
            }
        }

        if (verdict == MRZ_LAYOUT_UNKNOWN && numLines >= 3 && allCandidateLines) {
            const mrz_ocr_rect_t *l0 = &allCandidateLines[numLines - 1];
            const mrz_ocr_rect_t *l1 = &allCandidateLines[numLines - 2];
            const mrz_ocr_rect_t *l2 = &allCandidateLines[numLines - 3];
            int maxTol = l0->h * 2;
            int dx01 = l0->x - l1->x;
            int dx02 = l0->x - l2->x;
            int dh01 = l0->h - l1->h;
            if (dx01 < 0) dx01 = -dx01;
            if (dx02 < 0) dx02 = -dx02;
            if (dh01 < 0) dh01 = -dh01;
            if (dx01 <= maxTol && dx02 <= maxTol && dh01 <= l0->h / 2) {
                verdict = MRZ_LAYOUT_TD1_TRIPLE_30;
            }
        }

        if (verdict == MRZ_LAYOUT_UNKNOWN)
            verdict = MRZ_LAYOUT_SINGLE_LINE_30;
    }

    free(vertProj);
    return verdict;
}
