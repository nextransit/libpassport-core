#!/usr/bin/env python3
"""One-off patch: keep the top-N prefixes per residue pair in the DP.

Why
---
The DP merges two prefixes that share the same (own, comp) residue pair
and keeps only the higher-scoring one.  That is exact for finding the
MOST LIKELY body, but it is exactly wrong for the case that matters
here: the truth's characters have probability ~0.000, so the truth's
prefix is discarded at an intermediate column and the DP can never
reach the end with it.

The fix is to keep the top-N prefixes per residue pair.  N is small
(2..4), so the state table stays bounded, and the search stays exact
for the top-N-per-residue problem -- which is what a decoder that has
to find a low-probability but checksum-consistent reading needs.
"""
import sys

P = "src/mrz_ocr_cnn.c"
s = open(P, encoding="utf-8").read()

if "TOP-N PREFIXES PER RESIDUE" in s:
    print("already patched")
    sys.exit(0)

old = """    /* The state table is indexed DIRECTLY by the residue pair
     * (own, comp) -> own * 10 + comp, so there is no side index and no
     * chance of the two drifting apart. */
    static mrz_field_dp cur[100], nxt[100];
    int live[100];
    int nlive = 0;

    for (int i = 0; i < 100; ++i) cur[i].score = -1e300;
    cur[0].score = 0.0;
    cur[0].own = 0;
    cur[0].comp = 0;
    cur[0].body[0] = 0;
    live[nlive++] = 0;   /* own 0, comp 0 */
"""

new = """    /* The state table is indexed DIRECTLY by the residue pair
     * (own, comp) -> own * 10 + comp, so there is no side index and no
     * chance of the two drifting apart.
     *
     * TOP-N PREFIXES PER RESIDUE.
     *
     * Merging every prefix that shares a residue pair and keeping only
     * the most likely one is exact for the MAXIMUM-LIKELINESS body,
     * but it is exactly wrong for the case that matters here: the
     * truth's characters have probability ~0.000, so the truth's
     * prefix is discarded at an intermediate column and the DP can
     * never reach the end with it.  A decoder that must find a
     * low-probability but checksum-consistent reading therefore has to
     * keep several prefixes per residue.
     *
     * N is small and the table is bounded, so the cost stays
     * (columns x 100 x N x candidates). */
    static mrz_field_dp cur[MRZ_FLD_DP_STATES], nxt[MRZ_FLD_DP_STATES];
    int live[MRZ_FLD_DP_STATES];
    int nlive = 0;

    for (int i = 0; i < MRZ_FLD_DP_STATES; ++i) cur[i].score = -1e300;
    cur[0].score = 0.0;
    cur[0].own = 0;
    cur[0].comp = 0;
    cur[0].body[0] = 0;
    live[nlive++] = 0;   /* own 0, comp 0 */
"""

assert old in s
s = s.replace(old, new, 1)

old2 = """    for (int k = 0; k < ncol; ++k) {
        int c = start + k;
        int w = wtab[k % 3];          /* field-local weight */
        int cw = mrz_comp_weight(c);  /* composite weight */
        for (int i = 0; i < 100; ++i) nxt[i].score = -1e300;
        for (int li = 0; li < nlive; ++li) {
            const mrz_field_dp *st = &cur[live[li]];
            /* every candidate class of this column is an option */
            for (int j = 0; j < mrz_topk(); ++j) {
                int g = t3[c].cand[j];
                if (g < 0) continue;
                if (j > 1 && g == t3[c].cand[1]) continue;  /* duplicate */
                char ch = MRZ_OCR_GLYPHS[g].ch;
                int v = mrz_cd_value(ch);
                int no = (st->own + v * w) % 10;
                int nc = (st->comp + v * cw) % 10;
                int ni = no * 10 + nc;
                double sc = st->score + mrz_beam_lp(&t3[c], j);
                if (sc > nxt[ni].score) {
                    nxt[ni].score = sc;
                    nxt[ni].own = no;
                    nxt[ni].comp = nc;
                    memcpy(nxt[ni].body, st->body, (size_t)k);
                    nxt[ni].body[k] = ch;
                }
            }
        }
        /* collect the live states of the next level */
        int nlive_next = 0;
        for (int i = 0; i < 100; ++i)
            if (nxt[i].score > -1e299) live[nlive_next++] = i;
        for (int i = 0; i < 100; ++i) cur[i] = nxt[i];
        nlive = nlive_next;
        if (nlive == 0) return 0;
    }
"""

new2 = """    for (int k = 0; k < ncol; ++k) {
        int c = start + k;
        int w = wtab[k % 3];          /* field-local weight */
        int cw = mrz_comp_weight(c);  /* composite weight */
        for (int i = 0; i < MRZ_FLD_DP_STATES; ++i) nxt[i].score = -1e300;
        /* Insertion into the per-residue top-N list. */
        for (int li = 0; li < nlive; ++li) {
            const mrz_field_dp *st = &cur[live[li]];
            /* every candidate class of this column is an option */
            for (int j = 0; j < mrz_topk(); ++j) {
                int g = t3[c].cand[j];
                if (g < 0) continue;
                if (j > 1 && g == t3[c].cand[1]) continue;  /* duplicate */
                char ch = MRZ_OCR_GLYPHS[g].ch;
                int v = mrz_cd_value(ch);
                int no = (st->own + v * w) % 10;
                int nc = (st->comp + v * cw) % 10;
                int ni = no * 10 + nc;
                double sc = st->score + mrz_beam_lp(&t3[c], j);
                /* find the insertion point in the per-residue list */
                int slot = -1;
                int count = 0;
                for (int q = 0; q < nnext; ++q) {
                    if (nnext_res[q] != ni) continue;
                    if (count == MRZ_FLD_DP_KEEP - 1) { slot = q; break; }
                    count++;
                    if (sc > nnext[q].score) { slot = q; break; }
                }
                if (slot < 0) {
                    /* append (if the list for this residue is not full)
                     * or drop (if it is) */
                    int cnt = 0;
                    for (int q = 0; q < nnext; ++q)
                        if (nnext_res[q] == ni) cnt++;
                    if (cnt < MRZ_FLD_DP_KEEP) {
                        nnext_res[nnext] = ni;
                        nnext[nnext].score = sc;
                        nnext[nnext].own = no;
                        nnext[nnext].comp = nc;
                        memcpy(nnext[nnext].body, st->body, (size_t)k);
                        nnext[nnext].body[k] = ch;
                        nnext++;
                        continue;
                    }
                    continue;   /* the residue list is full and sc is low */
                }
                /* shift the tail down and insert */
                for (int q = nnext; q > slot; --q) {
                    nnext[q] = nnext[q - 1];
                    nnext_res[q] = nnext_res[q - 1];
                }
                nnext_res[slot] = ni;
                nnext[slot].score = sc;
                nnext[slot].own = no;
                nnext[slot].comp = nc;
                memcpy(nnext[slot].body, st->body, (size_t)k);
                nnext[slot].body[k] = ch;
                nnext++;
            }
        }
        /* compact: keep only the states that are in the live list */
        for (int i = 0; i < nnext; ++i) cur[i] = nnext[i];
        for (int i = 0; i < nnext; ++i) live[i] = i;
        nlive = nnext;
        if (nlive == 0) return 0;
    }
"""

assert old2 in s
s = s.replace(old2, new2, 1)

# declare the temporaries used above
old3 = """    static const int wtab[3] = {7, 3, 1};
    int start = MRZ_FLD_BODY[f][0], end = MRZ_FLD_BODY[f][1];
    int ncol = end - start;
    int written = 0;

    if (ncol > 15) ncol = 15;
"""
new3 = """    static const int wtab[3] = {7, 3, 1};
    int start = MRZ_FLD_BODY[f][0], end = MRZ_FLD_BODY[f][1];
    int ncol = end - start;
    int written = 0;
    static mrz_field_dp nnext[MRZ_FLD_DP_STATES];
    static int nnext_res[MRZ_FLD_DP_STATES];
    int nnext_n = 0;

    if (ncol > 15) ncol = 15;
"""
assert old3 in s
s = s.replace(old3, new3, 1)

# the loop uses nnext as the counter; rename to avoid clashing with the
# array name
s = s.replace("for (int q = 0; q < nnext; ++q) {\n                    if (nnext_res[q] != ni) continue;",
              "for (int q = 0; q < nnext_n; ++q) {\n                    if (nnext_res[q] != ni) continue;")
s = s.replace("""                    int cnt = 0;
                    for (int q = 0; q < nnext; ++q)
                        if (nnext_res[q] == ni) cnt++;
                    if (cnt < MRZ_FLD_DP_KEEP) {
                        nnext_res[nnext] = ni;
                        nnext[nnext].score = sc;
                        nnext[nnext].own = no;
                        nnext[nnext].comp = nc;
                        memcpy(nnext[nnext].body, st->body, (size_t)k);
                        nnext[nnext].body[k] = ch;
                        nnext++;
                        continue;
                    }""",
"""                    int cnt = 0;
                    for (int q = 0; q < nnext_n; ++q)
                        if (nnext_res[q] == ni) cnt++;
                    if (cnt < MRZ_FLD_DP_KEEP && nnext_n < MRZ_FLD_DP_STATES) {
                        nnext_res[nnext_n] = ni;
                        nnext[nnext_n].score = sc;
                        nnext[nnext_n].own = no;
                        nnext[nnext_n].comp = nc;
                        memcpy(nnext[nnext_n].body, st->body, (size_t)k);
                        nnext[nnext_n].body[k] = ch;
                        nnext_n++;
                        continue;
                    }""")
s = s.replace("""                for (int q = nnext; q > slot; --q) {""",
              """                for (int q = nnext_n; q > slot; --q) {""")
s = s.replace("""                nnext[slot].body[k] = ch;
                nnext++;
            }""",
"""                nnext[slot].body[k] = ch;
                nnext_n++;
            }""")
s = s.replace("""        for (int i = 0; i < nnext; ++i) cur[i] = nnext[i];
        for (int i = 0; i < nnext; ++i) live[i] = i;
        nlive = nnext;""",
"""        for (int i = 0; i < nnext_n; ++i) cur[i] = nnext[i];
        for (int i = 0; i < nnext_n; ++i) live[i] = i;
        nlive = nnext_n;
        nnext_n = 0;""")

# the definition of the state count
old4 = """typedef struct {
    double score;
    int    own;
    int    comp;
    char   body[16];
} mrz_field_dp;"""
new4 = """typedef struct {
    double score;
    int    own;
    int    comp;
    char   body[16];
} mrz_field_dp;

/* How many prefixes to keep per residue pair.  1 gives the exact
 * maximum-likelihood body; more is needed to reach a low-probability
 * but checksum-consistent reading. */
#ifndef MRZ_FLD_DP_KEEP
#define MRZ_FLD_DP_KEEP 4
#endif
/* 100 residue pairs x MRZ_FLD_DP_KEEP prefixes. */
#define MRZ_FLD_DP_STATES (100 * MRZ_FLD_DP_KEEP)"""
assert old4 in s
s = s.replace(old4, new4, 1)

open(P, "w", encoding="utf-8").write(s)
print("patched")
