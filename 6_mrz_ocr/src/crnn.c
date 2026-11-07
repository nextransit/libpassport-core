#include "crnn.h"
/* Each weight set ships its OWN CRNN head, trained on that set's CNN
 * features (features come out of the line head, not out of a shared
 * trunk).  Reusing the passport head in the pic binary was measured as
 * a silent failure (§21.3: a head scoring 28% while `fuse == cnn` hid
 * it), so the pic variant carried none until now -- selecting
 * "CRNN序列"/"融合CNN+CRNN" silently ran the plain CNN head there. */
#if defined(MRZ_WEIGHTS_SET_PIC)
#include "crnn_weights_pic.h"
#else
#include "crnn_weights.h"
#endif
#include <math.h>
#include <string.h>

int crnn_weights_available(void) {
    return MRZ_CRNN_WEIGHTS_AVAILABLE;
}

/* Load one direction's GRU weights.  The forward and backward halves of
 * the exported header use the same array names with a _B suffix for the
 * backward direction, so the two directions share one loader; the
 * `backward` flag selects which set of struct members is written. */
static void crnn_load_dir(crnn_seq_t *net, int backward,
                           const float *proj_w, const float *proj_b,
                           const float *pos_w,
                           const float *wz, const float *uz, const float *bz,
                           const float *wr, const float *ur, const float *br,
                           const float *wh, const float *uh,
                           const float *bh_in, const float *bh_hn) {
    /* The projection and the position embedding are shared by both
     * directions, so they are only written once. */
    if (!backward) {
        memcpy(net->proj_w, proj_w, sizeof(net->proj_w));
        memcpy(net->proj_b, proj_b, sizeof(net->proj_b));
        memcpy(net->pos_w, pos_w, sizeof(net->pos_w));
    }
    if (backward) {
        memcpy(net->wzb, wz, sizeof(net->wzb));
        memcpy(net->uzb, uz, sizeof(net->uzb));
        memcpy(net->bzb, bz, sizeof(net->bzb));
        memcpy(net->wrb, wr, sizeof(net->wrb));
        memcpy(net->urb, ur, sizeof(net->urb));
        memcpy(net->brb, br, sizeof(net->brb));
        memcpy(net->whb, wh, sizeof(net->whb));
        memcpy(net->uhb, uh, sizeof(net->uhb));
        memcpy(net->bh_inb, bh_in, sizeof(net->bh_inb));
        memcpy(net->bh_hnb, bh_hn, sizeof(net->bh_hnb));
    } else {
        memcpy(net->wz, wz, sizeof(net->wz));
        memcpy(net->uz, uz, sizeof(net->uz));
        memcpy(net->bz, bz, sizeof(net->bz));
        memcpy(net->wr, wr, sizeof(net->wr));
        memcpy(net->ur, ur, sizeof(net->ur));
        memcpy(net->br, br, sizeof(net->br));
        memcpy(net->wh, wh, sizeof(net->wh));
        memcpy(net->uh, uh, sizeof(net->uh));
        memcpy(net->bh_in, bh_in, sizeof(net->bh_in));
        memcpy(net->bh_hn, bh_hn, sizeof(net->bh_hn));
    }
}

void crnn_init(crnn_seq_t *net) {
    /* Single-head init is kept for tests/tools: it loads the line-1 head.
     * Production uses crnn_pair_init(), which loads both heads. */
    if (!net) return;
    memset(net, 0, sizeof(*net));
#if MRZ_CRNN_WEIGHTS_AVAILABLE
    crnn_load_dir(net, 0,
                  CRNN1_PROJ_W, CRNN1_PROJ_B, CRNN1_POS_W,
                  CRNN1_WZ, CRNN1_UZ, CRNN1_BZ,
                  CRNN1_WR, CRNN1_UR, CRNN1_BR,
                  CRNN1_WH, CRNN1_UH, CRNN1_BH_IN, CRNN1_BH_HN);
    crnn_load_dir(net, 1,
                  CRNN1_PROJ_W, CRNN1_PROJ_B, CRNN1_POS_W,
                  CRNN1_WZB, CRNN1_UZB, CRNN1_BZB,
                  CRNN1_WRB, CRNN1_URB, CRNN1_BRB,
                  CRNN1_WHB, CRNN1_UHB, CRNN1_BH_INB, CRNN1_BH_HNB);
    memcpy(net->out_w, CRNN1_OUT_W, sizeof(net->out_w));
    memcpy(net->out_b, CRNN1_OUT_B, sizeof(net->out_b));
#endif
}

void crnn_pair_init(crnn_pair_t *pair) {
    if (!pair) return;
    crnn_init(&pair->line1);
    crnn_init(&pair->line2);
#if MRZ_CRNN_WEIGHTS_AVAILABLE
    crnn_load_dir(&pair->line2, 0,
                  CRNN2_PROJ_W, CRNN2_PROJ_B, CRNN2_POS_W,
                  CRNN2_WZ, CRNN2_UZ, CRNN2_BZ,
                  CRNN2_WR, CRNN2_UR, CRNN2_BR,
                  CRNN2_WH, CRNN2_UH, CRNN2_BH_IN, CRNN2_BH_HN);
    crnn_load_dir(&pair->line2, 1,
                  CRNN2_PROJ_W, CRNN2_PROJ_B, CRNN2_POS_W,
                  CRNN2_WZB, CRNN2_UZB, CRNN2_BZB,
                  CRNN2_WRB, CRNN2_URB, CRNN2_BRB,
                  CRNN2_WHB, CRNN2_UHB, CRNN2_BH_INB, CRNN2_BH_HNB);
    memcpy(pair->line2.out_w, CRNN2_OUT_W, sizeof(pair->line2.out_w));
    memcpy(pair->line2.out_b, CRNN2_OUT_B, sizeof(pair->line2.out_b));
#endif
}

static float sigmoidf_safe(float x) {
    if (x < -12.0f) return 0.00000614f;
    if (x > 12.0f) return 0.99999386f;
    return 1.0f / (1.0f + expf(-x));
}

/* Fast tanh via 3rd-order rational approximation.
 * tanh(x) ≈ x * (27 + x²) / (27 + 9 * x²), max error < 0.5% for |x| < 4.5.
 * For |x| >= 4.5, tanh ≈ ±1 (error < 0.0001). */
static inline float fast_tanhf(float x) {
    if (x > 4.5f) return 1.0f;
    if (x < -4.5f) return -1.0f;
    float x2 = x * x;
    return x * (27.0f + x2) / (27.0f + 9.0f * x2);
}

/* Fast sigmoid via tanh identity: sigmoid(x) = 0.5 + 0.5 * tanh(x/2). */
static inline float fast_sigmoidf(float x) {
    if (x < -12.0f) return 0.00000614f;
    if (x > 12.0f) return 0.99999386f;
    return 0.5f + 0.5f * fast_tanhf(0.5f * x);
}

/* One GRU direction over the whole row.
 *
 * `emb` holds the projected + position-embedded input for every step
 * (steps x CRNN_HIDDEN).  `out` receives the hidden state per step.
 * `reverse` selects the backward traversal.
 *
 * PyTorch GRU new-gate: n = tanh(W x + b_in + r * (U h + b_hn)).
 * The reset gate multiplies the WHOLE recurrent term including its
 * bias, so the bias is split into an input part and a recurrent part. */
static void crnn_gru_dir(const float *emb, int steps,
                         int reverse,
                         const float *wz, const float *uz, const float *bz,
                         const float *wr, const float *ur, const float *br,
                         const float *wh, const float *uh,
                         const float *bh_in, const float *bh_hn,
                         float *out) {
    float h[CRNN_HIDDEN];
    for (int j = 0; j < CRNN_HIDDEN; ++j) h[j] = 0.0f;
    for (int k = 0; k < steps; ++k) {
        int t = reverse ? (steps - 1 - k) : k;
        const float *x = emb + t * CRNN_HIDDEN;
        float z[CRNN_HIDDEN], r[CRNN_HIDDEN], n[CRNN_HIDDEN];
        /* Fused z/r/n gate loop: load x[i] and h[i] once instead of 3×. */
        for (int j = 0; j < CRNN_HIDDEN; ++j) {
            float sz = bz[j], sr = br[j], sn = bh_in[j], sh = bh_hn[j];
            for (int i = 0; i < CRNN_HIDDEN; ++i) {
                float xi = x[i], hi = h[i];
                int base = i * CRNN_HIDDEN + j;
                sz += xi * wz[base] + hi * uz[base];
                sr += xi * wr[base] + hi * ur[base];
                sn += xi * wh[base];
                sh += hi * uh[base];
            }
            z[j] = sigmoidf_safe(sz);
            r[j] = sigmoidf_safe(sr);
            n[j] = tanhf(sn + r[j] * sh);
        }
        for (int j = 0; j < CRNN_HIDDEN; ++j)
            h[j] = (1.0f - z[j]) * n[j] + z[j] * h[j];
        for (int j = 0; j < CRNN_HIDDEN; ++j)
            out[t * CRNN_HIDDEN + j] = h[j];
    }
}

static void crnn_forward_net(const crnn_seq_t *net, const float *features,
                             int steps, float *probs) {
    if (!net || !features || !probs || steps <= 0) return;
    if (steps > CRNN_STEPS) steps = CRNN_STEPS;

    /* Project + position embedding. */
    float emb[CRNN_STEPS * CRNN_HIDDEN];
    for (int t = 0; t < steps; ++t) {
        float *e = emb + t * CRNN_HIDDEN;
        for (int j = 0; j < CRNN_HIDDEN; ++j) {
            float s = net->proj_b[j];
            for (int i = 0; i < CRNN_FEATURES; ++i)
                s += features[t * CRNN_FEATURES + i] * net->proj_w[i * CRNN_HIDDEN + j];
            e[j] = (s > 0.0f ? s : 0.0f) + net->pos_w[t * CRNN_HIDDEN + j];
        }
    }

    float fwd[CRNN_STEPS * CRNN_HIDDEN];
    float bwd[CRNN_STEPS * CRNN_HIDDEN];
    crnn_gru_dir(emb, steps, 0,
                 net->wz, net->uz, net->bz,
                 net->wr, net->ur, net->br,
                 net->wh, net->uh, net->bh_in, net->bh_hn, fwd);
    crnn_gru_dir(emb, steps, 1,
                 net->wzb, net->uzb, net->bzb,
                 net->wrb, net->urb, net->brb,
                 net->whb, net->uhb, net->bh_inb, net->bh_hnb, bwd);

    for (int t = 0; t < steps; ++t) {
        float *row = probs + t * CRNN_OUT;
        float maxv = -INFINITY;
        for (int o = 0; o < CRNN_OUT; ++o) {
            float s = net->out_b[o];
            for (int j = 0; j < CRNN_HIDDEN; ++j)
                s += fwd[t * CRNN_HIDDEN + j] * net->out_w[j * CRNN_OUT + o];
            for (int j = 0; j < CRNN_HIDDEN; ++j)
                s += bwd[t * CRNN_HIDDEN + j] *
                     net->out_w[(CRNN_HIDDEN + j) * CRNN_OUT + o];
            row[o] = s;
            if (s > maxv) maxv = s;
        }
        float total = 0.0f;
        for (int o = 0; o < CRNN_OUT; ++o) { row[o] = expf(row[o] - maxv); total += row[o]; }
        for (int o = 0; o < CRNN_OUT; ++o) row[o] /= total > 0.0f ? total : 1.0f;
    }
}

void crnn_forward_line(const crnn_pair_t *pair, int line_idx,
                       const float *features, int steps, float *probs) {
    if (!pair) return;
    crnn_forward_net(line_idx == 0 ? &pair->line1 : &pair->line2,
                     features, steps, probs);
}
