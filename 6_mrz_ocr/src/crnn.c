#include "crnn.h"
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
#include <arm_neon.h>
#endif
#if defined(MRZ_WEIGHTS_SET_PIC)
#include "crnn_weights_pic.h"
#else
#include "crnn_weights.h"
#endif
#include "crnn_weights_cscd.h"
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
    crnn_cscd_init(&pair->cscd);
}

void crnn_cscd_init(crnn_seq_t *net) {
    if (!net) return;
    memset(net, 0, sizeof(*net));
#if defined(MRZ_CRNN_CSCD_WEIGHTS_AVAILABLE) && MRZ_CRNN_CSCD_WEIGHTS_AVAILABLE
    /* pos_w for CSCD is only CRNN_STEPS_CSCD * CRNN_HIDDEN floats,
     * not CRNN_STEPS * CRNN_HIDDEN, so we cannot use crnn_load_dir()
     * which copies sizeof(net->pos_w) bytes. */
    memcpy(net->proj_w, CRNN_CSCD_PROJ_W, sizeof(net->proj_w));
    memcpy(net->proj_b, CRNN_CSCD_PROJ_B, sizeof(net->proj_b));
    memcpy(net->pos_w, CRNN_CSCD_POS_W,
           CRNN_STEPS_CSCD * CRNN_HIDDEN * sizeof(float));
    /* Forward direction. */
    memcpy(net->wz, CRNN_CSCD_WZ, sizeof(net->wz));
    memcpy(net->uz, CRNN_CSCD_UZ, sizeof(net->uz));
    memcpy(net->bz, CRNN_CSCD_BZ, sizeof(net->bz));
    memcpy(net->wr, CRNN_CSCD_WR, sizeof(net->wr));
    memcpy(net->ur, CRNN_CSCD_UR, sizeof(net->ur));
    memcpy(net->br, CRNN_CSCD_BR, sizeof(net->br));
    memcpy(net->wh, CRNN_CSCD_WH, sizeof(net->wh));
    memcpy(net->uh, CRNN_CSCD_UH, sizeof(net->uh));
    memcpy(net->bh_in, CRNN_CSCD_BH_IN, sizeof(net->bh_in));
    memcpy(net->bh_hn, CRNN_CSCD_BH_HN, sizeof(net->bh_hn));
    /* Backward direction. */
    memcpy(net->wzb, CRNN_CSCD_WZB, sizeof(net->wzb));
    memcpy(net->uzb, CRNN_CSCD_UZB, sizeof(net->uzb));
    memcpy(net->bzb, CRNN_CSCD_BZB, sizeof(net->bzb));
    memcpy(net->wrb, CRNN_CSCD_WRB, sizeof(net->wrb));
    memcpy(net->urb, CRNN_CSCD_URB, sizeof(net->urb));
    memcpy(net->brb, CRNN_CSCD_BRB, sizeof(net->brb));
    memcpy(net->whb, CRNN_CSCD_WHB, sizeof(net->whb));
    memcpy(net->uhb, CRNN_CSCD_UHB, sizeof(net->uhb));
    memcpy(net->bh_inb, CRNN_CSCD_BH_INB, sizeof(net->bh_inb));
    memcpy(net->bh_hnb, CRNN_CSCD_BH_HNB, sizeof(net->bh_hnb));
    /* Classifier. */
    memcpy(net->out_w, CRNN_CSCD_OUT_W, sizeof(net->out_w));
    memcpy(net->out_b, CRNN_CSCD_OUT_B, sizeof(net->out_b));
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
 * bias, so the bias is split into an input part and a recurrent part.
 *
 * Loop order is i-outer / j-inner so weight access (wz[i*H+j] etc.)
 * is contiguous in j, enabling NEON 4-way SIMD.  Each j accumulator
 * still receives terms in i=0,1,...,H-1 order, preserving the same
 * floating-point accumulation sequence per output. */
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
        float sz[CRNN_HIDDEN], sr[CRNN_HIDDEN], sn[CRNN_HIDDEN], sh[CRNN_HIDDEN];
        for (int j = 0; j < CRNN_HIDDEN; ++j) {
            sz[j] = bz[j]; sr[j] = br[j];
            sn[j] = bh_in[j]; sh[j] = bh_hn[j];
        }
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        for (int i = 0; i < CRNN_HIDDEN; ++i) {
            float xi = x[i], hi = h[i];
            const float *wz_i = wz + i * CRNN_HIDDEN;
            const float *uz_i = uz + i * CRNN_HIDDEN;
            const float *wr_i = wr + i * CRNN_HIDDEN;
            const float *ur_i = ur + i * CRNN_HIDDEN;
            const float *wh_i = wh + i * CRNN_HIDDEN;
            const float *uh_i = uh + i * CRNN_HIDDEN;
            float32x4_t xiv = vdupq_n_f32(xi);
            float32x4_t hiv = vdupq_n_f32(hi);
            for (int j = 0; j < CRNN_HIDDEN; j += 4) {
                float32x4_t sz_v = vld1q_f32(sz + j);
                float32x4_t sr_v = vld1q_f32(sr + j);
                float32x4_t sn_v = vld1q_f32(sn + j);
                float32x4_t sh_v = vld1q_f32(sh + j);
                sz_v = vmlaq_f32(sz_v, xiv, vld1q_f32(wz_i + j));
                sz_v = vmlaq_f32(sz_v, hiv, vld1q_f32(uz_i + j));
                sr_v = vmlaq_f32(sr_v, xiv, vld1q_f32(wr_i + j));
                sr_v = vmlaq_f32(sr_v, hiv, vld1q_f32(ur_i + j));
                sn_v = vmlaq_f32(sn_v, xiv, vld1q_f32(wh_i + j));
                sh_v = vmlaq_f32(sh_v, hiv, vld1q_f32(uh_i + j));
                vst1q_f32(sz + j, sz_v);
                vst1q_f32(sr + j, sr_v);
                vst1q_f32(sn + j, sn_v);
                vst1q_f32(sh + j, sh_v);
            }
        }
#else
        for (int i = 0; i < CRNN_HIDDEN; ++i) {
            float xi = x[i], hi = h[i];
            const float *wz_i = wz + i * CRNN_HIDDEN;
            const float *uz_i = uz + i * CRNN_HIDDEN;
            const float *wr_i = wr + i * CRNN_HIDDEN;
            const float *ur_i = ur + i * CRNN_HIDDEN;
            const float *wh_i = wh + i * CRNN_HIDDEN;
            const float *uh_i = uh + i * CRNN_HIDDEN;
            for (int j = 0; j < CRNN_HIDDEN; ++j) {
                sz[j] += xi * wz_i[j] + hi * uz_i[j];
                sr[j] += xi * wr_i[j] + hi * ur_i[j];
                sn[j] += xi * wh_i[j];
                sh[j] += hi * uh_i[j];
            }
        }
#endif
        float z[CRNN_HIDDEN], r[CRNN_HIDDEN], n[CRNN_HIDDEN];
        for (int j = 0; j < CRNN_HIDDEN; ++j) {
            z[j] = sigmoidf_safe(sz[j]);
            r[j] = sigmoidf_safe(sr[j]);
            n[j] = tanhf(sn[j] + r[j] * sh[j]);
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

    /* Project + position embedding: proj_w is [CRNN_FEATURES x CRNN_HIDDEN]
     * stored row-major (i*H+j).  Loop i-outer / j-inner for contiguous
     * weight access and NEON 4-way parallelism across j. */
    float emb[CRNN_STEPS * CRNN_HIDDEN];
    for (int t = 0; t < steps; ++t) {
        float *e = emb + t * CRNN_HIDDEN;
        const float *ft = features + t * CRNN_FEATURES;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        for (int j = 0; j < CRNN_HIDDEN; j += 4)
            vst1q_f32(e + j, vld1q_f32(net->proj_b + j));
        for (int i = 0; i < CRNN_FEATURES; ++i) {
            float32x4_t fi = vdupq_n_f32(ft[i]);
            const float *wr = net->proj_w + i * CRNN_HIDDEN;
            for (int j = 0; j < CRNN_HIDDEN; j += 4) {
                float32x4_t ev = vld1q_f32(e + j);
                ev = vmlaq_f32(ev, fi, vld1q_f32(wr + j));
                vst1q_f32(e + j, ev);
            }
        }
        for (int j = 0; j < CRNN_HIDDEN; j += 4) {
            float32x4_t ev = vld1q_f32(e + j);
            float32x4_t posv = vld1q_f32(net->pos_w + t * CRNN_HIDDEN + j);
            ev = vaddq_f32(vmaxq_f32(ev, vdupq_n_f32(0.0f)), posv);
            vst1q_f32(e + j, ev);
        }
#else
        for (int j = 0; j < CRNN_HIDDEN; ++j) {
            float s = net->proj_b[j];
            for (int i = 0; i < CRNN_FEATURES; ++i)
                s += ft[i] * net->proj_w[i * CRNN_HIDDEN + j];
            e[j] = (s > 0.0f ? s : 0.0f) + net->pos_w[t * CRNN_HIDDEN + j];
        }
#endif
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

    /* Output classifier: out_w is [CRNN_CTX x CRNN_OUT] stored row-major
     * (j*OUT+o).  For each timestep, compute logits[o] = out_b[o] +
     * sum_j fwd[j]*out_w[j*OUT+o] + sum_j bwd[j]*out_w[(H+j)*OUT+o].
     * Loop j-outer / o-inner for contiguous weight access and NEON. */
    for (int t = 0; t < steps; ++t) {
        float *row = probs + t * CRNN_OUT;
        const float *fwd_t = fwd + t * CRNN_HIDDEN;
        const float *bwd_t = bwd + t * CRNN_HIDDEN;
#if defined(__ARM_NEON) || defined(__ARM_NEON__)
        int o4 = CRNN_OUT & ~3;
        float32x4_t logitsv[10];
        for (int o = 0; o < o4; o += 4) {
            logitsv[o >> 2] = vld1q_f32(net->out_b + o);
        }
        for (int j = 0; j < CRNN_HIDDEN; ++j) {
            float fv = fwd_t[j];
            float32x4_t fvv = vdupq_n_f32(fv);
            const float *w = net->out_w + j * CRNN_OUT;
            for (int o = 0; o < o4; o += 4) {
                logitsv[o >> 2] = vmlaq_f32(logitsv[o >> 2], fvv, vld1q_f32(w + o));
            }
        }
        for (int j = 0; j < CRNN_HIDDEN; ++j) {
            float bv = bwd_t[j];
            float32x4_t bvv = vdupq_n_f32(bv);
            const float *w = net->out_w + (CRNN_HIDDEN + j) * CRNN_OUT;
            for (int o = 0; o < o4; o += 4) {
                logitsv[o >> 2] = vmlaq_f32(logitsv[o >> 2], bvv, vld1q_f32(w + o));
            }
        }
        float tmp[CRNN_OUT];
        for (int o = 0; o < o4; o += 4)
            vst1q_f32(tmp + o, logitsv[o >> 2]);
        for (int o = o4; o < CRNN_OUT; ++o) {
            float s = net->out_b[o];
            for (int j = 0; j < CRNN_HIDDEN; ++j)
                s += fwd_t[j] * net->out_w[j * CRNN_OUT + o];
            for (int j = 0; j < CRNN_HIDDEN; ++j)
                s += bwd_t[j] * net->out_w[(CRNN_HIDDEN + j) * CRNN_OUT + o];
            tmp[o] = s;
        }
        float maxv = -INFINITY;
        for (int o = 0; o < CRNN_OUT; ++o)
            if (tmp[o] > maxv) maxv = tmp[o];
        float total = 0.0f;
        for (int o = 0; o < CRNN_OUT; ++o) {
            row[o] = expf(tmp[o] - maxv);
            total += row[o];
        }
        float inv = total > 0.0f ? 1.0f / total : 1.0f;
        for (int o = 0; o < CRNN_OUT; ++o) row[o] *= inv;
#else
        float maxv = -INFINITY;
        for (int o = 0; o < CRNN_OUT; ++o) {
            float s = net->out_b[o];
            for (int j = 0; j < CRNN_HIDDEN; ++j)
                s += fwd_t[j] * net->out_w[j * CRNN_OUT + o];
            for (int j = 0; j < CRNN_HIDDEN; ++j)
                s += bwd_t[j] * net->out_w[(CRNN_HIDDEN + j) * CRNN_OUT + o];
            row[o] = s;
            if (s > maxv) maxv = s;
        }
        float total = 0.0f;
        for (int o = 0; o < CRNN_OUT; ++o) { row[o] = expf(row[o] - maxv); total += row[o]; }
        for (int o = 0; o < CRNN_OUT; ++o) row[o] /= total > 0.0f ? total : 1.0f;
#endif
    }
}

void crnn_forward_line(const crnn_pair_t *pair, int line_idx,
                       const float *features, int steps, float *probs) {
    if (!pair) return;
    crnn_forward_net(line_idx == 0 ? &pair->line1 : &pair->line2,
                     features, steps, probs);
}

void crnn_forward_cscd(const crnn_seq_t *net,
                       const float *features, int steps, float *probs) {
    crnn_forward_net(net, features, steps, probs);
}
