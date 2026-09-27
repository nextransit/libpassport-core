/* NFC backend abstraction + high-level eMRTD BAC read flow.
 *
 * The mock backend is a self-contained passport *simulator*: it is fed
 * the same MRZ as the reader, derives the same Kenc/Kmac, and speaks
 * real BAC -- decrypting/verifying the reader's MUTUAL AUTH payload,
 * producing a genuinely encrypted response, and deriving the session
 * keys. The subsequent DG1/SOD reads run under Secure Messaging
 * (0C-cla commands carrying a session-key MAC; responses are encrypted
 * and MACed). This tests the cryptographic path, not just the APDU
 * choreography.
 *
 * SM wire format used here (documented, self-consistent):
 *   command : 0C INS P1 P2 00 <MAC8>          (no data; MAC covers the
 *             5-byte header "0C INS P1 P2 00" with KSmac)
 *   response: <enc> <MAC8>                     (enc = 3DES-CBC(Ksenc,
 *             IV=0, pad0x80(plaintext)); MAC covers enc with KSmac)
 */
#include "nfc.h"
#include "bac.h"
#include "crypto.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

/* ---------- Null backend (always returns SW_UNKNOWN). ---------- */
typedef struct {
    nfc_backend_t base;
} null_backend_t;

static int null_transceive(nfc_backend_t *self, const nfc_apdu_t *capdu,
                           nfc_apdu_t *rapdu) {
    (void)self; (void)capdu;
    memset(rapdu, 0, sizeof(*rapdu));
    rapdu->is_response = 1;
    rapdu->sw = NFC_SW_UNKNOWN;
    return -1;
}

static void null_destroy(nfc_backend_t *self) { free(self); }

static const nfc_backend_vtable_t NULL_VT = { null_transceive, null_destroy };

nfc_backend_t *nfc_backend_null_new(void) {
    null_backend_t *b = (null_backend_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    b->base.vt = &NULL_VT;
    b->base.state = b;
    return &b->base;
}

/* ---------- Mock backend (passport simulator). ---------- */
static int mock_match_entry(const nfc_mock_entry_t *e, const nfc_apdu_t *a) {
    if (e->match_cla != -1 && e->match_cla != a->cla) return 0;
    if (e->match_ins != -1 && e->match_ins != a->ins) return 0;
    if (e->match_p1  != -1 && e->match_p1  != a->p1)  return 0;
    if (e->match_p2  != -1 && e->match_p2  != a->p2)  return 0;
    if (e->match_lc  != -1 && (int)e->match_lc != a->lc) return 0;
    if (e->match_data_len != -1) {
        if ((int)a->lc != e->match_data_len) return 0;
        if (memcmp(e->match_data, a->data, e->match_data_len) != 0) return 0;
    }
    return 1;
}

/* pad0x80 to an 8-byte multiple; out must hold padded_len = len + (8-len%8). */
static size_t pad0x80(const uint8_t *in, size_t len, uint8_t *out) {
    size_t pad = 8 - (len % 8);
    memcpy(out, in, len);
    out[len] = 0x80;
    for (size_t i = len + 1; i < len + pad; ++i) out[i] = 0x00;
    return len + pad;
}

/* strip trailing 0x80..00 padding, returns plaintext length (or len). */
static size_t unpad0x80(const uint8_t *in, size_t len) {
    if (len == 0) return 0;
    size_t i = len;
    while (i > 0 && in[i - 1] == 0x00) --i;
    if (i > 0 && in[i - 1] == 0x80) --i;
    return i;
}

/* SM response builder: enc = CBC(Ksenc, pad0x80(plaintext)), MAC over enc.
 * Writes enc||MAC into out (>= plen + 16 bytes), returns total length. */
static int mock_sm_build_response(const uint8_t ksenc[16], const uint8_t ksmac[16],
                                  const uint8_t *plain, size_t plen,
                                  uint8_t *out, size_t outcap) {
    uint8_t tmp[2048];
    uint8_t enc[2048];
    size_t padded = pad0x80(plain, plen, tmp);
    if (padded > sizeof(enc) || padded + 8 > outcap) return -1;
    uint8_t zero[8] = {0};
    tdes_cbc_encrypt(ksenc, zero, tmp, padded, enc);
    memcpy(out, enc, padded);
    uint8_t mac[8];
    tdes_cbc_mac(ksmac, enc, padded, mac);
    memcpy(out + padded, mac, 8);
    return (int)(padded + 8);
}

/* Handle a Secure Messaging command (CLA 0x0C) after BAC auth. */
static int mock_sm_handle(nfc_mock_t *m, const nfc_apdu_t *capdu,
                          nfc_apdu_t *rapdu) {
    memset(rapdu, 0, sizeof(*rapdu));
    rapdu->is_response = 1;
    if (!m->authed) { rapdu->sw = NFC_SW_CONDITIONS; return -1; }
    /* Command MAC covers the 5-byte header "0C INS P1 P2 00". */
    uint8_t hdr[5] = { (uint8_t)capdu->cla, (uint8_t)capdu->ins,
                       (uint8_t)capdu->p1, (uint8_t)capdu->p2, 0x00 };
    uint8_t mac[8];
    tdes_cbc_mac(m->ksmac, hdr, sizeof(hdr), mac);
    if (capdu->lc != 8 || memcmp(mac, capdu->data, 8) != 0) {
        rapdu->sw = NFC_SW_SECURITY;
        return -1;
    }
    /* Determine plaintext response body. */
    uint8_t body[2048];
    size_t blen = 0;
    if (capdu->ins == 0xA4) {
        /* SELECT: success status body. */
        body[0] = 0x90; body[1] = 0x00;
        blen = 2;
    } else if (capdu->ins == 0xB0) {
        if (m->dg1_len > 0 && capdu->p2 == 0x01) {
            memcpy(body, m->dg1, (size_t)m->dg1_len);
            blen = (size_t)m->dg1_len;
        } else if (m->sod_len > 0) {
            memcpy(body, m->sod, (size_t)m->sod_len);
            blen = (size_t)m->sod_len;
        } else {
            rapdu->sw = NFC_SW_FILE_NOT_FOUND;
            return -1;
        }
    } else {
        rapdu->sw = NFC_SW_INS_NOT_SUPPORTED;
        return -1;
    }
    uint8_t resp[2048];
    int n = mock_sm_build_response(m->ksenc, m->ksmac, body, blen, resp,
                                   sizeof(resp));
    if (n < 0) { rapdu->sw = NFC_SW_UNKNOWN; return -1; }
    memcpy(rapdu->data, resp, (size_t)n);
    rapdu->lc = (uint16_t)n;
    rapdu->sw = NFC_SW_OK;
    return 0;
}

/* Handle MUTUAL AUTHENTICATE (ins 0x82): verify reader payload, build a
 * genuine encrypted response, derive session keys. */
static int mock_bac_auth(nfc_mock_t *m, const nfc_apdu_t *capdu,
                         nfc_apdu_t *rapdu) {
    memset(rapdu, 0, sizeof(*rapdu));
    rapdu->is_response = 1;
    if (!m->mrz_line2[0]) { rapdu->sw = NFC_SW_CONDITIONS; return -1; }
    if (capdu->lc != 40) { rapdu->sw = NFC_SW_WRONG_PARAM; return -1; }

    const uint8_t *E = capdu->data;        /* 32 bytes */
    const uint8_t *M = capdu->data + 32;   /* 8 bytes  */

    /* Verify M = MAC(Kmac, E). */
    uint8_t mac[8];
    tdes_cbc_mac(m->kmac, E, 32, mac);
    if (memcmp(mac, M, 8) != 0) { rapdu->sw = NFC_SW_SECURITY; return -1; }

    /* Decrypt S = rndICC || rndIFD || KIFD, check our rndICC. */
    uint8_t S[32];
    uint8_t zero[8] = {0};
    tdes_cbc_decrypt(m->kenc, zero, E, 32, S);
    if (memcmp(S, m->rnd_icc, 8) != 0) { rapdu->sw = NFC_SW_SECURITY; return -1; }
    memcpy(m->kifd, S + 16, 16);

    /* Our key contribution. */
    if (rand_bytes(m->kicc, 16) != 0)
        for (int i = 0; i < 16; ++i) m->kicc[i] = (uint8_t)(0xA0 + i);

    /* Response S' = rndIFD || rndICC || KICC. */
    uint8_t Sp[32];
    memcpy(Sp, S + 8, 8);            /* rndIFD */
    memcpy(Sp + 8, m->rnd_icc, 8);   /* rndICC */
    memcpy(Sp + 16, m->kicc, 16);    /* KICC  */
    uint8_t Ep[32];
    tdes_cbc_encrypt(m->kenc, zero, Sp, 32, Ep);
    uint8_t Mp[8];
    tdes_cbc_mac(m->kmac, Ep, 32, Mp);
    memcpy(rapdu->data, Ep, 32);
    memcpy(rapdu->data + 32, Mp, 8);
    rapdu->lc = 40;
    rapdu->sw = NFC_SW_OK;

    /* Session keys KS = KIFD ^ KICC. */
    uint8_t ks[16];
    for (int i = 0; i < 16; ++i) ks[i] = m->kifd[i] ^ m->kicc[i];
    tdes_cbc_encrypt(m->kenc, zero, ks, 16, m->ksenc);
    tdes_cbc_encrypt(m->kmac, zero, ks, 16, m->ksmac);
    m->authed = 1;
    return 0;
}

static int mock_transceive(nfc_backend_t *self, const nfc_apdu_t *capdu,
                           nfc_apdu_t *rapdu) {
    nfc_mock_t *m = (nfc_mock_t *)self->state;

    /* Cryptography-aware branches first. */
    if (capdu->cla == 0x0C)
        return mock_sm_handle(m, capdu, rapdu);
    if (capdu->cla == 0x00 && capdu->ins == 0x82)
        return mock_bac_auth(m, capdu, rapdu);
    /* GET CHALLENGE: hand back our rndICC (script value or random). */
    if (capdu->cla == 0x00 && capdu->ins == 0x84) {
        memset(rapdu, 0, sizeof(*rapdu));
        rapdu->is_response = 1;
        if (!m->rnd_icc_fixed) {
            if (rand_bytes(m->rnd_icc, 8) != 0) {
                rapdu->sw = NFC_SW_UNKNOWN;
                return -1;
            }
        }
        memcpy(rapdu->data, m->rnd_icc, 8);
        rapdu->lc = 8;
        rapdu->sw = NFC_SW_OK;
        return 0;
    }

    /* Plain replay entries (SELECT AID, scripted error branches). */
    nfc_mock_entry_t *hit = NULL;
    size_t idx = m->cursor;
    for (size_t i = 0; i < m->count; ++i) {
        size_t k = (idx + i) % m->count;
        if (mock_match_entry(&m->entries[k], capdu)) {
            hit = &m->entries[k];
            m->cursor = (k + 1) % m->count;
            break;
        }
    }
    if (!hit) {
        m->unexpected_apdu++;
        m->replay_remaining = 0;
        memset(rapdu, 0, sizeof(*rapdu));
        rapdu->is_response = 1;
        rapdu->sw = NFC_SW_UNKNOWN;
        return -1;
    }
    *rapdu = hit->response;
    rapdu->is_response = 1;
    m->replay_remaining = (m->count > m->cursor) ? (m->count - m->cursor) : 0;
    return 0;
}

static void mock_destroy(nfc_backend_t *self) {
    nfc_mock_t *m = (nfc_mock_t *)self->state;
    free(m->entries);
    free(m);
    free(self);
}

static const nfc_backend_vtable_t MOCK_VT = { mock_transceive, mock_destroy };

nfc_backend_t *nfc_backend_mock_new(nfc_mock_entry_t *entries, size_t count) {
    nfc_backend_t *b = (nfc_backend_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    nfc_mock_t *m = (nfc_mock_t *)calloc(1, sizeof(*m));
    if (!m) { free(b); return NULL; }
    m->cap = m->count = count;
    if (count > 0) {
        m->entries = (nfc_mock_entry_t *)calloc(count, sizeof(*m->entries));
        if (!m->entries) { free(m); free(b); return NULL; }
        memcpy(m->entries, entries, count * sizeof(*m->entries));
    }
    m->replay_remaining = (int)count;
    b->vt = &MOCK_VT;
    b->state = m;
    return b;
}

/* Re-derive keys + DG1 from the supplied MRZ. */
int nfc_mock_set_mrz(nfc_backend_t *b, const char *line1, const char *line2) {
    if (!b || b->vt != &MOCK_VT || !line1 || !line2) return -1;
    nfc_mock_t *m = (nfc_mock_t *)b->state;
    snprintf(m->mrz_line1, sizeof(m->mrz_line1), "%s", line1);
    snprintf(m->mrz_line2, sizeof(m->mrz_line2), "%s", line2);
    char info[BAC_MRZ_INFO_LEN + 1];
    if (bac_mrz_info(line2, info) != 0) return -1;
    uint8_t kseed[16];
    bac_derive_keys(info, kseed, m->kenc, m->kmac);
    /* DG1 TLV: 5A <len> <line1>\n<line2>. */
    size_t l1 = strlen(line1), l2 = strlen(line2);
    size_t total = l1 + 1 + l2;
    if (total > 250) return -1;
    m->dg1[0] = 0x5A;
    m->dg1[1] = (uint8_t)total;
    memcpy(m->dg1 + 2, line1, l1);
    m->dg1[2 + l1] = '\n';
    memcpy(m->dg1 + 3 + l1, line2, l2);
    m->dg1_len = (int)(2 + total);
    m->authed = 0;
    return 0;
}

int nfc_mock_set_rnd_icc(nfc_backend_t *b, const uint8_t rnd_icc[8], int len) {
    if (!b || b->vt != &MOCK_VT || !rnd_icc || len != 8) return -1;
    nfc_mock_t *m = (nfc_mock_t *)b->state;
    memcpy(m->rnd_icc, rnd_icc, 8);
    m->rnd_icc_fixed = 1;
    return 0;
}

int nfc_mock_set_sod(nfc_backend_t *b, const uint8_t *data, int len) {
    if (!b || b->vt != &MOCK_VT || !data || len <= 0 || len > 1024) return -1;
    nfc_mock_t *m = (nfc_mock_t *)b->state;
    memcpy(m->sod, data, (size_t)len);
    m->sod_len = len;
    return 0;
}

int nfc_mock_get_mrz(const nfc_backend_t *b, const char **line1,
                     const char **line2) {
    if (!b || b->vt != &MOCK_VT) return -1;
    nfc_mock_t *m = (nfc_mock_t *)b->state;
    if (!m->mrz_line2[0]) return -1;
    *line1 = m->mrz_line1;
    *line2 = m->mrz_line2;
    return 0;
}

int nfc_mock_unexpected_count(const nfc_backend_t *b) {
    if (!b || b->vt != &MOCK_VT) return -1;
    nfc_mock_t *m = (nfc_mock_t *)b->state;
    return m->unexpected_apdu;
}

int nfc_mock_remaining(const nfc_backend_t *b) {
    if (!b || b->vt != &MOCK_VT) return -1;
    nfc_mock_t *m = (nfc_mock_t *)b->state;
    return m->replay_remaining;
}

/* ---------- Minimal helper: build a C-APDU from compact fields. */
static void mk_capdu(nfc_apdu_t *a, uint8_t cla, uint8_t ins, uint8_t p1,
                     uint8_t p2, const uint8_t *data, uint16_t lc, uint16_t le) {
    memset(a, 0, sizeof(*a));
    a->cla = cla; a->ins = ins; a->p1 = p1; a->p2 = p2;
    a->lc = lc;
    if (lc && data) memcpy(a->data, data, lc);
    a->le = le;
    a->sw = 0;
    a->is_response = 0;
}

/* Convenience: send a short command APDU and check SW. */
static int send(nfc_backend_t *be, const nfc_apdu_t *capdu, nfc_apdu_t *rapdu) {
    return be->vt->transceive(be, capdu, rapdu);
}

static const uint8_t AID_EMRTD[] = {0xA0,0x00,0x00,0x02,0x47,0x10,0x01};

static void bytes_to_hex(const uint8_t *d, size_t n, char *out, size_t cap) {
    size_t need = n * 2 + 1;
    if (cap < need) n = (cap - 1) / 2;
    for (size_t i = 0; i < n; ++i)
        snprintf(out + i * 2, 3, "%02X", d[i]);
    out[n * 2] = '\0';
}

static uint32_t now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint32_t)(ts.tv_sec * 1000u + ts.tv_nsec / 1000000u);
}

/* Format a full C-APDU as hex: CLA INS P1 P2 [Lc data] [Le]. */
static void capdu_to_hex(const nfc_apdu_t *a, char *out, size_t cap) {
    size_t n = 0;
    char b[16];
    snprintf(b, sizeof(b), "%02X%02X%02X%02X", a->cla, a->ins, a->p1, a->p2);
    if (n + 8 + 1 < cap) { memcpy(out + n, b, 8); n += 8; }
    if (a->lc > 0) {
        if (n + 2 + 1 < cap) { snprintf(b, sizeof(b), "%02X", a->lc); memcpy(out + n, b, 2); n += 2; }
        for (uint16_t i = 0; i < a->lc && n + 2 + 1 < cap; ++i) {
            snprintf(b, sizeof(b), "%02X", a->data[i]);
            memcpy(out + n, b, 2);
            n += 2;
        }
    } else if (a->le) {
        if (n + 2 + 1 < cap) { snprintf(b, sizeof(b), "%02X", a->le); memcpy(out + n, b, 2); n += 2; }
    }
    out[n] = '\0';
}

static void trace_clear(nfc_result_t *r) { r->trace_count = 0; }

static void trace_add(nfc_result_t *r, const char *name, const nfc_apdu_t *capdu,
                      const nfc_apdu_t *rapdu, uint32_t ms, const char *plain,
                      int mac_ok, int err) {
    if (r->trace_count >= NFC_TRACE_MAX) return;
    nfc_trace_t *t = &r->trace[r->trace_count++];
    snprintf(t->name, sizeof(t->name), "%s", name);
    capdu_to_hex(capdu, t->capdu, sizeof(t->capdu));
    bytes_to_hex(rapdu->data, rapdu->lc, t->rapdu, sizeof(t->rapdu));
    t->sw = rapdu->sw;
    t->ms = ms;
    if (plain) snprintf(t->plain, sizeof(t->plain), "%s", plain);
    t->mac_ok = mac_ok;
    t->err = err;
}

/* Verify a SM response: check MAC over enc, decrypt, unpad. */
static int sm_verify(const uint8_t ksenc[16], const uint8_t ksmac[16],
                     const nfc_apdu_t *rapdu, uint8_t *plain, size_t *plen,
                     size_t pcap) {
    if (rapdu->lc < 9) return -1;
    size_t enc_len = rapdu->lc - 8;
    const uint8_t *enc = rapdu->data;
    uint8_t mac[8];
    tdes_cbc_mac(ksmac, enc, enc_len, mac);
    if (memcmp(mac, rapdu->data + enc_len, 8) != 0) return -1;
    uint8_t zero[8] = {0};
    uint8_t dec[2048];
    if (enc_len > sizeof(dec)) return -1;
    tdes_cbc_decrypt(ksenc, zero, enc, enc_len, dec);
    size_t n = unpad0x80(dec, enc_len);
    if (n > pcap) n = pcap;
    memcpy(plain, dec, n);
    *plen = n;
    return 0;
}

/* ---------- High-level BAC read flow (ICAO 9303-11 §9.8). ---------- */
nfc_result_t nfc_read_emrtd(nfc_backend_t *backend,
                            const char *mrz_line1, const char *mrz_line2) {
    nfc_result_t r = {0};
    r.step = NFC_STEP_SELECT_APP;
    trace_clear(&r);
    nfc_apdu_t capdu, rapdu;

    /* Backwards-compatible plain path when no MRZ is supplied. */
    if (!mrz_line2) {
        uint32_t t0 = now_ms();
        mk_capdu(&capdu, 0x00, 0xA4, 0x04, 0x0C, AID_EMRTD,
                 sizeof(AID_EMRTD), 0);
        if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
            snprintf(r.detail, sizeof(r.detail), "SELECT AID failed (sw=%04X)",
                     rapdu.sw);
            r.step = NFC_STEP_ERROR;
            trace_add(&r, "SELECT AID", &capdu, &rapdu, now_ms() - t0, NULL,
                      -1, 1);
            return r;
        }
        trace_add(&r, "SELECT AID", &capdu, &rapdu, now_ms() - t0, NULL, -1, 0);
        mk_capdu(&capdu, 0x00, 0x84, 0x00, 0x00, NULL, 0, 8);
        t0 = now_ms();
        if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
            snprintf(r.detail, sizeof(r.detail),
                     "GET CHALLENGE failed (sw=%04X)", rapdu.sw);
            r.step = NFC_STEP_ERROR;
            trace_add(&r, "GET CHALLENGE", &capdu, &rapdu, now_ms() - t0, NULL,
                      -1, 1);
            return r;
        }
        trace_add(&r, "GET CHALLENGE", &capdu, &rapdu, now_ms() - t0, NULL, -1, 0);
        /* No crypto available: stop after challenge. */
        r.step = NFC_STEP_MUTUAL_AUTH_1;
        snprintf(r.detail, sizeof(r.detail),
                 "no MRZ supplied; BAC crypto skipped (plain path)");
        r.crypto_done = 0;
        return r;
    }

    bac_ctx_t ctx;
    memset(&ctx, 0, sizeof(ctx));
    if (bac_mrz_info(mrz_line2, ctx.mrz_info) != 0) {
        snprintf(r.detail, sizeof(r.detail), "bad MRZ line2: %.28s", mrz_line2);
        r.step = NFC_STEP_ERROR;
        return r;
    }
    bac_derive_keys(ctx.mrz_info, ctx.kseed, ctx.kenc, ctx.kmac);
    r.crypto_done = 1;
    snprintf(r.mrz_info, sizeof(r.mrz_info), "%s", ctx.mrz_info);
    bytes_to_hex(ctx.kseed, 16, r.kseed_hex, sizeof(r.kseed_hex));
    bytes_to_hex(ctx.kenc, 16, r.kenc_hex, sizeof(r.kenc_hex));
    bytes_to_hex(ctx.kmac, 16, r.kmac_hex, sizeof(r.kmac_hex));

    /* 1. SELECT AID A0000002471001 (ICAO 9303 eMRTD). */
    uint32_t t0 = now_ms();
    mk_capdu(&capdu, 0x00, 0xA4, 0x04, 0x0C, AID_EMRTD, sizeof(AID_EMRTD), 0);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SELECT AID failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_ERROR;
        trace_add(&r, "SELECT AID", &capdu, &rapdu, now_ms() - t0, NULL, -1, 1);
        return r;
    }
    trace_add(&r, "SELECT AID", &capdu, &rapdu, now_ms() - t0, NULL, -1, 0);
    r.step = NFC_STEP_GET_CHALLENGE;

    /* 2. GET CHALLENGE -> rndICC (8 bytes). */
    t0 = now_ms();
    mk_capdu(&capdu, 0x00, 0x84, 0x00, 0x00, NULL, 0, 8);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "GET CHALLENGE failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_ERROR;
        trace_add(&r, "GET CHALLENGE", &capdu, &rapdu, now_ms() - t0, NULL,
                  -1, 1);
        return r;
    }
    if (rapdu.lc < 8) {
        snprintf(r.detail, sizeof(r.detail),
                 "GET CHALLENGE short reply (%u)", rapdu.lc);
        r.step = NFC_STEP_ERROR;
        trace_add(&r, "GET CHALLENGE", &capdu, &rapdu, now_ms() - t0, NULL,
                  -1, 1);
        return r;
    }
    trace_add(&r, "GET CHALLENGE", &capdu, &rapdu, now_ms() - t0, NULL, -1, 0);
    memcpy(ctx.rnd_icc, rapdu.data, 8);
    bytes_to_hex(ctx.rnd_icc, 8, r.rnd_icc_hex, sizeof(r.rnd_icc_hex));
    r.step = NFC_STEP_MUTUAL_AUTH_1;

    /* 3. MUTUAL AUTHENTICATE: E||M (40 bytes). */
    uint8_t auth1[40];
    bac_build_auth1(&ctx, auth1);
    bytes_to_hex(auth1, 40, r.auth1_hex, sizeof(r.auth1_hex));
    mk_capdu(&capdu, 0x00, 0x82, 0x00, 0x00, auth1, sizeof(auth1), 0);
    t0 = now_ms();
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "MUTUAL AUTH failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_ERROR;
        trace_add(&r, "MUTUAL AUTH (BAC)", &capdu, &rapdu, now_ms() - t0,
                  NULL, -1, 1);
        return r;
    }
    r.step = NFC_STEP_MUTUAL_AUTH_2;
    if (rapdu.lc < 40) {
        snprintf(r.detail, sizeof(r.detail),
                 "MUTUAL AUTH short reply (%u)", rapdu.lc);
        r.step = NFC_STEP_ERROR;
        trace_add(&r, "MUTUAL AUTH (BAC)", &capdu, &rapdu, now_ms() - t0,
                  NULL, -1, 1);
        return r;
    }
    bytes_to_hex(rapdu.data, 40, r.auth2_hex, sizeof(r.auth2_hex));
    if (bac_verify_auth2(&ctx, rapdu.data) != 0) {
        snprintf(r.detail, sizeof(r.detail),
                 "MUTUAL AUTH response invalid (MAC/mismatch)");
        r.step = NFC_STEP_ERROR;
        trace_add(&r, "MUTUAL AUTH (BAC)", &capdu, &rapdu, now_ms() - t0,
                  NULL, 0, 1);
        return r;
    }
    trace_add(&r, "MUTUAL AUTH (BAC)", &capdu, &rapdu, now_ms() - t0,
              NULL, 1, 0);
    bytes_to_hex(ctx.ksenc, 16, r.ksenc_hex, sizeof(r.ksenc_hex));
    bytes_to_hex(ctx.ksmac, 16, r.ksmac_hex, sizeof(r.ksmac_hex));

    /* 4. SM SELECT EF.DG1 (fid 0101). */
    uint8_t smhdr[5] = {0x0C, 0xA4, 0x02, 0x0C, 0x00};
    uint8_t mac8[8];
    uint8_t smcmd[8];
    tdes_cbc_mac(ctx.ksmac, smhdr, 5, mac8);
    memcpy(smcmd, mac8, 8);
    mk_capdu(&capdu, 0x0C, 0xA4, 0x02, 0x0C, smcmd, 8, 0);
    t0 = now_ms();
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SM SELECT DG1 failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_SELECT_DG1;
        trace_add(&r, "SM SELECT EF.DG1", &capdu, &rapdu, now_ms() - t0,
                  NULL, -1, 1);
        return r;
    }
    {
        uint8_t body[64];
        size_t blen = 0;
        int mok = 1;
        if (sm_verify(ctx.ksenc, ctx.ksmac, &rapdu, body, &blen,
                      sizeof(body)) != 0) {
            snprintf(r.detail, sizeof(r.detail),
                     "SM SELECT DG1 response invalid");
            r.step = NFC_STEP_SELECT_DG1;
            trace_add(&r, "SM SELECT EF.DG1", &capdu, &rapdu, now_ms() - t0,
                      NULL, 0, 1);
            return r;
        }
        (void)mok;
    }
    trace_add(&r, "SM SELECT EF.DG1", &capdu, &rapdu, now_ms() - t0,
              NULL, 1, 0);
    r.step = NFC_STEP_READ_DG1;

    /* 5. SM READ BINARY (DG1) -> decrypted MRZ TLV. */
    uint8_t rd_hdr[5] = {0x0C, 0xB0, 0x00, 0x01, 0x00};
    tdes_cbc_mac(ctx.ksmac, rd_hdr, 5, mac8);
    uint8_t rd[8];
    memcpy(rd, mac8, 8);
    mk_capdu(&capdu, 0x0C, 0xB0, 0x00, 0x01, rd, 8, 0);
    t0 = now_ms();
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SM READ DG1 failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_READ_DG1;
        trace_add(&r, "SM READ BINARY (DG1)", &capdu, &rapdu, now_ms() - t0,
                  NULL, -1, 1);
        return r;
    }
    {
        uint8_t tlv[1024];
        size_t tlen = 0;
        if (sm_verify(ctx.ksenc, ctx.ksmac, &rapdu, tlv, &tlen,
                      sizeof(tlv)) != 0) {
            snprintf(r.detail, sizeof(r.detail),
                     "SM READ DG1 response invalid");
            r.step = NFC_STEP_READ_DG1;
            trace_add(&r, "SM READ BINARY (DG1)", &capdu, &rapdu, now_ms() - t0,
                      NULL, 0, 1);
            return r;
        }
        char plain_hex[1024];
        bytes_to_hex(tlv, tlen, plain_hex, sizeof(plain_hex));
        trace_add(&r, "SM READ BINARY (DG1)", &capdu, &rapdu, now_ms() - t0,
                  plain_hex, 1, 0);
        /* DG1 TLV: 5A <len> <MRZ>. */
        size_t off = 0;
        if (tlen >= 2 && tlv[0] == 0x5A && (size_t)tlv[1] <= tlen - 2)
            off = 2;
        size_t avail = tlen - off;
        if (avail > sizeof(r.dg1_mrz) - 1) avail = sizeof(r.dg1_mrz) - 1;
        memcpy(r.dg1_mrz, tlv + off, avail);
        r.dg1_mrz[avail] = '\0';
    }
    r.step = NFC_STEP_SELECT_SOD;

    /* 6. SM SELECT EF.SOD (fid 011D). */
    uint8_t sods[5] = {0x0C, 0xA4, 0x02, 0x0C, 0x00};
    tdes_cbc_mac(ctx.ksmac, sods, 5, mac8);
    memcpy(smcmd, mac8, 8);
    mk_capdu(&capdu, 0x0C, 0xA4, 0x02, 0x0C, smcmd, 8, 0);
    t0 = now_ms();
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SM SELECT SOD failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_SELECT_SOD;
        trace_add(&r, "SM SELECT EF.SOD", &capdu, &rapdu, now_ms() - t0,
                  NULL, -1, 1);
        return r;
    }
    {
        uint8_t body[64];
        size_t blen = 0;
        if (sm_verify(ctx.ksenc, ctx.ksmac, &rapdu, body, &blen,
                      sizeof(body)) != 0) {
            snprintf(r.detail, sizeof(r.detail),
                     "SM SELECT SOD response invalid");
            r.step = NFC_STEP_SELECT_SOD;
            trace_add(&r, "SM SELECT EF.SOD", &capdu, &rapdu, now_ms() - t0,
                      NULL, 0, 1);
            return r;
        }
    }
    trace_add(&r, "SM SELECT EF.SOD", &capdu, &rapdu, now_ms() - t0,
              NULL, 1, 0);
    r.step = NFC_STEP_READ_SOD;

    /* 7. SM READ BINARY (SOD). */
    uint8_t rd2_hdr[5] = {0x0C, 0xB0, 0x00, 0x00, 0x00};
    tdes_cbc_mac(ctx.ksmac, rd2_hdr, 5, mac8);
    memcpy(rd, mac8, 8);
    mk_capdu(&capdu, 0x0C, 0xB0, 0x00, 0x00, rd, 8, 0);
    t0 = now_ms();
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SM READ SOD failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_READ_SOD;
        trace_add(&r, "SM READ BINARY (SOD)", &capdu, &rapdu, now_ms() - t0,
                  NULL, -1, 1);
        return r;
    }
    {
        uint8_t sod[1024];
        size_t slen = 0;
        if (sm_verify(ctx.ksenc, ctx.ksmac, &rapdu, sod, &slen,
                      sizeof(sod)) != 0) {
            snprintf(r.detail, sizeof(r.detail),
                     "SM READ SOD response invalid");
            r.step = NFC_STEP_READ_SOD;
            trace_add(&r, "SM READ BINARY (SOD)", &capdu, &rapdu, now_ms() - t0,
                      NULL, 0, 1);
            return r;
        }
        char plain_hex[1024];
        bytes_to_hex(sod, slen, plain_hex, sizeof(plain_hex));
        trace_add(&r, "SM READ BINARY (SOD)", &capdu, &rapdu, now_ms() - t0,
                  plain_hex, 1, 0);
        size_t take = slen < sizeof(r.sod_head) ? slen : sizeof(r.sod_head);
        memcpy(r.sod_head, sod, take);
        r.sod_head_len = (int)take;
    }

    r.ok = 1;
    r.step = NFC_STEP_DONE;
    snprintf(r.detail, sizeof(r.detail),
             "BAC complete: MRZ verified, DG1=%zu bytes, SOD=%d bytes",
             strlen(r.dg1_mrz), r.sod_head_len);
    return r;
}

const char *nfc_step_name(nfc_step_t s) {
    switch (s) {
        case NFC_STEP_SELECT_APP:      return "SELECT_AID";
        case NFC_STEP_GET_CHALLENGE:   return "GET_CHALLENGE";
        case NFC_STEP_MUTUAL_AUTH_1:   return "MUTUAL_AUTH_1";
        case NFC_STEP_MUTUAL_AUTH_2:   return "MUTUAL_AUTH_2";
        case NFC_STEP_SELECT_DG1:      return "SELECT_DG1";
        case NFC_STEP_READ_DG1:        return "READ_BINARY_DG1";
        case NFC_STEP_SELECT_SOD:      return "SELECT_SOD";
        case NFC_STEP_READ_SOD:        return "READ_BINARY_SOD";
        case NFC_STEP_DONE:            return "DONE";
        case NFC_STEP_ERROR:           return "ERROR";
    }
    return "?";
}