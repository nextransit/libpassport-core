/* NFC backend abstraction + high-level eMRTD BAC read flow. */
#include "nfc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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

/* ---------- Mock backend. ---------- */
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

static int mock_transceive(nfc_backend_t *self, const nfc_apdu_t *capdu,
                           nfc_apdu_t *rapdu) {
    nfc_mock_t *m = (nfc_mock_t *)self->state;
    /* Find the next matching entry (default: any matching pattern). */
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
}

static const nfc_backend_vtable_t MOCK_VT = { mock_transceive, mock_destroy };

nfc_backend_t *nfc_backend_mock_new(nfc_mock_entry_t *entries, size_t count) {
    if (count == 0 || !entries) return NULL;
    nfc_backend_t *b = (nfc_backend_t *)calloc(1, sizeof(*b));
    if (!b) return NULL;
    nfc_mock_t *m = (nfc_mock_t *)calloc(1, sizeof(*m));
    if (!m) { free(b); return NULL; }
    m->cap = m->count = count;
    m->entries = (nfc_mock_entry_t *)calloc(count, sizeof(*m->entries));
    if (!m->entries) { free(m); free(b); return NULL; }
    memcpy(m->entries, entries, count * sizeof(*m->entries));
    m->replay_remaining = (int)count;
    b->vt = &MOCK_VT;
    b->state = m;
    return b;
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

/* ---------- High-level BAC read flow.
 *
 * Real BAC requires DES key derivation from MRZ. We simulate that by
 * carrying a static session key inside the mock script, which the test
 * author sets to whatever they want; the in-flight encryption is not
 * what we're testing here -- we test the APDU choreography and the
 * error-handling branches. */

static const uint8_t AID_EMRTD[] = {0xA0,0x00,0x00,0x02,0x47,0x10,0x01};

nfc_result_t nfc_read_emrtd(nfc_backend_t *backend) {
    nfc_result_t r = {0};
    r.ok = 0;
    r.step = NFC_STEP_SELECT_APP;

    nfc_apdu_t capdu, rapdu;

    /* 1. SELECT AID A0000002471001 (ICAO 9303 eMRTD). */
    mk_capdu(&capdu, 0x00, 0xA4, 0x04, 0x0C, AID_EMRTD, sizeof(AID_EMRTD), 0);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SELECT AID failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_ERROR;
        return r;
    }
    r.step = NFC_STEP_GET_CHALLENGE;

    /* 2. GET CHALLENGE (initiates BAC). */
    mk_capdu(&capdu, 0x00, 0x84, 0x00, 0x00, NULL, 0, 8);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "GET CHALLENGE failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_ERROR;
        return r;
    }
    r.step = NFC_STEP_MUTUAL_AUTH_1;

    /* 3. MUTUAL AUTHENTICATE command (sends encrypted rndICC||rndIFD||KIFD). */
    uint8_t ma1[32] = {0};
    mk_capdu(&capdu, 0x00, 0x82, 0x00, 0x00, ma1, sizeof(ma1), 0);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "MUTUAL AUTH 1 failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_ERROR;
        return r;
    }
    r.step = NFC_STEP_MUTUAL_AUTH_2;

    /* 4. MUTUAL AUTHENTICATE response (sends encrypted rndIFD||rndICC||KICC). */
    uint8_t ma2[32] = {0};
    mk_capdu(&capdu, 0x00, 0x82, 0x00, 0x00, ma2, sizeof(ma2), 0);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "MUTUAL AUTH 2 failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_ERROR;
        return r;
    }

    /* 5. SELECT EF.DG1 (file id 0101). */
    uint8_t fid_dg1[2] = {0x01, 0x01};
    mk_capdu(&capdu, 0x00, 0xA4, 0x02, 0x0C, fid_dg1, 2, 0);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SELECT DG1 failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_SELECT_DG1;
        return r;
    }
    r.step = NFC_STEP_READ_DG1;

    /* 6. READ BINARY for DG1 (assume 128-byte payload). */
    mk_capdu(&capdu, 0x00, 0xB0, 0x00, 0x00, NULL, 0, 0x80);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "READ BINARY DG1 failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_READ_DG1;
        return r;
    }
    /* DG1 payload layout: tag 5A (MRZ) wrapped in TLV. For the mock
     * we accept either raw MRZ or a 0x5A 0x5F TLV with the MRZ. */
    {
        size_t n = rapdu.data[0]; /* If Le=0x80 returned a full frame... */
        size_t off = 0;
        if (rapdu.lc > 0 && rapdu.data[0] == 0x5A) {
            off = 2;
        }
        size_t avail = rapdu.lc > off ? rapdu.lc - off : 0;
        if (avail > sizeof(r.dg1_mrz) - 1) avail = sizeof(r.dg1_mrz) - 1;
        memcpy(r.dg1_mrz, rapdu.data + off, avail);
        r.dg1_mrz[avail] = '\0';
        (void)n;
    }

    /* 7. SELECT EF.SOD (file id 011D). */
    uint8_t fid_sod[2] = {0x01, 0x1D};
    mk_capdu(&capdu, 0x00, 0xA4, 0x02, 0x0C, fid_sod, 2, 0);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "SELECT SOD failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_SELECT_SOD;
        return r;
    }
    r.step = NFC_STEP_READ_SOD;

    /* 8. READ BINARY for SOD. */
    mk_capdu(&capdu, 0x00, 0xB0, 0x00, 0x00, NULL, 0, 0x80);
    if (send(backend, &capdu, &rapdu) != 0 || rapdu.sw != NFC_SW_OK) {
        snprintf(r.detail, sizeof(r.detail),
                 "READ BINARY SOD failed (sw=%04X)", rapdu.sw);
        r.step = NFC_STEP_READ_SOD;
        return r;
    }
    size_t take = rapdu.lc < sizeof(r.sod_head) ? rapdu.lc : sizeof(r.sod_head);
    memcpy(r.sod_head, rapdu.data, take);
    r.sod_head_len = (int)take;

    r.ok = 1;
    r.step = NFC_STEP_DONE;
    snprintf(r.detail, sizeof(r.detail),
             "BAC happy path complete; DG1=%zu bytes; SOD=%d bytes",
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
