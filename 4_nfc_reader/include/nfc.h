#ifndef NFC_H
#define NFC_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ISO 7816 / ICAO 9303 status words. */
#define NFC_SW_OK                 0x9000
#define NFC_SW_FILE_NOT_FOUND     0x6A82
#define NFC_SW_WRONG_PARAM        0x6A86
#define NFC_SW_SECURITY           0x6982
#define NFC_SW_CONDITIONS         0x6985
#define NFC_SW_INS_NOT_SUPPORTED  0x6D00
#define NFC_SW_CLA_NOT_SUPPORTED  0x6E00
#define NFC_SW_UNKNOWN            0x6F00

/* Buffer holding a single command or response APDU. */
typedef struct {
    uint8_t  cla;
    uint8_t  ins;
    uint8_t  p1;
    uint8_t  p2;
    uint16_t lc;
    uint8_t  data[1024];
    uint16_t le;     /* 0x0000 if no Le */
    uint16_t sw;     /* status word for response; 0 for command */
    int      is_response; /* 1 for R-APDU, 0 for C-APDU */
} nfc_apdu_t;

/* Backend abstraction: swap real NFC reader for mock recording. */
typedef struct nfc_backend nfc_backend_t;
typedef struct {
    int (*transceive)(nfc_backend_t *self, const nfc_apdu_t *capdu,
                      nfc_apdu_t *rapdu);
    void (*destroy)(nfc_backend_t *self);
} nfc_backend_vtable_t;

struct nfc_backend {
    const nfc_backend_vtable_t *vt;
    void *state;
};

/* ----- Real-ish backend that talks to /dev/none (always returns 6F00).
 * Useful as a placeholder. */
nfc_backend_t *nfc_backend_null_new(void);

/* ----- Mock backend that replays a JSON script of C-APDU -> R-APDU
 * pairs. Each entry may optionally assert that the incoming C-APDU
 * matches a pattern. */
typedef struct {
    /* Expected fields (use -1 to mean "any"). */
    int     match_cla;
    int     match_ins;
    int     match_p1;
    int     match_p2;
    int     match_lc;       /* exact-or-any */
    uint8_t match_data[260];
    int     match_data_len; /* exact-or-any (-1 = any) */
    /* Pre-recorded response. */
    nfc_apdu_t response;
} nfc_mock_entry_t;

typedef struct {
    nfc_mock_entry_t *entries;
    size_t count;
    size_t cap;
    size_t cursor;
    /* Diagnostic counters. */
    int unexpected_apdu;
    int replay_remaining;
} nfc_mock_t;

/* Construct a mock backend. The script is copied into the backend. */
nfc_backend_t *nfc_backend_mock_new(nfc_mock_entry_t *entries, size_t count);

/* Convenience: build a mock from a JSON file path. The JSON is parsed
 * by a minimal recursive-descent parser in src/mock_json.c (no
 * external deps). Returns NULL on parse error. */
nfc_backend_t *nfc_backend_mock_from_json(const char *path);

/* Read recorded script counters. */
int nfc_mock_unexpected_count(const nfc_backend_t *b);
int nfc_mock_remaining(const nfc_backend_t *b);

/* ----- High level eMRTD read-out API ----- */
typedef enum {
    NFC_STEP_SELECT_APP = 0,
    NFC_STEP_GET_CHALLENGE,
    NFC_STEP_MUTUAL_AUTH_1,
    NFC_STEP_MUTUAL_AUTH_2,
    NFC_STEP_SELECT_DG1,
    NFC_STEP_READ_DG1,
    NFC_STEP_SELECT_SOD,
    NFC_STEP_READ_SOD,
    NFC_STEP_DONE,
    NFC_STEP_ERROR,
} nfc_step_t;

typedef struct {
    int      ok;
    nfc_step_t step;
    char     detail[128];
    /* Decoded DG1 (MRZ from chip) is stored here (88 chars + NUL). */
    char     dg1_mrz[128];
    /* First 16 bytes of SOD (PKD signature placeholder). */
    uint8_t  sod_head[16];
    int      sod_head_len;
} nfc_result_t;

/* Drive the full BAC read sequence over the given backend. */
nfc_result_t nfc_read_emrtd(nfc_backend_t *backend);

const char *nfc_step_name(nfc_step_t s);

#ifdef __cplusplus
}
#endif
#endif /* NFC_H */
