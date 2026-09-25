/* Minimal JSON parser just powerful enough to read NFC mock scripts.
 * Supports: objects, arrays, strings, numbers (as int or hex with "0x"),
 * booleans, null. Sufficient for our own data/*.json scripts. */
#include "nfc.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <errno.h>

typedef struct {
    const char *s;
    size_t pos;
    size_t len;
} pstate_t;

static void skip_ws(pstate_t *p) {
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        if (c == ' ' || c == '\t' || c == '\n' || c == '\r') p->pos++;
        else break;
    }
}

static int peek(pstate_t *p) {
    skip_ws(p);
    return p->pos < p->len ? (unsigned char)p->s[p->pos] : -1;
}

static int eat(pstate_t *p, char c) {
    if (peek(p) == c) { p->pos++; return 1; }
    return 0;
}

static int parse_string(pstate_t *p, char *out, size_t cap) {
    if (peek(p) != '"') return -1;
    p->pos++;
    size_t i = 0;
    while (p->pos < p->len && p->s[p->pos] != '"') {
        if (p->s[p->pos] == '\\' && p->pos + 1 < p->len) {
            p->pos++;
            char esc = p->s[p->pos++];
            char c = esc;
            switch (esc) {
                case 'n': c = '\n'; break;
                case 't': c = '\t'; break;
                case 'r': c = '\r'; break;
                case '"': c = '"'; break;
                case '\\': c = '\\'; break;
                default: c = esc;
            }
            if (i + 1 < cap) out[i++] = c;
        } else {
            if (i + 1 < cap) out[i++] = p->s[p->pos++];
        }
    }
    if (p->pos >= p->len) return -1;
    p->pos++; /* closing quote */
    if (cap) out[i < cap ? i : cap - 1] = '\0';
    return 0;
}

static int parse_int(pstate_t *p, long *out) {
    skip_ws(p);
    int is_hex = 0;
    if (p->pos + 1 < p->len && p->s[p->pos] == '0' &&
        (p->s[p->pos+1] == 'x' || p->s[p->pos+1] == 'X')) {
        is_hex = 1; p->pos += 2;
    }
    long v = 0;
    int any = 0;
    while (p->pos < p->len) {
        char c = p->s[p->pos];
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (is_hex && c >= 'a' && c <= 'f') d = 10 + c - 'a';
        else if (is_hex && c >= 'A' && c <= 'F') d = 10 + c - 'A';
        else break;
        v = v * (is_hex ? 16 : 10) + d;
        any = 1;
        p->pos++;
    }
    if (!any) return -1;
    *out = v;
    return 0;
}

/* Find next key matching `name` at the current object level; advance
 * the cursor past it. Returns 1 if found, 0 otherwise. */
static int find_key(pstate_t *p, const char *name) {
    if (!eat(p, '{')) return -1;
    while (1) {
        char key[64];
        if (parse_string(p, key, sizeof(key)) != 0) return -1;
        if (!eat(p, ':')) return -1;
        /* Return having consumed the value implicitly? No, we return
         * with the cursor right after the colon so caller can parse
         * the value. We compare key up front though, so caller's
         * re-parse is consistent. */
        if (strcmp(key, name) == 0) return 1;
        /* Skip the value: a single value, object, or array. */
        int depth = 0;
        int in_str = 0;
        while (p->pos < p->len) {
            char c = p->s[p->pos];
            if (in_str) {
                if (c == '\\' && p->pos + 1 < p->len) { p->pos += 2; continue; }
                if (c == '"') in_str = 0;
                p->pos++; continue;
            }
            if (c == '"') { in_str = 1; p->pos++; continue; }
            if (c == '{' || c == '[') depth++;
            else if (c == '}' || c == ']') {
                if (depth == 0) break;
                depth--;
            }
            p->pos++;
        }
        if (!eat(p, ',')) {
            if (!eat(p, '}')) return -1;
            return 0;
        }
    }
}

/* Read a hex string of form "AABBCC..." (case-insensitive) and decode
 * it into `out` (max `cap` bytes). Returns decoded length or -1. */
static int parse_hex_string(pstate_t *p, uint8_t *out, size_t cap) {
    char buf[2048];
    if (parse_string(p, buf, sizeof(buf)) != 0) return -1;
    size_t n = strlen(buf);
    if (n % 2 != 0) return -1;
    size_t bytes = n / 2;
    if (bytes > cap) bytes = cap;
    for (size_t i = 0; i < bytes; ++i) {
        char hi = buf[i*2], lo = buf[i*2+1];
        int h = (hi >= '0' && hi <= '9') ? hi - '0' :
                (hi >= 'a' && hi <= 'f') ? 10 + hi - 'a' :
                (hi >= 'A' && hi <= 'F') ? 10 + hi - 'A' : -1;
        int l = (lo >= '0' && lo <= '9') ? lo - '0' :
                (lo >= 'a' && lo <= 'f') ? 10 + lo - 'a' :
                (lo >= 'A' && lo <= 'F') ? 10 + lo - 'A' : -1;
        if (h < 0 || l < 0) return -1;
        out[i] = (uint8_t)((h << 4) | l);
    }
    return (int)bytes;
}

/* Parse one entry { match: {...}, response: {...} }. */
/* Helper: parse an integer string (possibly hex). */
static int parse_int_from_str(const char *s, long *out) {
    int is_hex = 0;
    if (s[0] == '0' && (s[1] == 'x' || s[1] == 'X')) { is_hex = 1; s += 2; }
    long v = 0;
    int any = 0;
    while (*s) {
        char c = *s;
        int d;
        if (c >= '0' && c <= '9') d = c - '0';
        else if (is_hex && c >= 'a' && c <= 'f') d = 10 + c - 'a';
        else if (is_hex && c >= 'A' && c <= 'F') d = 10 + c - 'A';
        else break;
        v = v * (is_hex ? 16 : 10) + d;
        any = 1; s++;
    }
    if (!any) return -1;
    *out = v;
    return 0;
}

static int parse_quoted_int(pstate_t *p, long *out) {
    char tok[64];
    if (parse_string(p, tok, sizeof(tok)) != 0) return -1;
    return parse_int_from_str(tok, out);
}

static int parse_entry(pstate_t *p, nfc_mock_entry_t *e) {
    memset(e, -1, sizeof(*e)); /* -1 == "any" */
    e->response.sw = NFC_SW_OK;
    if (!eat(p, '{')) return -1;
    while (1) {
        char key[64];
        if (parse_string(p, key, sizeof(key)) != 0) return -1;
        if (!eat(p, ':')) return -1;
        if (strcmp(key, "match") == 0) {
            if (!eat(p, '{')) { fprintf(stderr, "no { after match\n"); return -1; }
            while (1) {
                char k[64]; long v;
                if (parse_string(p, k, sizeof(k)) != 0) return -1;
                if (!eat(p, ':')) return -1;
                if (strcmp(k, "cla") == 0) { if (parse_quoted_int(p, &v)!=0) return -1; e->match_cla = (int)v; }
                else if (strcmp(k, "ins") == 0) { if (parse_quoted_int(p, &v)!=0) return -1; e->match_ins = (int)v; }
                else if (strcmp(k, "p1") == 0) { if (parse_quoted_int(p, &v)!=0) return -1; e->match_p1 = (int)v; }
                else if (strcmp(k, "p2") == 0) { if (parse_quoted_int(p, &v)!=0) return -1; e->match_p2 = (int)v; }
                else if (strcmp(k, "lc") == 0) { if (parse_quoted_int(p, &v)!=0) return -1; e->match_lc = (int)v; }
                else if (strcmp(k, "le") == 0) { /* ignored, just consume */ if (parse_quoted_int(p, &v)!=0) return -1; }
                else if (strcmp(k, "data") == 0) {
                    char plain[1024];
                    if (parse_string(p, plain, sizeof(plain)) != 0) return -1;
                    size_t plen = strlen(plain);
                    int is_hex = (plen > 0 && plen % 2 == 0);
                    if (is_hex) {
                        for (size_t i = 0; i < plen; ++i) {
                            char c = plain[i];
                            int ok = (c >= '0' && c <= '9') ||
                                     (c >= 'a' && c <= 'f') ||
                                     (c >= 'A' && c <= 'F');
                            if (!ok) { is_hex = 0; break; }
                        }
                    }
                    if (is_hex) {
                        size_t bytes = plen / 2;
                        if (bytes > sizeof(e->match_data)) bytes = sizeof(e->match_data);
                        for (size_t i = 0; i < bytes; ++i) {
                            char hi = plain[i*2], lo = plain[i*2+1];
                            int h = (hi >= '0' && hi <= '9') ? hi - '0' :
                                    (hi >= 'a' && hi <= 'f') ? 10 + hi - 'a' :
                                    (hi >= 'A' && hi <= 'F') ? 10 + hi - 'A' : 0;
                            int l = (lo >= '0' && lo <= '9') ? lo - '0' :
                                    (lo >= 'a' && lo <= 'f') ? 10 + lo - 'a' :
                                    (lo >= 'A' && lo <= 'F') ? 10 + lo - 'A' : 0;
                            e->match_data[i] = (uint8_t)((h << 4) | l);
                        }
                        e->match_data_len = (int)bytes;
                    } else {
                        size_t n = plen;
                        if (n > sizeof(e->match_data)) n = sizeof(e->match_data);
                        memcpy(e->match_data, plain, n);
                        e->match_data_len = (int)n;
                    }
                }
                else return -1;
                if (!eat(p, ',')) break;
            }
            if (!eat(p, '}')) return -1;
        } else if (strcmp(key, "response") == 0) {
            if (!eat(p, '{')) { fprintf(stderr, "no { after response\n"); return -1; }
            while (1) {
                char k[64]; long v;
                if (parse_string(p, k, sizeof(k)) != 0) return -1;
                if (!eat(p, ':')) return -1;
                if (strcmp(k, "sw") == 0) {
                    if (parse_quoted_int(p, &v) != 0) return -1;
                    e->response.sw = (uint16_t)v;
                } else if (strcmp(k, "data") == 0) {
                    /* Parse the string verbatim first. If the content
                     * is valid even-length hex, decode it; otherwise
                     * keep it as bytes. */
                    char plain[1024];
                    if (parse_string(p, plain, sizeof(plain)) != 0)
                        return -1;
                    size_t plen = strlen(plain);
                    int is_hex = (plen > 0 && plen % 2 == 0);
                    if (is_hex) {
                        for (size_t i = 0; i < plen; ++i) {
                            char c = plain[i];
                            int ok = (c >= '0' && c <= '9') ||
                                     (c >= 'a' && c <= 'f') ||
                                     (c >= 'A' && c <= 'F');
                            if (!ok) { is_hex = 0; break; }
                        }
                    }
                    if (is_hex) {
                        size_t bytes = plen / 2;
                        if (bytes > sizeof(e->response.data)) bytes = sizeof(e->response.data);
                        for (size_t i = 0; i < bytes; ++i) {
                            char hi = plain[i*2], lo = plain[i*2+1];
                            int h = (hi >= '0' && hi <= '9') ? hi - '0' :
                                    (hi >= 'a' && hi <= 'f') ? 10 + hi - 'a' :
                                    (hi >= 'A' && hi <= 'F') ? 10 + hi - 'A' : 0;
                            int l = (lo >= '0' && lo <= '9') ? lo - '0' :
                                    (lo >= 'a' && lo <= 'f') ? 10 + lo - 'a' :
                                    (lo >= 'A' && lo <= 'F') ? 10 + lo - 'A' : 0;
                            e->response.data[i] = (uint8_t)((h << 4) | l);
                        }
                        e->response.lc = (uint16_t)bytes;
                    } else {
                        size_t n = plen;
                        if (n > sizeof(e->response.data)) n = sizeof(e->response.data);
                        memcpy(e->response.data, plain, n);
                        e->response.lc = (uint16_t)n;
                    }
                } else if (strcmp(k, "le") == 0) {
                    if (parse_quoted_int(p, &v) != 0) return -1;
                    e->response.le = (uint16_t)v;
                } else return -1;
                if (!eat(p, ',')) break;
            }
            if (!eat(p, '}')) return -1;
        } else return -1;
        if (!eat(p, ',')) {
            /* No trailing comma -> this is the last entry. */
            break;
        }
    }
    if (!eat(p, '}')) { fprintf(stderr, "parse_entry: no closing } at pos %zu\n", p->pos); return -1; }
    return 0;
}

nfc_backend_t *nfc_backend_mock_from_json(const char *path) {
    FILE *f = fopen(path, "rb");
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    fseek(f, 0, SEEK_SET);
    if (n <= 0 || n > 4 * 1024 * 1024) { fprintf(stderr, "DBG: bad size %ld\n", n); fclose(f); return NULL; }
    char *buf = (char *)malloc((size_t)n + 1);
    if (!buf) { fclose(f); return NULL; }
    if (fread(buf, 1, (size_t)n, f) != (size_t)n) { fprintf(stderr, "DBG: fread short\n"); free(buf); fclose(f); return NULL; }
    buf[n] = '\0';
    fclose(f);

    pstate_t st = { buf, 0, (size_t)n };
    if (!eat(&st, '{')) { fprintf(stderr, "DBG: no top {\n"); free(buf); return NULL; }
    char k[64];
    if (parse_string(&st, k, sizeof(k)) != 0 || strcmp(k, "entries") != 0) {
        fprintf(stderr, "DBG: missing entries key, got %s\n", k); free(buf); return NULL;
    }
    if (!eat(&st, ':')) { fprintf(stderr, "DBG: no : after entries\n"); free(buf); return NULL; }
    if (!eat(&st, '[')) { fprintf(stderr, "DBG: no [ at pos %zu\n", st.pos); free(buf); return NULL; }

    nfc_mock_entry_t *arr = NULL;
    size_t cap = 0, count = 0;
    while (peek(&st) == '{') {
        if (count == cap) {
            size_t nc = cap ? cap * 2 : 8;
            arr = (nfc_mock_entry_t *)realloc(arr, nc * sizeof(*arr));
            cap = nc;
        }
        if (parse_entry(&st, &arr[count]) != 0) {
            fprintf(stderr, "DBG: parse_entry failed at count=%zu pos=%zu\n", count, st.pos);
            free(arr); free(buf); return NULL;
        }
        count++;
        /* If a comma follows, consume it before the next iteration. */
        if (peek(&st) == ',') eat(&st, ',');
    }
    /* Skip any trailing cruft up to closing brace. */
    while (st.pos < st.len && st.s[st.pos] != '}') st.pos++;
    eat(&st, '}');
    free(buf);
    if (count == 0) { free(arr); return NULL; }
    return nfc_backend_mock_new(arr, count);
}
