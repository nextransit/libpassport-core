/* PPM / BMP I/O and basic image helpers. */
#include "face.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

face_image_t *face_image_new(int w, int h) {
    if (w <= 0 || h <= 0) return NULL;
    face_image_t *img = (face_image_t *)malloc(sizeof(*img));
    if (!img) return NULL;
    img->width = w; img->height = h;
    img->pixels = (uint8_t *)calloc((size_t)w * h * 3, 1);
    if (!img->pixels) { free(img); return NULL; }
    return img;
}

void face_image_free(face_image_t *img) {
    if (!img) return;
    free(img->pixels);
    free(img);
}

/* skip whitespace + comments in a PPM stream */
static int next_token(FILE *f, char *tok, size_t cap) {
    int c;
    /* skip whitespace and comments */
    do {
        c = fgetc(f);
        if (c == '#') { while (c != '\n' && c != EOF) c = fgetc(f); }
    } while (c != EOF && isspace(c));
    if (c == EOF) return 0;
    size_t i = 0;
    tok[i++] = (char)c;
    while ((c = fgetc(f)) != EOF && !isspace(c) && i + 1 < cap) tok[i++] = (char)c;
    if (c != EOF && isspace(c)) {
        /* The character that terminated the token was whitespace --
         * push it back so the next call to next_token does not skip
         * a byte of binary payload. */
        ungetc(c, f);
    }
    tok[i] = '\0';
    return 1;
}

face_image_t *face_image_load_ppm(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    char tok[64];
    if (!next_token(f, tok, sizeof(tok)) || strcmp(tok, "P6") != 0) {
        fclose(f); return NULL;
    }
    int w = 0, h = 0, maxv = 0;
    if (!next_token(f, tok, sizeof(tok))) { fclose(f); return NULL; }
    w = atoi(tok);
    if (!next_token(f, tok, sizeof(tok))) { fclose(f); return NULL; }
    h = atoi(tok);
    if (!next_token(f, tok, sizeof(tok))) { fclose(f); return NULL; }
    maxv = atoi(tok);
    if (w <= 0 || h <= 0 || maxv != 255) { fclose(f); return NULL; }
    /* exactly one whitespace separator before binary data */
    fgetc(f);

    face_image_t *img = face_image_new(w, h);
    if (!img) { fclose(f); return NULL; }
    size_t n = (size_t)w * h * 3;
    if (fread(img->pixels, 1, n, f) != n) {
        face_image_free(img); fclose(f); return NULL;
    }
    fclose(f);
    return img;
}

int face_image_save_ppm(const face_image_t *img, const char *path) {
    if (!img || !img->pixels) return -1;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    fprintf(f, "P6\n%d %d\n255\n", img->width, img->height);
    size_t n = (size_t)img->width * img->height * 3;
    if (fwrite(img->pixels, 1, n, f) != n) { fclose(f); return -1; }
    fclose(f);
    return 0;
}

/* ---------- BMP 24-bit reader/writer ----------
 * Supports uncompressed BI_RGB bitmaps with 24 bits per pixel.
 * Rows are stored 4-byte aligned. */
#pragma pack(push, 1)
typedef struct {
    uint16_t type;       /* "BM" */
    uint32_t size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t off_bits;
} bmp_file_header_t;

typedef struct {
    uint32_t size;
    int32_t  width;
    int32_t  height;
    uint16_t planes;
    uint16_t bit_count;
    uint32_t compression;
    uint32_t size_image;
    int32_t  x_pels_per_meter;
    int32_t  y_pels_per_meter;
    uint32_t clr_used;
    uint32_t clr_important;
} bmp_info_header_t;
#pragma pack(pop)

static uint32_t le32(uint32_t v) {
    uint32_t r = 0;
    r |= (v & 0x000000FFu) << 24;
    r |= (v & 0x0000FF00u) << 8;
    r |= (v & 0x00FF0000u) >> 8;
    r |= (v & 0xFF000000u) >> 24;
    return r;
}

face_image_t *face_image_load_bmp(const char *path) {
    FILE *f = fopen(path, "rb");
    if (!f) return NULL;
    bmp_file_header_t fh;
    bmp_info_header_t ih;
    if (fread(&fh, sizeof(fh), 1, f) != 1) { fclose(f); return NULL; }
    if (fread(&ih, sizeof(ih), 1, f) != 1) { fclose(f); return NULL; }
    if (fh.type != 0x4D42 || ih.bit_count != 24 || ih.compression != 0) {
        fclose(f); return NULL;
    }
    int w = ih.width;
    int h = ih.height < 0 ? -ih.height : ih.height; /* handle top-down */
    int bottom_up = (ih.height > 0);
    face_image_t *img = face_image_new(w, h);
    if (!img) { fclose(f); return NULL; }
    size_t row_bytes = (size_t)w * 3;
    size_t row_padded = (row_bytes + 3) & ~(size_t)3;
    uint8_t *row = (uint8_t *)malloc(row_padded);
    if (!row) { face_image_free(img); fclose(f); return NULL; }
    /* seek to pixel data */
    fseek(f, (long)fh.off_bits, SEEK_SET);
    for (int y = 0; y < h; ++y) {
        if (fread(row, 1, row_padded, f) != row_padded) {
            free(row); face_image_free(img); fclose(f); return NULL;
        }
        int dst_y = bottom_up ? (h - 1 - y) : y;
        uint8_t *dst = img->pixels + (size_t)dst_y * row_bytes;
        for (int x = 0; x < w; ++x) {
            /* BMP stores BGR */
            dst[x*3 + 0] = row[x*3 + 2];
            dst[x*3 + 1] = row[x*3 + 1];
            dst[x*3 + 2] = row[x*3 + 0];
        }
    }
    free(row);
    fclose(f);
    return img;
}

int face_image_save_bmp(const face_image_t *img, const char *path) {
    if (!img || !img->pixels) return -1;
    int w = img->width, h = img->height;
    size_t row_bytes = (size_t)w * 3;
    size_t row_padded = (row_bytes + 3) & ~(size_t)3;
    size_t pixel_bytes = row_padded * (size_t)h;
    bmp_file_header_t fh;
    bmp_info_header_t ih;
    fh.type = 0x4D42;
    fh.size = (uint32_t)(sizeof(fh) + sizeof(ih) + pixel_bytes);
    fh.reserved1 = fh.reserved2 = 0;
    fh.off_bits = (uint32_t)(sizeof(fh) + sizeof(ih));
    ih.size = (uint32_t)sizeof(ih);
    ih.width = w;
    ih.height = h; /* bottom-up */
    ih.planes = 1;
    ih.bit_count = 24;
    ih.compression = 0;
    ih.size_image = (uint32_t)pixel_bytes;
    ih.x_pels_per_meter = 0;
    ih.y_pels_per_meter = 0;
    ih.clr_used = 0;
    ih.clr_important = 0;
    FILE *f = fopen(path, "wb");
    if (!f) return -1;
    if (fwrite(&fh, sizeof(fh), 1, f) != 1) goto fail;
    if (fwrite(&ih, sizeof(ih), 1, f) != 1) goto fail;
    uint8_t *row = (uint8_t *)calloc(row_padded, 1);
    if (!row) goto fail;
    for (int y = h - 1; y >= 0; --y) {
        const uint8_t *src = img->pixels + (size_t)y * row_bytes;
        for (int x = 0; x < w; ++x) {
            row[x*3 + 0] = src[x*3 + 2]; /* B */
            row[x*3 + 1] = src[x*3 + 1]; /* G */
            row[x*3 + 2] = src[x*3 + 0]; /* R */
        }
        if (fwrite(row, 1, row_padded, f) != row_padded) { free(row); goto fail; }
    }
    free(row);
    fclose(f);
    return 0;
fail:
    fclose(f);
    return -1;
}
