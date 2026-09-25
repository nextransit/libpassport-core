/* CNN-based MRZ OCR pipeline.
 *
 * Identical structure to mrz_ocr.c but uses the tiny CNN
 * (cnn_default_init / cnn_forward) for character recognition instead
 * of the template-bank correlation.
 */
#include "mrz_ocr.h"
#include "cnn.h"
#include "template.h"

typedef struct { int x, y, w, h; } rect_t;

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int resample_char_cnn(const uint8_t *bin, int W, int H,
                             int rx, int ry, int rw, int rh,
                             uint8_t out[MRZ_OCR_GLYPH_H][MRZ_OCR_GLYPH_W]) {
    if (rw <= 0 || rh <= 0) return -1;
    for (int oy = 0; oy < MRZ_OCR_GLYPH_H; ++oy) {
        int sy0 = oy * rh / MRZ_OCR_GLYPH_H;
        int sy1 = (oy + 1) * rh / MRZ_OCR_GLYPH_H;
        if (sy1 <= sy0) sy1 = sy0 + 1;
        for (int ox = 0; ox < MRZ_OCR_GLYPH_W; ++ox) {
            int sx0 = ox * rw / MRZ_OCR_GLYPH_W;
            int sx1 = (ox + 1) * rw / MRZ_OCR_GLYPH_W;
            if (sx1 <= sx0) sx1 = sx0 + 1;
            int ink = 0, total = 0;
            for (int sy = sy0; sy < sy1; ++sy) {
                int yy = ry + sy;
                if (yy < 0 || yy >= H) continue;
                for (int sx = sx0; sx < sx1; ++sx) {
                    int xx = rx + sx;
                    if (xx < 0 || xx >= W) continue;
                    ink += bin[yy * W + xx];
                    total++;
                }
            }
            /* Threshold at 1/2 (>=50%% of the box must be ink),
             * matching the traditional mrz_ocr.c binarisation step so
             * the same character produces the same 8x12 bitmap on
             * both pipelines. */
            out[oy][ox] = (total > 0 && ink * 2 >= total) ? 1 : 0;
        }
    }
    return 0;
}

mrz_ocr_status_t mrz_ocr_recognise_cnn(const face_image_t *img,
                                       mrz_ocr_result_t *out) {
    if (!img || !out) return MRZ_OCR_ERR_LOAD;
    memset(out, 0, sizeof(*out));
    int W = img->width, H = img->height;
    if (W <= 0 || H <= 0) return MRZ_OCR_ERR_LOAD;

    size_t n = (size_t)W * H;
    uint8_t *gray = (uint8_t *)malloc(n);
    if (!gray) return MRZ_OCR_ERR_LOAD;
    face_to_grayscale(img, gray);
    int t = 127;
    {
        size_t hist[256] = {0};
        for (size_t i = 0; i < n; ++i) hist[gray[i]]++;
        double sum = 0;
        for (int i = 0; i < 256; ++i) sum += (double)i * hist[i];
        double sumB = 0;
        int wB = 0;
        double maxVar = -1;
        for (int i = 0; i < 256; ++i) {
            wB += (int)hist[i];
            if (wB == 0) continue;
            int wF = (int)n - wB;
            if (wF == 0) break;
            sumB += (double)i * hist[i];
            double mB = sumB / wB, mF = (sum - sumB) / wF;
            double v = (double)wB * wF * (mB - mF) * (mB - mF);
            if (v > maxVar) { maxVar = v; t = i; }
        }
        if (hist[0] > 0 && hist[255] == 0) t = 127;
        if (t == 0 && hist[255] > 0) t = 127;
    }
    uint8_t *bin = (uint8_t *)malloc(n);
    if (!bin) { free(gray); return MRZ_OCR_ERR_LOAD; }
    for (size_t i = 0; i < n; ++i) bin[i] = (gray[i] < t) ? 1 : 0;
    free(gray);

    /* Locate band. */
    mrz_ocr_rect_t band;
    if (mrz_ocr_locate_band(bin, W, H, &band) != 0) {
        free(bin);
        return MRZ_OCR_ERR_NO_BAND;
    }
    int bx = band.x, by = band.y, bw = band.w, bh = band.h;
    out->band_x = bx; out->band_y = by; out->band_w = bw; out->band_h = bh;
    if (bw < MRZ_OCR_GLYPH_W * 5 || bh < MRZ_OCR_GLYPH_H) {
        free(bin);
        return MRZ_OCR_ERR_BAD_GEOMETRY;
    }
    uint8_t *band_pixels = (uint8_t *)malloc((size_t)bw * bh);
    if (!band_pixels) { free(bin); return MRZ_OCR_ERR_LOAD; }
    for (int y = 0; y < bh; ++y)
        memcpy(band_pixels + y * bw, bin + (by + y) * W + bx, bw);
    /* Split lines. */
    mrz_ocr_rect_t lines[2];
    if (mrz_ocr_split_lines(band_pixels, bw, bh, lines) != 0) {
        free(band_pixels); free(bin);
        return MRZ_OCR_ERR_BAD_LINES;
    }
    /* Initialise the CNN. */
    cnn_t net;
    cnn_default_init(&net);

    /* For each line, segment and recognise. */
    for (int li = 0; li < 2; ++li) {
        int base = (li == 0) ? lines[0].y : lines[1].y;
        int line_h = (li == 0) ? lines[0].h : lines[1].h;
        int line_w = (li == 0) ? lines[0].w : lines[1].w;
        if (line_h <= 0) continue;
        uint8_t *line_pixels = (uint8_t *)malloc((size_t)bw * line_h);
        if (!line_pixels) { free(band_pixels); free(bin); return MRZ_OCR_ERR_LOAD; }
        for (int y = 0; y < line_h; ++y)
            memcpy(line_pixels + y * bw, band_pixels + (base + y) * bw, bw);
        
        /* Segment characters. */
        mrz_ocr_rect_t chars[64];
        int nchars = mrz_ocr_segment_line(line_pixels, bw, line_h, chars, 64);
        if (nchars < 30) { free(line_pixels); free(band_pixels); free(bin); return MRZ_OCR_ERR_BAD_LINES; }
        char *dest = (li == 0) ? out->line1 : out->line2;
        int *dest_len = (li == 0) ? &out->line1_len : &out->line2_len;
        int *dest_conf = (li == 0) ? &out->line1_avg_conf : &out->line2_avg_conf;
        int conf_sum = 0;
        for (int c = 0; c < nchars && c < 44; ++c) {
            uint8_t ch[MRZ_OCR_GLYPH_H][MRZ_OCR_GLYPH_W];
            resample_char_cnn(line_pixels, bw, line_h,
                              chars[c].x, chars[c].y, chars[c].w, chars[c].h, ch);
            int pred = -1;
            float probs[37];
            cnn_forward(&net, ch, &pred, probs);
            if (pred < 0) { dest[c] = '?'; conf_sum += 0; }
            else {
                dest[c] = MRZ_OCR_GLYPHS[pred].ch;
                conf_sum += (int)(probs[pred] * 100);
            }
        }
        *dest_len = nchars < 44 ? nchars : 44;
        dest[*dest_len] = '\0';
        *dest_conf = nchars > 0 ? (conf_sum / nchars) : 0;
        free(line_pixels);
    }
    free(band_pixels);
    free(bin);
    return MRZ_OCR_OK;
}
