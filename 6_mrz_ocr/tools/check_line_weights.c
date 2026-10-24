/* Sanity check for the shipped line models.
 *
 * Feeds stored 16x12 real-photo glyph slices through the SAME C code the
 * pipeline uses and reports the top-1 accuracy per line.  A weight file
 * whose layout was mangled on export shows up here as a single-digit
 * accuracy, which is far easier to notice than a few wrong MRZ rows.
 *
 * Usage: check_line_weights <slices.ppm-dir> ...
 * The slice set is a directory of "<idx>_<label>.ppm" files. */
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>
#include <dirent.h>
#include "cnn.h"

static const char *GLYPHS = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ<";

int main(int argc, char **argv) {
    if (argc < 3) {
        fprintf(stderr, "usage: %s <slice-dir> <line0|line1> [line1]\n", argv[0]);
        return 2;
    }
    int line = atoi(argv[2]);
    cnn_t net;
    cnn_line_init(&net, line);
    int ok = 0, tot = 0;
    char path[1024];
    /* argv[3] is a shell glob of slice file names; too many for argv on
     * some systems, so read the directory instead. */
    char **names = NULL;
    int n_names = 0;
    if (argc == 3) {
        DIR *d = opendir(argv[1]);
        if (!d) { fprintf(stderr, "cannot open %s\n", argv[1]); return 2; }
        struct dirent *e;
        while ((e = readdir(d))) {
            if (e->d_name[0] == '.') continue;
            names = realloc(names, (n_names + 1) * sizeof(char *));
            names[n_names++] = strdup(e->d_name);
        }
        closedir(d);
    }
    for (int i = 0; i < (argc > 3 ? argc - 3 : n_names); i++) {
        const char *nm = (argc > 3) ? argv[i + 3] : names[i];
        snprintf(path, sizeof(path), "%s/%s", argv[1], nm);
        FILE *f = fopen(path, "rb");
        if (!f) continue;
        int w = 0, h = 0, mx = 0;
        char magic[3] = {0};
        if (fscanf(f, "%2s", magic) != 1) { fclose(f); continue; }
        if (fscanf(f, "%d %d %d", &w, &h, &mx) != 3) { fclose(f); continue; }
        float glyph[CNN_IN_H][CNN_IN_W];
        for (int y = 0; y < CNN_IN_H; y++)
            for (int x = 0; x < CNN_IN_W; x++) glyph[y][x] = 0.0f;
        for (int y = 0; y < h; y++) {
            for (int x = 0; x < w; x++) {
                int c;
                if (fscanf(f, "%d", &c) != 1) c = 0;
                int oy = y * CNN_IN_H / h, ox = x * CNN_IN_W / w;
                if (oy < CNN_IN_H && ox < CNN_IN_W)
                    glyph[oy][ox] = 1.0f - (float)c / 255.0f;
            }
        }
        fclose(f);
        /* label = last '_' separated token of the file name */
        const char *base = strrchr(nm, '/');
        base = base ? base + 1 : nm;
        char lab = 0;
        char *u = strrchr(base, '_');
        if (u && u[1]) lab = u[1];
        if (!lab) continue;
        float probs[CNN_OUT];
        int pred = cnn_forward(&net, glyph, probs);
        const char *wp = strchr(GLYPHS, lab);
        if (!wp) continue;
        int want = (int)(wp - GLYPHS);
        if (want >= 0 && want < CNN_OUT) {
            tot++;
            if (pred == want) ok++;
        }
    }
    fprintf(stderr, "line%d: %d/%d (%.2f%%)\n", line, ok, tot,
            tot ? 100.0 * ok / tot : 0.0);
    return tot ? (ok == tot ? 0 : 1) : 2;
}
