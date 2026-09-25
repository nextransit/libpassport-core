"""Generate the Passport Test Bench app icon.

Design (512x512, ICAO e-passport "test bench" motif):
  - deep navy background (radial)
  - an open e-passport: deep-red cover + light page
  - gold MRZ band (2 horizontal OCR lines) across the page
  - an NFC field arc (wireless chip comms) on the cover
  - a small green check badge = "verification / test pass"
"""
from PIL import Image, ImageDraw, ImageFilter

S = 512
img = Image.new("RGBA", (S, S), (0, 0, 0, 0))
d = ImageDraw.Draw(img)

# ---- background: radial-ish (two concentric ellipses) ----
d.rounded_rectangle([0, 0, S - 1, S - 1], radius=110, fill=(11, 27, 56, 255))
d.rounded_rectangle([28, 28, S - 29, S - 29], radius=88, fill=(16, 38, 78, 255))
# subtle inner vignette
d.rounded_rectangle([56, 56, S - 57, S - 57], radius=70, fill=(22, 52, 102, 255))

# ---- e-passport (open book): back cover behind, front cover ----
cx = S // 2
# page (light)
page = [(cx - 190, 150), (cx + 190, 352)]
d.rounded_rectangle(page, radius=14, fill=(242, 242, 240, 255),
                    outline=(208, 208, 205, 255), width=2)
# spine shadow
d.rectangle([cx - 3, 150, cx + 3, 352], fill=(180, 180, 176, 255))
# front cover (deep red, classic passport)
cover = [(cx - 190, 118), (cx + 190, 320)]
d.rounded_rectangle(cover, radius=16, fill=(139, 27, 40, 255),
                    outline=(101, 15, 26, 255), width=3)
# cover emblem circle + dot
d.ellipse([cx - 38, 142, cx + 38, 218], outline=(228, 178, 66, 255), width=4)
d.ellipse([cx - 16, 166, cx + 16, 198], fill=(228, 178, 66, 255))
# cover text line (subtle)
d.line([cx - 90, 252, cx + 90, 252], fill=(210, 210, 205, 255), width=5)

# ---- gold MRZ band on the page ----
mrz_y0, mrz_y1 = 286, 330
d.rounded_rectangle([cx - 170, mrz_y0, cx + 170, mrz_y1],
                    radius=8, fill=(255, 255, 255, 255),
                    outline=(150, 150, 148, 255), width=1)
for i, yy in enumerate(range(mrz_y0 + 14, mrz_y1 - 2, 16)):
    d.line([cx - 158, yy, cx + 158, yy], fill=(35, 35, 35, 255), width=3)

# ---- NFC field arcs above the book ----
for r, w in ((56, 5), (44, 4), (32, 3)):
    d.arc([cx - r, 62 - r + 10, cx + r, 62 + r + 10],
          start=200, end=340, fill=(120, 196, 255, 235), width=w)
d.line([cx - 30, 150, cx - 8, 150], fill=(120, 196, 255, 235), width=5)
d.line([cx + 8, 150, cx + 30, 150], fill=(120, 196, 255, 235), width=5)

# ---- green check badge (test passed) ----
bd = [(cx + 120, 92), (cx + 178, 150)]
d.ellipse(bd, fill=(46, 160, 90, 255), outline=(30, 120, 66, 255), width=3)
d.line([cx + 132, 122, cx + 144, 132, cx + 166, 108],
       fill=(255, 255, 255, 255), width=8, joint="curve")

# soft shadow under book
shadow = Image.new("RGBA", (S, S), (0, 0, 0, 0))
sd = ImageDraw.Draw(shadow)
sd.rounded_rectangle([cx - 205, 368, cx + 205, 400], radius=16,
                     fill=(0, 0, 0, 70))
shadow = shadow.filter(ImageFilter.GaussianBlur(6))
img = Image.alpha_composite(img, shadow)

img = img.resize((S, S), Image.LANCZOS)
out = "tests/assets/app_icon.png"
import os
os.makedirs("tests/assets", exist_ok=True)
img.save(out, "PNG")
print("wrote", out, img.size)

# variants
for s in (256, 128, 64, 32):
    v = img.resize((s, s), Image.LANCZOS)
    v.save(f"tests/assets/app_icon_{s}.png", "PNG")
print("variants written")
