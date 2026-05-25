"""
Generate temporal_boost.gif — demonstrates temporal reasoning boost.

Scene:
  1. 5 memories from a real MCP dev session (mixed ages, mixed relevance)
  2. Query types in: "what retrieval changes did we make recently?"
  3. Temporal window resolves: "recently -> last 14 days"
  4. In-window memories get +0.25 boost (cyan glow + badge)
  5. Cards animate to new ranked order — recent ones float to top
"""

from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent / "demo_gifs"
OUT_DIR.mkdir(exist_ok=True)

W, H = 900, 560

# ── Exact colors from index.html ──────────────────────────────────────────────
BG       = (10,  25,  47)    # #0A192F
CYAN     = (0,   212, 255)   # #00D4FF
WHITE    = (255, 255, 255)
CARD     = (22,  37,  57)    # bg-white/5 on #0A192F
BORD     = (35,  48,  68)    # border-white/10 on #0A192F
GRAY4    = (156, 163, 175)   # text-gray-400
GRAY6    = (107, 114, 128)   # text-gray-600
AMBER    = (245, 158,  11)   # #F59E0B
GREEN    = (16,  185, 129)   # #10B981
BOOST_BG = (8,   35,  58)    # subtle cyan tint
BOOST_BD = (0,   155, 195)   # muted cyan border

# ── Fonts ─────────────────────────────────────────────────────────────────────
_FONT = "/System/Library/Fonts/HelveticaNeue.ttc"

def F(size, bold=False):
    try:
        return ImageFont.truetype(_FONT, size, index=1 if bold else 0)
    except Exception:
        return ImageFont.load_default()

def tw(font, text):
    try:
        return int(font.getlength(text))
    except Exception:
        return len(text) * 7

def rr(d, x1, y1, x2, y2, r=8, fill=None, outline=None, lw=1):
    d.rounded_rectangle([x1, y1, x2, y2], radius=r, fill=fill, outline=outline, width=lw)

def lerp(a, b, t):
    return tuple(int(a[i] + (b[i] - a[i]) * t) for i in range(3))

def ease_out(t):
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    return t * t * (3 - 2 * t)

# ── Logo (matches index.html SVG) ─────────────────────────────────────────────
def draw_logo(d, x0, y0, size=26):
    def bar(vx, vy, vw, vh, fill):
        sc = size / 100
        d.rounded_rectangle(
            [x0 + vx*sc, y0 + vy*sc, x0 + (vx+vw)*sc, y0 + (vy+vh)*sc],
            radius=max(1, round(1.5)), fill=fill,
        )
    bar(10, 80, 80, 10, WHITE)
    bar(25, 60, 50, 10, WHITE)
    bar(40, 40, 15, 10, WHITE)
    bar(60, 40,  5, 10, CYAN)
    bar(47.5,20,  5, 10, (0, 170, 210))

# ── Memory data ───────────────────────────────────────────────────────────────
MEMORIES = [
    # Sorted by base score descending (initial display order)
    {"text": "DuckDB locally, Postgres in production",     "days": 22, "score": 0.71},
    {"text": "FastAPI pagination fixed /memories timeout", "days": 35, "score": 0.65},
    {"text": "Switched to BM25 + vector hybrid retrieval", "days":  9, "score": 0.62},
    {"text": "Tested Pinecone, kept local embeddings",     "days": 58, "score": 0.51},
    {"text": "Entity graph expansion added to recall",     "days":  2, "score": 0.47},
]
WINDOW = 14
BOOST  = 0.25
QUERY  = "what retrieval changes did we make recently?"

# Final order after boost
BOOSTED = sorted(
    range(len(MEMORIES)),
    key=lambda i: MEMORIES[i]["score"] + (BOOST if MEMORIES[i]["days"] <= WINDOW else 0),
    reverse=True,
)

def day_label(d):
    if d == 1:  return "1 day ago"
    if d < 14:  return f"{d} days ago"
    if d < 30:  return f"{d//7}w ago"
    return f"{d//30}mo ago"

# ── Layout (all in logical px at 900×560) ────────────────────────────────────
PAD    = 26
NAV_H  = 50
QY     = NAV_H + 14          # 64
QH     = 38
BADGE_Y = QY + QH + 9        # 111
BADGE_H = 24                 # ends at 135
HDR_Y  = BADGE_Y + BADGE_H + 9   # 144
ROW0   = HDR_Y + 20          # 164
CARD_H = 66
GAP    = 8

# ── Draw one frame ────────────────────────────────────────────────────────────
def draw_frame(query_chars=0, show_window=False, boost_alpha=0.0, rerank_t=0.0):
    img = Image.new("RGB", (W, H), BG)
    d   = ImageDraw.Draw(img)

    # Dot grid background
    for x in range(0, W + 30, 30):
        for y in range(0, H + 30, 30):
            d.ellipse([x-1, y-1, x+1, y+1], fill=(16, 33, 58))

    # ── Nav bar ──────────────────────────────────────────────────────────────
    d.rectangle([0, 0, W, NAV_H], fill=CARD)
    d.line([(0, NAV_H), (W, NAV_H)], fill=BORD, width=1)
    draw_logo(d, PAD + 2, 12, size=26)
    d.text((PAD + 34, 15), "YourMemory", font=F(15, bold=True), fill=WHITE)
    tag = "recall_memory"
    tf  = F(11)
    tw_tag = tw(tf, tag)
    rr(d, W - PAD - tw_tag - 20, 15, W - PAD, NAV_H - 14,
       r=6, fill=(18, 32, 52), outline=BORD)
    d.text((W - PAD - tw_tag - 12, 17), tag, font=tf, fill=CYAN)

    # ── Query input ───────────────────────────────────────────────────────────
    full = query_chars == len(QUERY)
    rr(d, PAD, QY, W - PAD, QY + QH,
       r=9, fill=(18, 32, 52),
       outline=CYAN if full else BORD, lw=2 if full else 1)
    q_shown = QUERY[:query_chars]
    qf = F(13)
    d.text((PAD + 14, QY + 12), q_shown, font=qf, fill=WHITE if full else GRAY4)
    if not full:
        cx = PAD + 14 + tw(qf, q_shown) + 1
        d.rectangle([cx, QY + 11, cx + 2, QY + 27], fill=CYAN)

    # ── Window badge ──────────────────────────────────────────────────────────
    if show_window and boost_alpha > 0:
        alpha = min(1.0, boost_alpha * 1.8)
        label = f"recently  ->  last {WINDOW} days"
        wf    = F(11, bold=True)
        ww    = tw(wf, label) + 36
        wx    = W // 2 - ww // 2
        wy    = BADGE_Y
        bg_c  = lerp(BG,    BOOST_BG, alpha)
        bd_c  = lerp(BG,    BOOST_BD, alpha)
        tc    = lerp(GRAY6, CYAN,     alpha)
        rr(d, wx, wy, wx + ww, wy + BADGE_H, r=12, fill=bg_c, outline=bd_c)
        # Mini clock icon
        ic, ir = wx + 13, wy + BADGE_H // 2
        d.ellipse([ic-5, ir-5, ic+5, ir+5], outline=tc, width=1)
        d.line([ic, ir-3, ic, ir],    fill=tc, width=1)
        d.line([ic, ir,   ic+3, ir],  fill=tc, width=1)
        d.text((wx + 26, wy + 6), label, font=wf, fill=tc)

    # ── Results header ────────────────────────────────────────────────────────
    hf = F(10)
    if rerank_t > 0.92:
        hdr = "re-ranked by temporal relevance"
        hc  = lerp(GRAY6, CYAN, min(1.0, (rerank_t - 0.92) * 12))
    elif boost_alpha > 0.5:
        hdr = "5 memories  |  2 in window"
        hc  = GRAY4
    else:
        hdr = "5 memories"
        hc  = GRAY6
    d.text((PAD, HDR_Y), hdr, font=hf, fill=hc)

    # ── Memory cards ─────────────────────────────────────────────────────────
    for orig_i in range(len(MEMORIES)):
        base_rank   = orig_i
        visual_rank = float(base_rank)
        if rerank_t > 0:
            final_rank  = BOOSTED.index(orig_i)
            visual_rank = base_rank + (final_rank - base_rank) * ease_in_out(rerank_t)

        y   = ROW0 + visual_rank * (CARD_H + GAP)
        m   = MEMORIES[orig_i]
        win = m["days"] <= WINDOW
        ba  = boost_alpha if win else 0.0

        # Card bg + border
        card_c = lerp(CARD, BOOST_BG, ba * 0.7) if win else CARD
        bord_c = lerp(BORD, BOOST_BD, ba)        if win else BORD
        rr(d, PAD, y, W - PAD, y + CARD_H, r=11, fill=card_c, outline=bord_c)

        # Left accent stripe on in-window cards
        if win and ba > 0:
            sc = lerp(BORD, CYAN, ba)
            d.rounded_rectangle([PAD, y + 4, PAD + 3, y + CARD_H - 4], radius=2, fill=sc)

        # Rank badge
        disp_rank = BOOSTED.index(orig_i) if rerank_t > 0.92 else base_rank
        d.text((PAD + 10, y + 9), f"#{disp_rank + 1}",
               font=F(10, bold=True), fill=lerp(GRAY6, CYAN, ba))

        # Memory text
        d.text((PAD + 42, y + 9), m["text"], font=F(13, bold=True), fill=WHITE)

        # Timestamp
        dlbl = day_label(m["days"])
        tsf  = F(11)
        d.text((PAD + 42, y + 31), dlbl, font=tsf, fill=GRAY4)

        # "in window" dot + label
        if win and ba > 0:
            dot_a = min(1.0, ba * 2)
            gc    = lerp(GRAY4, GREEN, dot_a)
            off   = tw(tsf, dlbl) + 10
            dx, dy = PAD + 42 + off, y + 36
            d.ellipse([dx, dy, dx + 6, dy + 6], fill=gc)
            d.text((dx + 10, y + 31), "in window", font=tsf, fill=gc)

        # Score bar
        bst   = BOOST * min(1.0, ba)
        score = m["score"] + bst
        BX = W - PAD - 168
        BW = 88
        by = y + CARD_H // 2 - 5
        rr(d, BX, by, BX + BW, by + 10, r=5, fill=BORD)
        filled = int(BW * min(1.0, score))
        bc = CYAN if score >= 0.75 else AMBER if score >= 0.55 else GRAY4
        if filled > 0:
            rr(d, BX, by, BX + filled, by + 10, r=5, fill=bc)
        d.text((BX + BW + 8, by - 1), f"{score:.2f}", font=F(12, bold=True), fill=bc)

        # +BOOST badge
        if win and ba > 0.15:
            fade = min(1.0, (ba - 0.15) / 0.85)
            bx2  = BX - 72
            by2  = y + CARD_H // 2 - 11
            rr(d, bx2, by2, bx2 + 64, by2 + 22, r=11,
               fill=lerp(card_c, BOOST_BG, fade),
               outline=lerp(bord_c, BOOST_BD, fade))
            d.text((bx2 + 8, by2 + 6), f"+{BOOST} boost",
                   font=F(10, bold=True), fill=lerp(card_c, CYAN, fade))

    # ── Footer ────────────────────────────────────────────────────────────────
    cap = "Temporal boost: +0.25 for memories within the resolved time window"
    cf  = F(10)
    d.text((W // 2 - tw(cf, cap) // 2, H - 14), cap, font=cf, fill=GRAY6)

    return img


# ── GIF save via gifsicle (avoids Pillow's 90-frame truncation bug) ───────────
import subprocess, tempfile, os

def _save_gif(path, frames, durations):
    """Write each frame as a single-frame GIF then stitch with gifsicle."""
    tmp = tempfile.mkdtemp(prefix="ymgif_")
    frame_files = []
    try:
        for i, (frame, ms) in enumerate(zip(frames, durations)):
            fp = os.path.join(tmp, f"f{i:04d}.gif")
            frame.save(fp, format="GIF", optimize=False)
            frame_files.append((fp, ms))

        # Build gifsicle command: set each frame's delay (unit = 1/100s)
        cmd = ["gifsicle", "--loop=0", "--colors=256"]
        for fp, ms in frame_files:
            delay = max(2, round(ms / 10))   # centiseconds
            cmd += [f"--delay={delay}", fp]
        cmd += ["--output", str(path)]

        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"gifsicle failed: {result.stderr}")
    finally:
        for fp, _ in frame_files:
            try: os.remove(fp)
            except OSError: pass
        try: os.rmdir(tmp)
        except OSError: pass


# ── GIF assembly ──────────────────────────────────────────────────────────────
def make_temporal_gif():
    print("-> temporal_boost.gif ...")
    frames, dur = [], []
    QLEN = len(QUERY)

    # Phase 1: blank hold
    for _ in range(14):
        frames.append(draw_frame(0)); dur.append(80)

    # Phase 2: type query
    for i in range(1, QLEN + 1):
        frames.append(draw_frame(i)); dur.append(42)

    # Phase 3: hold full query
    for _ in range(10):
        frames.append(draw_frame(QLEN)); dur.append(80)

    # Phase 4: fade in window badge + boost highlights
    for i in range(22):
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=ease_out(i / 21)))
        dur.append(50)

    # Phase 5: hold with boost
    for _ in range(16):
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=1.0))
        dur.append(75)

    # Phase 6: rerank animation
    for i in range(28):
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=1.0,
                                 rerank_t=ease_in_out(i / 27)))
        dur.append(52)

    # Phase 7: hold final state
    for _ in range(26):
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=1.0, rerank_t=1.0))
        dur.append(80)

    out = OUT_DIR / "temporal_boost.gif"
    print(f"  {len(frames)} frames, saving ...")
    _save_gif(out, frames, dur)
    print(f"  {out.stat().st_size // 1024} KB -> {out}")


if __name__ == "__main__":
    make_temporal_gif()
