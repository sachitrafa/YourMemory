"""
Generate temporal_boost.gif — demonstrates the temporal reasoning boost in action.

Scene:
  1. 5 memories shown with creation timestamps (ranging from 2 days to 60 days ago)
  2. Query types in: "what did we discuss recently about caching?"
  3. Temporal window resolves: "recently → last 14 days"
  4. Memories within the window get a +0.25 BOOST highlight
  5. Re-ranked results surface the recent memory to top
"""

import math, os
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont

OUT_DIR = Path(__file__).parent / "demo_gifs"
OUT_DIR.mkdir(exist_ok=True)

W, H = 900, 560

NAVY     = (10,  25,  47)
CYAN     = (0,   212, 255)
CYAN_DIM = (0,   140, 180)
WHITE    = (255, 255, 255)
G_BG     = (13,  17,  23)
G_CARD   = (22,  27,  34)
G_CARD2  = (28,  34,  44)
G_BORD   = (48,  54,  61)
G_TEXT   = (220, 230, 242)
G_MUTE   = (110, 120, 135)
AMBER    = (245, 158, 11)
GREEN    = (63,  185,  80)
BOOST_BG = (0,   50,   60)
BOOST_BD = (0,   180, 220)

def F(size, bold=False):
    path = "/tmp/DMSans-Bold.ttf" if bold else "/tmp/DMSans-Regular.ttf"
    try:
        return ImageFont.truetype(path, size)
    except Exception:
        return ImageFont.load_default()

def tw(font, text):
    try:
        return int(font.getlength(text))
    except Exception:
        return len(text) * 7

def rr(d, xy, r=8, fill=None, outline=None, lw=1):
    d.rounded_rectangle(list(xy), radius=r, fill=fill, outline=outline, width=lw)

def ease_out(t):
    return 1 - (1 - t) ** 3

def ease_in_out(t):
    return t * t * (3 - 2 * t)

def draw_logo(d, x0, y0, size=28, bg=G_BG):
    s  = size / 100
    rv = max(1, round(2 * s))
    def bar(vx, vy, vw, vh, fill):
        d.rounded_rectangle(
            [x0+vx*s, y0+vy*s, x0+(vx+vw)*s, y0+(vy+vh)*s],
            radius=rv, fill=fill
        )
    bar(10, 80, 80, 10, WHITE)
    bar(25, 60, 50, 10, WHITE)
    bar(40, 40, 15, 10, WHITE)
    bar(60, 40,  5, 10, CYAN)
    bar(47.5,20, 5, 10, (0, 170, 210))

# ── Memory data ───────────────────────────────────────────────────────────────
MEMORIES = [
    {"text": "Redis caching failed: pickle, not JSON",       "days": 2,  "score": 0.61},
    {"text": "Switched to Cassandra for session caching",    "days": 8,  "score": 0.74},
    {"text": "Sachit uses DuckDB locally, Postgres in prod", "days": 21, "score": 0.55},
    {"text": "FastAPI pagination fixed /users timeout",      "days": 35, "score": 0.48},
    {"text": "Decided against Pinecone, DuckDB sufficient",  "days": 58, "score": 0.39},
]
WINDOW_DAYS = 14
BOOST       = 0.25
QUERY       = "what did we discuss recently about caching?"

PAD   = 28
NAV_H = 52
CARD_H= 66
GAP   = 8
ROW0  = NAV_H + 60   # first card top

def day_label(d):
    if d == 1:  return "1 day ago"
    if d < 14:  return f"{d} days ago"
    if d < 30:  return f"{d//7}w ago"
    return f"{d//30}mo ago"

def in_window(days):
    return days <= WINDOW_DAYS

def draw_frame(
    query_chars=0,
    show_window=False,
    boost_alpha=0.0,
    rerank_t=0.0,
    highlight_idx=-1,
):
    img = Image.new("RGB", (W, H), G_BG)
    d   = ImageDraw.Draw(img)

    # subtle dot grid
    for x in range(0, W+30, 30):
        for y in range(0, H+30, 30):
            d.ellipse([x-1,y-1,x+1,y+1], fill=(20,26,34))

    # ── Nav bar ──
    d.rectangle([0,0,W,NAV_H], fill=G_CARD)
    d.line([(0,NAV_H),(W,NAV_H)], fill=G_BORD, width=1)
    draw_logo(d, PAD, 12, size=28, bg=G_CARD)
    d.text((PAD+36, 16), "YourMemory", font=F(16, bold=True), fill=WHITE)
    tag = "recall_memory"
    tw_tag = tw(F(11), tag)
    rr(d, [W-PAD-tw_tag-16, 16, W-PAD, NAV_H-14], r=6, fill=G_CARD2, outline=G_BORD)
    d.text((W-PAD-tw_tag-8, 18), tag, font=F(11), fill=CYAN)

    # ── Query box ──
    QY = NAV_H + 16
    QH = 38
    rr(d, [PAD, QY, W-PAD, QY+QH], r=8, fill=G_CARD2, outline=CYAN if query_chars==len(QUERY) else G_BORD, lw=2 if query_chars==len(QUERY) else 1)
    q_shown = QUERY[:query_chars]
    d.text((PAD+14, QY+11), q_shown, font=F(13), fill=G_TEXT)
    # blinking cursor
    if query_chars < len(QUERY):
        cx = PAD + 14 + tw(F(13), q_shown)
        d.rectangle([cx+2, QY+10, cx+3, QY+28], fill=CYAN)

    # ── Temporal window badge ──
    if show_window and boost_alpha > 0:
        alpha = min(1.0, boost_alpha * 2)
        wlabel = f"⏱  recently  →  last {WINDOW_DAYS} days"
        ww = tw(F(12), wlabel) + 28
        wx = W//2 - ww//2
        wy = QY + QH + 8
        bg_col = tuple(int(c * alpha + G_BG[i] * (1-alpha)) for i,c in enumerate(BOOST_BG))
        bd_col = tuple(int(c * alpha + G_BG[i] * (1-alpha)) for i,c in enumerate(BOOST_BD))
        rr(d, [wx, wy, wx+ww, wy+26], r=13, fill=bg_col, outline=bd_col)
        txt_col = tuple(int(c * alpha + G_BG[i] * (1-alpha)) for i,c in enumerate(CYAN))
        d.text((wx+14, wy+7), wlabel, font=F(12), fill=txt_col)

    # ── Memory cards ──
    # Compute reranked order
    base_order = list(range(len(MEMORIES)))
    boosted_order = sorted(
        range(len(MEMORIES)),
        key=lambda i: MEMORIES[i]["score"] + (BOOST if in_window(MEMORIES[i]["days"]) else 0),
        reverse=True,
    )

    for rank, orig_i in enumerate(base_order):
        t_rank = rank
        if rerank_t > 0:
            final_rank = boosted_order.index(orig_i)
            t_rank = rank + (final_rank - rank) * ease_out(rerank_t)

        y = ROW0 + t_rank * (CARD_H + GAP)

        m   = MEMORIES[orig_i]
        win = in_window(m["days"])
        bst = BOOST if win else 0

        # card background — glow for in-window
        if win and boost_alpha > 0:
            glow_alpha = boost_alpha * 0.5
            glow_col = tuple(int(c*glow_alpha + G_CARD[i]*(1-glow_alpha)) for i,c in enumerate((0,40,55)))
            rr(d, [PAD-2, y-2, W-PAD+2, y+CARD_H+2], r=12, fill=glow_col)

        border = BOOST_BD if (win and boost_alpha > 0.3) else G_BORD
        rr(d, [PAD, y, W-PAD, y+CARD_H], r=10, fill=G_CARD, outline=border)

        # Memory text
        d.text((PAD+16, y+12), m["text"], font=F(13), fill=G_TEXT)

        # Timestamp
        dlbl = day_label(m["days"])
        d.text((PAD+16, y+34), dlbl, font=F(11), fill=G_MUTE)

        # Window indicator dot
        if win and boost_alpha > 0:
            dot_alpha = min(1.0, boost_alpha * 2)
            dot_col = tuple(int(c*dot_alpha + G_CARD[i]*(1-dot_alpha)) for i,c in enumerate(GREEN))
            d.ellipse([PAD+16+tw(F(11),dlbl)+8, y+38, PAD+16+tw(F(11),dlbl)+15, y+45], fill=dot_col)
            in_txt = "in window"
            in_col = tuple(int(c*dot_alpha + G_CARD[i]*(1-dot_alpha)) for i,c in enumerate(GREEN))
            d.text((PAD+16+tw(F(11),dlbl)+20, y+34), in_txt, font=F(11), fill=in_col)

        # Score bar
        score_with_boost = m["score"] + bst * min(1.0, boost_alpha)
        BAR_X  = W - PAD - 180
        BAR_W2 = 100
        by     = y + CARD_H//2 - 5
        rr(d, [BAR_X, by, BAR_X+BAR_W2, by+10], r=5, fill=G_BORD)
        filled = int(BAR_W2 * score_with_boost)
        bar_col2 = CYAN if score_with_boost >= 0.75 else AMBER if score_with_boost >= 0.55 else G_MUTE
        rr(d, [BAR_X, by, BAR_X+filled, by+10], r=5, fill=bar_col2)

        score_lbl = f"{score_with_boost:.2f}"
        d.text((BAR_X + BAR_W2 + 8, by-1), score_lbl, font=F(12, bold=True), fill=bar_col2)

        # BOOST badge
        if win and boost_alpha > 0.2:
            ba = min(1.0, (boost_alpha - 0.2) / 0.8)
            bx2 = BAR_X - 76
            by2 = y + CARD_H//2 - 12
            bg2 = tuple(int(c*ba + G_CARD[i]*(1-ba)) for i,c in enumerate(BOOST_BG))
            bd2 = tuple(int(c*ba + G_CARD[i]*(1-ba)) for i,c in enumerate(BOOST_BD))
            rr(d, [bx2, by2, bx2+66, by2+24], r=12, fill=bg2, outline=bd2)
            bl_col = tuple(int(c*ba + G_CARD[i]*(1-ba)) for i,c in enumerate(CYAN))
            d.text((bx2+8, by2+7), f"+{BOOST} boost", font=F(10), fill=bl_col)

    # ── Footer ──
    cap = "Temporal boost: +0.25 for memories within resolved time window"
    d.text((W//2 - tw(F(11),cap)//2, H-18), cap, font=F(11), fill=G_MUTE)

    return img


def make_temporal_gif():
    print("→ temporal_boost.gif …")
    frames, dur = [], []

    QLEN = len(QUERY)

    # Phase 1: blank state hold (18 frames)
    for _ in range(18):
        frames.append(draw_frame(0)); dur.append(80)

    # Phase 2: type query (one char per frame, ~40ms each)
    for i in range(1, QLEN + 1):
        frames.append(draw_frame(i)); dur.append(45)

    # Phase 3: hold full query (14 frames)
    for _ in range(14):
        frames.append(draw_frame(QLEN)); dur.append(80)

    # Phase 4: fade in window badge (20 frames)
    for i in range(20):
        a = ease_out(i / 19)
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=a))
        dur.append(55)

    # Phase 5: hold with boost visible (20 frames)
    for _ in range(20):
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=1.0))
        dur.append(75)

    # Phase 6: rerank animation (30 frames)
    for i in range(30):
        t = ease_in_out(i / 29)
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=1.0, rerank_t=t))
        dur.append(55)

    # Phase 7: hold final state (30 frames)
    for _ in range(30):
        frames.append(draw_frame(QLEN, show_window=True, boost_alpha=1.0, rerank_t=1.0))
        dur.append(80)

    p = OUT_DIR / "temporal_boost.gif"
    frames[0].save(
        p, save_all=True, append_images=frames[1:],
        loop=0, duration=dur, optimize=False,
    )
    print(f"  ✓ {p.stat().st_size // 1024} KB, {len(frames)} frames")
    print(f"  Saved → {p}")


if __name__ == "__main__":
    make_temporal_gif()
