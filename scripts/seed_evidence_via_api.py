"""Bulk evidence seeder — generates drone overlay images and uploads them as
evidence for existing cases via the ADA Enforcement API.

**Synthetic mode** (default): generates procedurally-drawn satellite-style images
with green/brown terrain, road grids, and violation overlays.

**LEVIR-CD mode** (--use-levir): uses real aerial imagery from the LEVIR-CD+
change-detection dataset (1024x1024, 0.5m/px orthophotos from 20 regions in
Texas) as the base backdrop, then applies the same overlay styling on top.

Usage (while uvicorn is running):
    pip install httpx Pillow
    # Synthetic mode
    python scripts/seed_evidence_via_api.py
    # Real LEVIR-CD imagery
    pip install datasets
    python scripts/seed_evidence_via_api.py --use-levir
"""

from __future__ import annotations

import argparse
import io
import math
import pathlib
import random
import sys
from datetime import datetime
from typing import Optional, List, Tuple

try:
    import httpx
except ImportError:
    print("❌ httpx is required. Install it with: pip install httpx")
    sys.exit(1)

try:
    from PIL import Image, ImageDraw, ImageFont, ImageFilter
except ImportError:
    print("❌ Pillow is required. Install it with: pip install Pillow")
    sys.exit(1)


# ── Color palettes by violation class ─────────────────────────────

VIOLATION_COLORS = {
    "new_construction":     {"fill": (220, 60, 60, 100),   "stroke": (200, 30, 30),   "label": "New Construction",    "icon": "🏗️"},
    "encroachment":         {"fill": (60, 100, 220, 100),  "stroke": (30, 60, 200),   "label": "Encroachment",        "icon": "🚧"},
    "horizontal_expansion": {"fill": (240, 160, 40, 100),  "stroke": (220, 130, 20),  "label": "Horizontal Expansion", "icon": "↔️"},
    "vertical_expansion":   {"fill": (200, 80, 200, 100),  "stroke": (170, 50, 170),  "label": "Vertical Expansion",   "icon": "⬆️"},
    "vegetation_clearance": {"fill": (40, 180, 60, 100),   "stroke": (20, 150, 40),   "label": "Vegetation Clearance", "icon": "🌳"},
    "unauthorized_paving":  {"fill": (160, 140, 120, 100), "stroke": (130, 110, 90),  "label": "Unauthorized Paving",  "icon": "🅿️"},
}

DEFAULT_COLOR = {"fill": (180, 180, 80, 100), "stroke": (150, 150, 50), "label": "Violation", "icon": "⚠️"}

LEVIR_DIR = pathlib.Path("data/levir_cd")


# ── Synthetic landscape generator (fallback) ─────────────────────

def _generate_landscape_base(draw, w: int, h: int):
    """Draw a satellite-style landscape background."""
    for y in range(int(h * 0.05)):
        t = y / (h * 0.05) if h * 0.05 else 1
        r = int(100 + t * 30)
        g = int(140 + t * 20)
        b = int(180 - t * 10)
        draw.line([(0, y), (w, y)], fill=(r, g, b))
    for y in range(int(h * 0.05), h):
        noise = random.randint(-15, 15)
        if y < h * 0.4:
            r = int(80 + noise + (y / h) * 30)
            g = int(130 + noise + (y / h) * 20)
            b = int(50 + noise)
        elif y < h * 0.7:
            r = int(110 + noise + (y / h) * 20)
            g = int(120 + noise)
            b = int(50 + noise)
        else:
            r = int(140 + noise + (y / h) * 20)
            g = int(120 + noise)
            b = int(80 + noise)
        draw.line([(0, y), (w, y)], fill=(max(0, min(255, r)), max(0, min(255, g)), max(0, min(255, b))))


def _draw_road_grid(draw, w: int, h: int):
    """Draw a faint road/grid network."""
    for x in range(0, w, random.randint(30, 70)):
        draw.line([(x, 0), (x, h)], fill=(180, 170, 150, random.randint(30, 60)), width=1)
    for y in range(0, h, random.randint(40, 80)):
        draw.line([(0, y), (w, y)], fill=(180, 170, 150, random.randint(20, 50)), width=1)


# ── Violation polygon overlay ─────────────────────────────────────

def _make_violation_overlay(w: int, h: int, colors: dict, seed: str) -> Tuple[Image.Image, List[Tuple[int, int]], Tuple[int, int]]:
    """Create a semi-transparent violation polygon overlay.

    Returns (overlay_RGBA_image, polygon_vertices, (cx, cy)).
    """
    random.seed(seed)
    cx, cy = w // 2, h // 2
    radius_x = random.randint(int(w * 0.15), int(w * 0.35))
    radius_y = random.randint(int(h * 0.15), int(h * 0.30))
    num_points = random.randint(6, 10)

    polygon = []
    for i in range(num_points):
        angle = (2 * math.pi / num_points) * i + random.uniform(-0.3, 0.3)
        rx = radius_x * (0.75 + random.random() * 0.5)
        ry = radius_y * (0.75 + random.random() * 0.5)
        polygon.append((int(cx + rx * math.cos(angle)), int(cy + ry * math.sin(angle))))

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    od = ImageDraw.Draw(overlay)
    od.polygon(polygon, fill=colors["fill"])
    overlay = overlay.filter(ImageFilter.GaussianBlur(radius=2))
    return overlay, polygon, (cx, cy)


def _draw_overlay_graphics(draw, polygon: List[Tuple[int, int]], cx: int, cy: int, colors: dict):
    """Draw polygon outline, dashed bounding box, and crosshair on the given draw surface."""
    draw.polygon(polygon, outline=colors["stroke"], width=3)
    min_x = min(p[0] for p in polygon)
    min_y = min(p[1] for p in polygon)
    max_x = max(p[0] for p in polygon)
    max_y = max(p[1] for p in polygon)
    dash_len = 6
    for i in range(min_x, max_x, dash_len * 2):
        draw.line([(i, min_y), (min(i + dash_len, max_x), min_y)], fill=(255, 255, 255, 180), width=1)
        draw.line([(i, max_y), (min(i + dash_len, max_x), max_y)], fill=(255, 255, 255, 180), width=1)
    for i in range(min_y, max_y, dash_len * 2):
        draw.line([(min_x, i), (min_x, min(i + dash_len, max_y))], fill=(255, 255, 255, 180), width=1)
        draw.line([(max_x, i), (max_x, min(i + dash_len, max_y))], fill=(255, 255, 255, 180), width=1)
    draw.line([(cx - 10, cy), (cx + 10, cy)], fill=(255, 255, 255), width=2)
    draw.line([(cx, cy - 10), (cx, cy + 10)], fill=(255, 255, 255), width=2)


def _draw_info_panel_overlay(w: int, h: int, accent_color: Tuple[int, int, int]) -> Image.Image:
    """Return a full-size RGBA overlay with a semi-transparent info panel at the top."""
    panel_h = 90
    panel = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    pd = ImageDraw.Draw(panel)
    pd.rectangle([(0, 0), (w, panel_h)], fill=(15, 25, 45, 200))
    pd.rectangle([(0, panel_h - 4), (w, panel_h)], fill=accent_color)
    return panel


def _draw_info_text(img: Image.Image, w: int, h: int, colors: dict, case_number: str,
                    area_m2: float, confidence: float, zone_type: str, evidence_type: str,
                    source_label: str = "Drone survey"):
    """Draw text labels on the info panel.

    Args:
        source_label: Shown as "Source: ..." in the info panel and in the
                      bottom watermark (e.g. "Drone survey" or "LEVIR-CD+ aerial survey").
    """
    draw = ImageDraw.Draw(img)
    try:
        font_large = ImageFont.truetype("arial.ttf", 18)
        font_small = ImageFont.truetype("arial.ttf", 13)
        font_tiny = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font_large = font_small = font_tiny = ImageFont.load_default()
    label = colors["label"]
    icon = colors["icon"]
    text_y = 10
    draw.text((14, text_y), f"{icon}  {label} — {case_number}", fill=(255, 255, 255), font=font_large)
    text_y += 26
    draw.text((14, text_y), f"Zone: {zone_type.replace('_', ' ').title()}  |  "
              f"Area: {area_m2:.1f} m²  |  Confidence: {confidence:.0%}",
              fill=(200, 210, 230), font=font_small)
    text_y += 20
    draw.text((14, text_y), f"Source: {source_label} ({datetime.now().strftime('%d %b %Y')})  |  "
              f"Type: {evidence_type.replace('_', ' ').title()}",
              fill=(160, 175, 200), font=font_tiny)
    dot_x, dot_y = w - 120, 12
    draw.ellipse([(dot_x, dot_y), (dot_x + 14, dot_y + 14)], fill=colors["stroke"])
    draw.text((dot_x + 20, dot_y + 1), label, fill=colors["stroke"], font=font_small)
    draw.text((w - 160, h - 16), f"ADA Enforcement · {source_label}",
              fill=(120, 130, 150), font=font_tiny)


# ── LEVIR-CD dataset loader ───────────────────────────────────────

def _ensure_levir_downloaded():
    """Download LEVIR-CD+ from Hugging Face if not already present on disk."""
    if LEVIR_DIR.exists() and any(LEVIR_DIR.rglob("*.png")):
        print("   ✅ LEVIR-CD dataset already downloaded")
        return
    print("   📥 LEVIR-CD+ not found locally. Downloading from Hugging Face...")
    print("      (This is ~2.8 GB and may take several minutes)")
    try:
        from datasets import load_dataset
    except ImportError:
        print("   ❌ 'datasets' package is required for LEVIR mode.")
        print("      Install it with: pip install datasets")
        sys.exit(1)

    ds = load_dataset("blanchon/LEVIR_CDPlus")
    train_val = ds["train"].train_test_split(test_size=0.1, seed=0)
    splits = {"train": train_val["train"], "val": train_val["test"], "test": ds["test"]}

    for split_name, split_ds in splits.items():
        a_dir = LEVIR_DIR / split_name / "A"
        b_dir = LEVIR_DIR / split_name / "B"
        label_dir = LEVIR_DIR / split_name / "label"
        for d in (a_dir, b_dir, label_dir):
            d.mkdir(parents=True, exist_ok=True)
        print(f"      Exporting {split_name}: {len(split_ds)} pairs...")
        for i, row in enumerate(split_ds):
            row["image1"].save(a_dir / f"{i:05d}.png")
            row["image2"].save(b_dir / f"{i:05d}.png")
            row["mask"].save(label_dir / f"{i:05d}.png")

    print("   ✅ LEVIR-CD dataset ready at data/levir_cd/")


def _load_levir_images(index: int) -> Tuple[Image.Image, Image.Image, Optional[Image.Image]]:
    """Load a LEVIR-CD image pair (A, B, mask) by index.

    Loops around splits to provide a variety of scenes. Returns
    (pre_change, post_change, change_mask_or_None).
    """
    splits = ["train", "val", "test"]
    split_name = splits[index % len(splits)]
    pair_idx = (index // len(splits)) % 200  # plenty of images per split
    a_path = LEVIR_DIR / split_name / "A" / f"{pair_idx:05d}.png"
    b_path = LEVIR_DIR / split_name / "B" / f"{pair_idx:05d}.png"
    mask_path = LEVIR_DIR / split_name / "label" / f"{pair_idx:05d}.png"

    a_img = Image.open(a_path).convert("RGB") if a_path.exists() else None
    b_img = Image.open(b_path).convert("RGB") if b_path.exists() else None
    mask = Image.open(mask_path).convert("L") if mask_path.exists() else None
    # Fallback: if the specific index doesn't exist, grab the first available
    if a_img is None or b_img is None:
        for sp in splits:
            a_dir = LEVIR_DIR / sp / "A"
            files = sorted(a_dir.glob("*.png"))
            if files:
                fname = files[pair_idx % len(files)].name
                a_img = Image.open(a_dir / fname).convert("RGB")
                b_img = Image.open(LEVIR_DIR / sp / "B" / fname).convert("RGB")
                mf = LEVIR_DIR / sp / "label" / fname
                mask = Image.open(mf).convert("L") if mf.exists() else None
                break
    return a_img, b_img, mask


# ── Core image generators ─────────────────────────────────────────

def generate_synthetic_image(
    case_number: str,
    violation_class: str = "new_construction",
    area_m2: float = 300.0,
    confidence: float = 0.92,
    zone_type: str = "heritage",
    evidence_type: str = "drone_imagery",
    image_size: tuple = (640, 480),
) -> bytes:
    """Generate a fully synthetic drone/overlay image (original behavior)."""
    w, h = image_size
    colors = VIOLATION_COLORS.get(violation_class, DEFAULT_COLOR)
    seed_key = f"{case_number}_{evidence_type}"
    random.seed(seed_key)

    img = Image.new("RGB", (w, h), (100, 120, 80))
    draw = ImageDraw.Draw(img, "RGBA")
    _generate_landscape_base(draw, w, h)
    _draw_road_grid(draw, w, h)

    overlay_img, polygon, (cx, cy) = _make_violation_overlay(w, h, colors, seed_key)
    img = Image.alpha_composite(img.convert("RGBA"), overlay_img)
    draw = ImageDraw.Draw(img, "RGBA")
    _draw_overlay_graphics(draw, polygon, cx, cy, colors)

    panel = _draw_info_panel_overlay(w, h, colors["stroke"])
    img = Image.alpha_composite(img, panel)
    _draw_info_text(img, w, h, colors, case_number, area_m2, confidence, zone_type,
                     evidence_type, source_label="Drone survey")

    if evidence_type == "change_detection_overlay":
        heat = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(heat)
        gd.polygon(polygon, fill=(255, 60, 0, 60))
        img = Image.alpha_composite(img, heat)
        border = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(border)
        bd.polygon(polygon, outline=(255, 120, 0, 120), width=6)
        img = Image.alpha_composite(img, border)

    img_rgb = img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=82)
    buf.seek(0)
    return buf.getvalue()


def generate_levir_evidence_image(
    case_index: int,
    case_number: str,
    violation_class: str = "new_construction",
    area_m2: float = 300.0,
    confidence: float = 0.92,
    zone_type: str = "heritage",
    evidence_type: str = "drone_imagery",
    image_size: tuple = (640, 480),
) -> bytes:
    """Generate an evidence image using a real LEVIR-CD+ aerial photo as base.

    For ``drone_imagery``: uses the post-change image (B) with overlay.
    For ``field_photo``: uses the pre-change image (A) with a softer treatment.
    For ``change_detection_overlay``: composites the actual change mask from
    LEVIR-CD in orange-red on top of the post-change image.
    """
    w, h = image_size
    colors = VIOLATION_COLORS.get(violation_class, DEFAULT_COLOR)
    seed_key = f"levir_{case_number}_{evidence_type}"
    random.seed(seed_key)

    # Load real LEVIR-CD imagery
    a_img, b_img, mask = _load_levir_images(case_index)

    # Choose base image: post-change (B) by default, pre-change (A) for field_photo
    if evidence_type == "field_photo" and a_img is not None:
        base = a_img.copy()
    elif b_img is not None:
        base = b_img.copy()
    else:
        # Fallback: synthetic if no LEVIR image available
        return generate_synthetic_image(case_number, violation_class, area_m2,
                                         confidence, zone_type, evidence_type, image_size)

    base = base.resize((w, h), Image.LANCZOS)
    base_rgba = base.convert("RGBA")

    # Load font for change-detect label
    try:
        font_tiny = ImageFont.truetype("arial.ttf", 11)
    except (IOError, OSError):
        font_tiny = ImageFont.load_default()

    # Create violation polygon overlay
    overlay_img, polygon, (cx, cy) = _make_violation_overlay(w, h, colors, seed_key)
    img = Image.alpha_composite(base_rgba, overlay_img)

    # Draw overlay graphics on the composited image
    draw = ImageDraw.Draw(img, "RGBA")
    _draw_overlay_graphics(draw, polygon, cx, cy, colors)

    # For change_detection_overlay: use the real change mask from LEVIR-CD
    if evidence_type == "change_detection_overlay" and mask is not None:
        mask_resized = mask.resize((w, h), Image.LANCZOS)
        # Create a colored overlay from the change mask using paste (vectorized)
        change_overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        # Threshold mask: white pixels -> orange-red tint at higher opacity for visibility
        mask_alpha = mask_resized.point(lambda p: 160 if p > 0 else 0)
        change_overlay.paste((255, 60, 0), mask=mask_alpha)
        # Also draw a semi-transparent orange fill over the whole overlay for contrast
        border = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(border)
        bd.polygon(polygon, outline=(255, 120, 0, 200), width=5)
        img = Image.alpha_composite(img, border)
        img = Image.alpha_composite(img, change_overlay)
        # Add a small "🔥 CHANGE DETECTED" label
        draw.text((w - 180, 120), "🔥 CHANGE DETECTED", fill=(255, 80, 0, 220), font=font_tiny)
    elif evidence_type == "change_detection_overlay":
        # Fallback: synthetic heatmap overlay if no mask available
        heat = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        gd = ImageDraw.Draw(heat)
        gd.polygon(polygon, fill=(255, 60, 0, 60))
        img = Image.alpha_composite(img, heat)
        border = Image.new("RGBA", (w, h), (0, 0, 0, 0))
        bd = ImageDraw.Draw(border)
        bd.polygon(polygon, outline=(255, 120, 0, 120), width=6)
        img = Image.alpha_composite(img, border)

    # For field_photo: add a subtle blue tint and "PRE-CHANGE" label
    if evidence_type == "field_photo":
        # Slight blue/cyan tint to distinguish pre-change from post-change
        tint = Image.new("RGBA", (w, h), (0, 40, 80, 30))
        img = Image.alpha_composite(img, tint)
        try:
            font_tiny_fp = ImageFont.truetype("arial.ttf", 12)
        except (IOError, OSError):
            font_tiny_fp = ImageFont.load_default()
        draw.text((14, 98), "📅 Pre-Change Reference", fill=(100, 180, 255, 200), font=font_tiny_fp)

    # Add info panel
    panel = _draw_info_panel_overlay(w, h, colors["stroke"])
    img = Image.alpha_composite(img, panel)
    _draw_info_text(img, w, h, colors, case_number, area_m2, confidence, zone_type,
                     evidence_type, source_label="LEVIR-CD+ aerial survey")

    # Export as JPEG
    img_rgb = img.convert("RGB")
    buf = io.BytesIO()
    img_rgb.save(buf, format="JPEG", quality=85)
    buf.seek(0)
    return buf.getvalue()




# ── API Evidence Seeder ─────────────────────────────────────────────


def get_token(client: httpx.Client, email: str, password: str) -> dict:
    """Log in and return auth headers."""
    resp = client.post("/api/v1/auth/login", json={
        "email": email,
        "password": password,
    })
    if resp.status_code != 200:
        print(f"❌ Login failed: {resp.status_code} {resp.text}")
        sys.exit(1)
    data = resp.json()
    token = data["access_token"]
    user = data["user"]
    print(f"   ✅ Logged in as {user['full_name']} ({user['role']})")
    return {"Authorization": f"Bearer {token}"}


def list_cases(client: httpx.Client, headers: dict) -> List[dict]:
    """Get all existing cases."""
    resp = client.get("/api/v1/cases?limit=200", headers=headers)
    if resp.status_code != 200:
        print(f"❌ Failed to list cases: {resp.status_code} {resp.text}")
        return []
    data = resp.json()
    cases = data.get("cases", [])
    print(f"   ✅ Found {len(cases)} cases in the system")
    return cases


def upload_evidence(
    client: httpx.Client,
    headers: dict,
    case_id: int,
    image_bytes: bytes,
    evidence_type: str,
    description: str,
) -> bool:
    """Upload an image as evidence for a case. Returns True on success."""
    files = {
        "file": (f"{evidence_type}_{case_id}.jpg", image_bytes, "image/jpeg"),
    }
    params = {
        "evidence_type": evidence_type,
        "description": description,
        "uploaded_by": "Drone Survey System",
    }
    resp = client.post(
        f"/api/v1/cases/{case_id}/evidence",
        headers=headers,
        files=files,
        params=params,
    )
    if resp.status_code == 200:
        return True
    else:
        print(f"         ❌ Upload failed: {resp.status_code} {resp.text[:200]}")
        return False


def seed_evidence(
    base_url: str,
    email: str = "admin@ada.gov.in",
    password: str = "admin123",
    max_cases: Optional[int] = None,
    use_levir: bool = False,
) -> dict:
    """Main seeding function. Returns stats."""
    client = httpx.Client(base_url=base_url, timeout=60)

    # 0. Download LEVIR-CD if requested
    if use_levir:
        print("🌍 Checking LEVIR-CD dataset...")
        _ensure_levir_downloaded()

    # 1. Login
    print("🔑 Logging in...")
    headers = get_token(client, email, password)

    # 2. List cases
    print("\n📋 Fetching cases...")
    cases = list_cases(client, headers)

    if not cases:
        print("   ⚠️  No cases found. Run seed_demo_via_api.py first.")
        client.close()
        return {"processed": 0, "uploaded": 0, "failed": 0}

    if max_cases:
        cases = cases[:max_cases]

    # 3. Generate and upload evidence for each case
    source_label = "LEVIR-CD+ aerial" if use_levir else "synthetic"
    print(f"\n📸 Generating & uploading {source_label} evidence for {len(cases)} cases...")
    total_uploaded = 0
    total_failed = 0

    for i, c in enumerate(cases):
        case_id = c["id"]
        case_number = c.get("case_number", f"#{case_id}")
        vclass = c.get("violation_class", "new_construction")
        area = c.get("area_m2", 300.0)
        confidence = c.get("confidence", 0.90)
        zone = c.get("zone_type", "other")

        # Determine which evidence types to generate based on case
        ev_types = ["drone_imagery"]
        if i % 2 == 0:
            ev_types.append("change_detection_overlay")
        if i % 3 == 0:
            ev_types.append("field_photo")

        print(f"\n   [{i+1}/{len(cases)}] {case_number} ({vclass})")

        for ev_type in ev_types:
            if use_levir:
                img_bytes = generate_levir_evidence_image(
                    case_index=i,
                    case_number=case_number,
                    violation_class=vclass,
                    area_m2=area,
                    confidence=confidence,
                    zone_type=zone,
                    evidence_type=ev_type,
                )
            else:
                img_bytes = generate_synthetic_image(
                    case_number=case_number,
                    violation_class=vclass,
                    area_m2=area,
                    confidence=confidence,
                    zone_type=zone,
                    evidence_type=ev_type,
                )

            # Build a realistic description
            if ev_type == "drone_imagery":
                desc = f"{'LEVIR-CD+ aerial' if use_levir else 'Drone survey'} imagery — {vclass.replace('_', ' ').title()} detected in {zone} zone"
            elif ev_type == "change_detection_overlay":
                desc = f"AI change detection overlay showing {area:.0f}m² violation footprint{' (LEVIR-CD+ change mask)' if use_levir else ''}"
            else:
                desc = f"Ground-level field photo of {vclass.replace('_', ' ').title()} violation"

            size_kb = len(img_bytes) / 1024
            size_str = f"{size_kb:.0f}KB" if size_kb > 10 else f"{size_kb:.1f}KB"

            success = upload_evidence(client, headers, case_id, img_bytes, ev_type, desc)
            if success:
                total_uploaded += 1
                print(f"         ✅ {ev_type.replace('_', ' ').title()} ({size_str})")
            else:
                total_failed += 1

    client.close()

    print()
    print("=" * 60)
    print(f"🎉 Done! {total_uploaded} evidence images uploaded ({total_failed} failed)")
    print(f"   Processed {len(cases)} cases using {source_label} imagery")
    print("=" * 60)

    return {
        "cases_processed": len(cases),
        "uploaded": total_uploaded,
        "failed": total_failed,
    }


# ── CLI ─────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="Generate and upload drone evidence images to existing cases",
    )
    parser.add_argument(
        "--url", default="http://127.0.0.1:8000",
        help="Base URL of the ADA API (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--email", default="admin@ada.gov.in",
        help="Admin email (default: admin@ada.gov.in)",
    )
    parser.add_argument(
        "--password", default="admin123",
        help="Admin password (default: admin123)",
    )
    parser.add_argument(
        "--max-cases", type=int, default=None,
        help="Max cases to process (default: all)",
    )
    parser.add_argument(
        "--use-levir", action="store_true",
        help="Use real LEVIR-CD+ aerial imagery instead of synthetic landscape",
    )
    args = parser.parse_args()

    mode = "LEVIR-CD+ aerial" if args.use_levir else "synthetic"
    print("=" * 60)
    print(f"📸 ADA Evidence Seeder — {mode} Imagery Upload")
    print("=" * 60)
    print(f"\n🔗 API URL: {args.url}")
    print(f"👤 User:    {args.email}")
    print(f"🖼️  Mode:    {mode}")
    if args.max_cases:
        print(f"📋 Cases:   max {args.max_cases}")
    print()

    seed_evidence(args.url, args.email, args.password, args.max_cases, args.use_levir)

    if args.use_levir:
        print()
        print("📌 Open the dashboard and click any case to see real LEVIR-CD aerial imagery.")
    else:
        print()
        print("📌 Tip: Use --use-levir to generate evidence from real LEVIR-CD+ aerial photos.")
    print()


if __name__ == "__main__":
    main()
