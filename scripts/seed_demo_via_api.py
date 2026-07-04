"""Seed demo data via the HTTP API (no direct database access needed).

This script calls the ADA Enforcement API to:
  1. Log in and get a JWT token
  2. Create 12 realistic violation cases around Agra
  3. Verify the data by fetching case list and stats

Usage (while uvicorn is running):
    python scripts/seed_demo_via_api.py [--url http://127.0.0.1:8000]

Why use this instead of seed_demo_data.py?
  - No need to stop the server (works while uvicorn is running)
  - Tests that the API endpoints work correctly
  - Can be run from a different machine (uses HTTP)
  - Doesn't require database file access
"""

from __future__ import annotations

import argparse
import sys
from typing import Optional

try:
    import httpx
except ImportError:
    print("❌ httpx is required. Install it with: pip install httpx")
    sys.exit(1)


# ── Demo Cases ──────────────────────────────────────────────────────
# Realistic violations around Agra, Uttar Pradesh, India

DEMO_CASES = [
    {
        "violation_class": "new_construction",
        "confidence": 0.94,
        "area_m2": 340.5,
        "zone_type": "heritage",
        "severity": "critical",
        "status": "detected",
        "lat": 27.1730,
        "lng": 78.0480,
        "assigned_to": None,
        "description": "Unapproved commercial structure detected within 200m of Taj Mahal east gate. Immediate escalation required per heritage zone regulations.",
    },
    {
        "violation_class": "encroachment",
        "confidence": 0.97,
        "area_m2": 890.2,
        "zone_type": "riverfront",
        "severity": "critical",
        "status": "assigned",
        "lat": 27.1900,
        "lng": 78.0100,
        "assigned_to": "Amit Sharma",
        "description": "Encroachment into Yamuna river floodplain — 890m² of unauthorized paving detected. Violates Riverfront Protection Zone.",
    },
    {
        "violation_class": "horizontal_expansion",
        "confidence": 0.88,
        "area_m2": 125.0,
        "zone_type": "commercial",
        "severity": "high",
        "status": "field_verified",
        "lat": 27.2150,
        "lng": 77.9580,
        "assigned_to": "Priya Singh",
        "description": "Unauthorized horizontal expansion of shop front by 125m² into pedestrian ROW. Violation of commercial zone setback requirements.",
    },
    {
        "violation_class": "new_construction",
        "confidence": 0.92,
        "area_m2": 210.0,
        "zone_type": "green_belt",
        "severity": "critical",
        "status": "enforcement_ready",
        "lat": 27.1811,
        "lng": 78.0412,
        "assigned_to": "Amit Sharma",
        "description": "Unauthorized structure in Mehtab Bagh green belt zone. 210m² building detected — no sanctioned plan on record.",
    },
    {
        "violation_class": "vertical_expansion",
        "confidence": 0.85,
        "area_m2": 450.0,
        "zone_type": "industrial",
        "severity": "high",
        "status": "detected",
        "lat": 27.1000,
        "lng": 77.8600,
        "assigned_to": None,
        "description": "Vertical expansion detected in industrial zone — 3 additional floors beyond approved height. Potential safety hazard.",
    },
    {
        "violation_class": "vegetation_clearance",
        "confidence": 0.91,
        "area_m2": 5600.0,
        "zone_type": "heritage",
        "severity": "high",
        "status": "assigned",
        "lat": 27.1794,
        "lng": 78.0213,
        "assigned_to": "Priya Singh",
        "description": "Large-scale vegetation clearance (5,600m²) detected in Agra Fort heritage buffer zone.",
    },
    {
        "violation_class": "vertical_expansion",
        "confidence": 0.78,
        "area_m2": 85.0,
        "zone_type": "commercial",
        "severity": "medium",
        "status": "resolved",
        "lat": 27.2000,
        "lng": 78.0200,
        "assigned_to": "Amit Sharma",
        "description": "Unauthorized rooftop extension in commercial zone. 85m² added without permit. Resolved — structure regularized with penalty.",
    },
    {
        "violation_class": "unauthorized_paving",
        "confidence": 0.82,
        "area_m2": 320.0,
        "zone_type": "residential",
        "severity": "medium",
        "status": "notice_issued",
        "lat": 27.1600,
        "lng": 78.0500,
        "assigned_to": "Priya Singh",
        "description": "Unauthorized paving of residential plot for commercial parking. 320m² paved area without environmental clearance.",
    },
    {
        "violation_class": "encroachment",
        "confidence": 0.95,
        "area_m2": 1500.0,
        "zone_type": "green_belt",
        "severity": "critical",
        "status": "escalated",
        "lat": 27.2400,
        "lng": 78.0300,
        "assigned_to": "Rajesh Kumar",
        "description": "Large-scale encroachment into protected green belt — 1,500m². Referred to High Court monitoring committee.",
    },
    {
        "violation_class": "new_construction",
        "confidence": 0.89,
        "area_m2": 180.0,
        "zone_type": "riverfront",
        "severity": "high",
        "status": "detected",
        "lat": 27.1700,
        "lng": 78.0080,
        "assigned_to": None,
        "description": "New construction detected in Yamuna riverfront restricted zone. 180m² structure within 50m of high-tide line.",
    },
    {
        "violation_class": "horizontal_expansion",
        "confidence": 0.72,
        "area_m2": 45.0,
        "zone_type": "residential",
        "severity": "low",
        "status": "resolved",
        "lat": 27.2150,
        "lng": 78.0050,
        "assigned_to": "Priya Singh",
        "description": "Minor horizontal extension of residential property — 45m². Within permissible limits. Compound wall constructed without prior approval.",
    },
    {
        "violation_class": "horizontal_expansion",
        "confidence": 0.65,
        "area_m2": 30.0,
        "zone_type": "residential",
        "severity": "low",
        "status": "field_verified",
        "lat": 27.2190,
        "lng": 77.9540,
        "assigned_to": "Amit Sharma",
        "description": "Minor setback violation in residential zone — 30m² balcony extension beyond permitted building line.",
    },
]


# ── API Seed Function ───────────────────────────────────────────────

def seed_via_api(base_url: str, email: str = "admin@ada.gov.in", password: str = "admin123") -> dict:
    """Seed demo data via HTTP API. Returns summary statistics."""
    client = httpx.Client(base_url=base_url, timeout=30)

    # 1. Login
    print(f"🔑 Logging in as {email}...")
    resp = client.post("/api/v1/auth/login", json={"email": email, "password": password})
    if resp.status_code != 200:
        print(f"❌ Login failed: {resp.status_code} {resp.text}")
        sys.exit(1)

    data = resp.json()
    token = data["access_token"]
    user = data["user"]
    print(f"   ✅ Logged in as {user['full_name']} ({user['role']})")

    # Set auth header for subsequent requests
    headers = {"Authorization": f"Bearer {token}"}

    # 2. Create cases
    print(f"\n🏗️  Creating {len(DEMO_CASES)} demo cases...")
    created = []
    errors = []

    for i, case_data in enumerate(DEMO_CASES):
        resp = client.post(
            "/api/v1/cases/create",
            json=case_data,
            headers=headers,
        )
        if resp.status_code == 200:
            c = resp.json()
            created.append(c)
            print(f"   ✅ [{i+1:02d}/{len(DEMO_CASES)}] {c['case_number']} — {c['violation_class']} ({c['severity']})")
        else:
            errors.append(f"      ❌ Case {i+1}: {resp.status_code} {resp.text}")
            print(f"   ❌ [{i+1:02d}/{len(DEMO_CASES)}] Failed: {resp.status_code}")

    # Print any errors
    for err in errors:
        print(err)

    # 3. Verify — fetch case list
    print(f"\n📋 Verifying — fetching case list...")
    resp = client.get("/api/v1/cases?limit=50", headers=headers)
    if resp.status_code == 200:
        cases = resp.json()
        print(f"   ✅ Total cases in system: {cases['total']}")
    else:
        print(f"   ❌ Failed to fetch cases: {resp.status_code}")

    # 4. Verify — fetch stats
    print(f"\n📊 Verifying — fetching stats...")
    resp = client.get("/api/v1/cases/stats", headers=headers)
    if resp.status_code == 200:
        stats = resp.json()
        print(f"   ✅ Total: {stats['total']}")
        print(f"      By status: {stats['by_status']}")
        print(f"      By severity: {stats['by_severity']}")
        print(f"      By zone: {stats['by_zone']}")
    else:
        print(f"   ❌ Failed to fetch stats: {resp.status_code}")

    client.close()
    return {"created": len(created), "errors": len(errors), "total": len(DEMO_CASES)}


# ── CLI ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Seed demo data via HTTP API (no database access needed)",
    )
    parser.add_argument(
        "--url",
        default="http://127.0.0.1:8000",
        help="Base URL of the ADA API (default: http://127.0.0.1:8000)",
    )
    parser.add_argument(
        "--email",
        default="admin@ada.gov.in",
        help="Admin email for login (default: admin@ada.gov.in)",
    )
    parser.add_argument(
        "--password",
        default="admin123",
        help="Admin password (default: admin123)",
    )
    args = parser.parse_args()

    print("=" * 60)
    print("🏗️  ADA Demo Data Seeder (via API)")
    print("=" * 60)
    print(f"\n🔗 API URL: {args.url}")
    print(f"👤 User:    {args.email}")
    print()

    result = seed_via_api(args.url, args.email, args.password)

    print()
    print("=" * 60)
    print(f"🎉 Done! {result['created']}/{result['total']} cases created ({result['errors']} errors)")
    print("=" * 60)
    print()
    print("📌 Open http://127.0.0.1:8000 and log in to see the data.")
    print("   (Clear your browser cache / use incognito if you see stale data)")


if __name__ == "__main__":
    main()
