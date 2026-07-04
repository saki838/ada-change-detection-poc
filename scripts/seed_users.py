"""Seed the database with initial users for the ADA enforcement system.

Creates:
  - Admin account (full access)
  - Supervisor account
  - Enforcement officer account (2)

Usage:
    python scripts/seed_users.py [--db-path data/ada.db]

After seeding, you can log in to the dashboard with:
    admin@ada.gov.in / admin123
"""

from __future__ import annotations

import argparse
import os
import sys

# Ensure the project root is on sys.path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from sqlalchemy import text
from sqlalchemy.orm import Session

from src.database.connection import get_engine
from src.database.models import UserRole, init_db
from src.services.auth_service import AuthService


# ── Default seed users ──────────────────────────────────────────────

SEED_USERS = [
    {
        "username": "admin",
        "email": "admin@ada.gov.in",
        "password": "admin123",
        "full_name": "Agra Development Authority Admin",
        "role": UserRole.ADMIN,
        "phone": "+91-9876543210",
    },
    {
        "username": "supervisor1",
        "email": "supervisor@ada.gov.in",
        "password": "super123",
        "full_name": "Rajesh Kumar — Zonal Supervisor",
        "role": UserRole.SUPERVISOR,
        "phone": "+91-9876543211",
    },
    {
        "username": "officer_sharma",
        "email": "sharma@ada.gov.in",
        "password": "field123",
        "full_name": "Amit Sharma — Enforcement Officer",
        "role": UserRole.ENFORCEMENT_OFFICER,
        "phone": "+91-9876543212",
    },
    {
        "username": "officer_singh",
        "email": "singh@ada.gov.in",
        "password": "field123",
        "full_name": "Priya Singh — Enforcement Officer",
        "role": UserRole.ENFORCEMENT_OFFICER,
        "phone": "+91-9876543213",
    },
]


def seed_users(db_path: str, users: list[dict] | None = None) -> int:
    """Seed users into the database. Returns count of new users created."""
    users = users or SEED_USERS
    init_db(db_path)
    engine = get_engine(db_path)
    created = 0

    with Session(engine) as session:
        for user_data in users:
            # Check if user already exists
            result = session.execute(
                text("SELECT id FROM users WHERE email = :email"),
                {"email": user_data["email"]},
            )
            if result.scalar() is not None:
                print(f"  ⏭️  {user_data['email']} — already exists")
                continue

            AuthService.create_user(
                session,
                username=user_data["username"],
                email=user_data["email"],
                password=user_data["password"],
                full_name=user_data["full_name"],
                role=user_data["role"],
                phone=user_data.get("phone"),
            )
            print(f"  ✅ {user_data['email']} — {user_data['role']}")
            created += 1

    return created


def main():
    parser = argparse.ArgumentParser(description="Seed ADA enforcement users")
    parser.add_argument(
        "--db-path",
        default="data/ada.db",
        help="Path to the SQLite database (default: data/ada.db)",
    )
    args = parser.parse_args()

    print("🌱 Seeding users...")
    print(f"   Database: {args.db_path}")
    print()

    created = seed_users(args.db_path)

    print()
    if created > 0:
        print(f"🎉 {created} new user(s) created!")
    else:
        print("✅ All users already exist — nothing to do.")

    print()
    print("Login credentials:")
    print("  Admin:       admin@ada.gov.in / admin123")
    print("  Supervisor:  supervisor@ada.gov.in / super123")
    print("  Officer:     sharma@ada.gov.in / field123")
    print("  Officer:     singh@ada.gov.in / field123")


if __name__ == "__main__":
    main()
