"""Notification service for the ADA enforcement system.

Supports:
  - Email notifications via SMTP (configurable)
  - SMS notifications via console log (demo mode)
  - In-app notification log stored in-memory

For demo purposes, email/SMS are logged to console. In production,
configure SMTP credentials and a Twilio/SMS provider.

Usage:
    from src.services.notification_service import NotificationService

    notifier = NotificationService(smtp_host="smtp.gmail.com", ...)
    notifier.send_case_assigned(case, officer)
    notifier.send_new_violation_alert(violations, zone_type)
"""

from __future__ import annotations

import logging
import smtplib
from datetime import datetime, timezone
from email.mime.text import MIMEText
from typing import Optional

logger = logging.getLogger(__name__)


class NotificationService:
    """Service for sending notifications about case events."""

    def __init__(
        self,
        smtp_host: Optional[str] = None,
        smtp_port: int = 587,
        smtp_user: Optional[str] = None,
        smtp_password: Optional[str] = None,
        from_email: Optional[str] = None,
        admin_email: Optional[str] = None,
        demo_mode: bool = True,
    ):
        self.smtp_host = smtp_host
        self.smtp_port = smtp_port
        self.smtp_user = smtp_user
        self.smtp_password = smtp_password
        self.from_email = from_email or "noreply@ada.gov.in"
        self.admin_email = admin_email or "admin@ada.gov.in"
        self.demo_mode = demo_mode
        self._log: list[dict] = []

    # ── Public API ──────────────────────────────────────────────

    def send_new_violation_alert(
        self,
        violation_count: int,
        zone_type: str,
        severity: str,
        case_numbers: list[str],
    ) -> dict:
        """Send an alert when new violations are detected."""
        subject = f"ADA Alert: {violation_count} new violation(s) detected in {zone_type} zone"
        body = (
            f"New violations detected by the ADA AI Change Detection System.\n\n"
            f"Count: {violation_count}\n"
            f"Zone: {zone_type}\n"
            f"Max Severity: {severity}\n"
            f"Cases: {', '.join(case_numbers)}\n\n"
            f"Please log in to the dashboard for details:\n"
            f"http://localhost:8000/dashboard"
        )
        return self._send_notification(
            recipient=self.admin_email,
            subject=subject,
            body=body,
            notification_type="new_violation",
        )

    def send_case_assigned(self, case_number: str, officer_name: str, zone_type: str) -> dict:
        """Send a notification when a case is assigned to an officer."""
        subject = f"ADA Case Assigned: {case_number}"
        body = (
            f"Case {case_number} has been assigned to you.\n\n"
            f"Officer: {officer_name}\n"
            f"Zone: {zone_type}\n\n"
            f"Please verify the violation in the field and update the case status."
        )
        return self._send_notification(
            recipient=self.admin_email,
            subject=subject,
            body=body,
            notification_type="case_assigned",
        )

    def send_case_escalated(self, case_number: str, severity: str, reason: str) -> dict:
        """Send a notification when a case is escalated."""
        subject = f"ADA ESCALATION: {case_number} ({severity.upper()})"
        body = (
            f"Case {case_number} has been escalated.\n\n"
            f"Severity: {severity}\n"
            f"Reason: {reason}\n\n"
            f"Immediate attention required."
        )
        return self._send_notification(
            recipient=self.admin_email,
            subject=subject,
            body=body,
            notification_type="case_escalated",
        )

    def send_notice_issued(self, case_number: str, notice_number: str) -> dict:
        """Send a notification when an enforcement notice is issued."""
        subject = f"ADA Notice Issued: {notice_number}"
        body = (
            f"Enforcement notice has been issued for case {case_number}.\n\n"
            f"Notice Number: {notice_number}\n\n"
            f"The notice PDF is available for download on the dashboard."
        )
        return self._send_notification(
            recipient=self.admin_email,
            subject=subject,
            body=body,
            notification_type="notice_issued",
        )

    def get_notification_log(self, limit: int = 50) -> list[dict]:
        """Return recent notification log entries."""
        return sorted(self._log, key=lambda x: x["sent_at"], reverse=True)[:limit]

    # ── Internal ────────────────────────────────────────────────

    def _send_notification(
        self,
        recipient: str,
        subject: str,
        body: str,
        notification_type: str,
    ) -> dict:
        """Internal: send notification via email (or log in demo mode)."""
        entry = {
            "type": notification_type,
            "recipient": recipient,
            "subject": subject,
            "sent_at": datetime.now(timezone.utc).isoformat(),
            "status": "sent" if not self.demo_mode else "logged",
        }

        if self.demo_mode:
            logger.info(f"[NOTIFICATION] {subject}")
            logger.info(f"  To: {recipient}")
            logger.info(f"  Body: {body[:200]}...")
            entry["status"] = "logged"
        else:
            try:
                self._send_email(recipient, subject, body)
                entry["status"] = "sent"
            except Exception as e:
                logger.error(f"Failed to send email: {e}")
                entry["status"] = "failed"
                entry["error"] = str(e)

        self._log.append(entry)
        return entry

    def _send_email(self, recipient: str, subject: str, body: str) -> None:
        """Send an email via SMTP."""
        if not self.smtp_host:
            raise ValueError("SMTP host not configured")

        msg = MIMEText(body)
        msg["Subject"] = subject
        msg["From"] = self.from_email
        msg["To"] = recipient

        with smtplib.SMTP(self.smtp_host, self.smtp_port) as server:
            server.starttls()
            if self.smtp_user and self.smtp_password:
                server.login(self.smtp_user, self.smtp_password)
            server.send_message(msg)
