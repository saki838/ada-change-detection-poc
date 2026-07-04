"""FastAPI router for notification management.

Endpoints:
    GET  /api/v1/notifications/log    — Get recent notification log
    POST /api/v1/notifications/test   — Send a test notification
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.services.notification_service import NotificationService

router = APIRouter(prefix="/api/v1/notifications", tags=["notifications"])

# Singleton notifier instance (demo mode by default)
_notifier = NotificationService(demo_mode=True)


class NotificationEntry(BaseModel):
    type: str
    recipient: str
    subject: str
    sent_at: str
    status: str
    error: str | None = None


class NotificationLogResponse(BaseModel):
    total: int
    entries: list[NotificationEntry]


class TestNotificationRequest(BaseModel):
    notification_type: str = "new_violation"  # new_violation | case_assigned | case_escalated | notice_issued
    message: str = "This is a test notification from the ADA system."


class TestNotificationResponse(BaseModel):
    success: bool
    message: str
    entry: NotificationEntry


@router.get(
    "/log",
    response_model=NotificationLogResponse,
    summary="Get recent notification log",
)
def get_log(limit: int = 50):
    """Return the most recent notification log entries."""
    entries = _notifier.get_notification_log(limit=limit)
    return NotificationLogResponse(
        total=len(entries),
        entries=[NotificationEntry(**e) for e in entries],
    )


@router.post(
    "/test",
    response_model=TestNotificationResponse,
    summary="Send a test notification",
)
def send_test(body: TestNotificationRequest):
    """Send a test notification to verify the notification system works."""
    entry = _notifier._send_notification(
        recipient="test@ada.gov.in",
        subject=f"ADA Test: {body.notification_type}",
        body=body.message,
        notification_type=body.notification_type,
    )
    return TestNotificationResponse(
        success=entry["status"] != "failed",
        message=f"Test notification sent ({entry['status']})",
        entry=NotificationEntry(**entry),
    )
