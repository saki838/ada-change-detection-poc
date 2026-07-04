"""Multi-language translation service for the ADA enforcement dashboard.

Currently supports:
  - en: English (default)
  - hi: Hindi (हिन्दी)

Usage:
    from src.services.translation_service import t, set_language, get_available_languages

    set_language("hi")
    print(t("dashboard.title"))  # एडीए प्रवर्तन डैशबोर्ड
"""

from __future__ import annotations

from typing import Optional

# ── Translation dictionary ──────────────────────────────────────────
# Keys are dot-separated paths for organization.
# Values are dicts with language code -> translated string.

TRANSLATIONS = {
    # Login page
    "login.title": {"en": "ADA Enforcement", "hi": "एडीए प्रवर्तन"},
    "login.subtitle": {
        "en": "Agra Development Authority — Violation Management System",
        "hi": "आगरा विकास प्राधिकरण — उल्लंघन प्रबंधन प्रणाली",
    },
    "login.email": {"en": "Email", "hi": "ईमेल"},
    "login.email_placeholder": {"en": "Enter your email", "hi": "अपना ईमेल दर्ज करें"},
    "login.password": {"en": "Password", "hi": "पासवर्ड"},
    "login.password_placeholder": {"en": "Enter your password", "hi": "अपना पासवर्ड दर्ज करें"},
    "login.sign_in": {"en": "Sign In", "hi": "साइन इन करें"},
    "login.signing_in": {"en": "Signing in...", "hi": "साइन इन हो रहा है..."},
    "login.error.invalid": {"en": "Invalid email or password", "hi": "गलत ईमेल या पासवर्ड"},
    "login.error.expired": {"en": "Session expired", "hi": "सत्र समाप्त हो गया"},

    # Dashboard header
    "dashboard.title": {"en": "Enforcement Dashboard", "hi": "प्रवर्तन डैशबोर्ड"},
    "dashboard.badge": {"en": "Agra Development Authority", "hi": "आगरा विकास प्राधिकरण"},
    "dashboard.sign_out": {"en": "Sign Out", "hi": "साइन आउट"},

    # Stats
    "stats.total_cases": {"en": "Total Cases", "hi": "कुल मामले"},
    "stats.open": {"en": "Open", "hi": "खुले"},
    "stats.critical": {"en": "Critical", "hi": "गंभीर"},
    "stats.high": {"en": "High", "hi": "उच्च"},
    "stats.resolved": {"en": "Resolved", "hi": "हल किए गए"},

    # Case filters
    "filter.all": {"en": "All", "hi": "सभी"},
    "filter.detected": {"en": "Detected", "hi": "पहचाने गए"},
    "filter.assigned": {"en": "Assigned", "hi": "सौंपे गए"},
    "filter.critical": {"en": "Critical", "hi": "गंभीर"},
    "filter.heritage": {"en": "Heritage", "hi": "विरासत"},
    "filter.resolved": {"en": "Resolved", "hi": "हल किए गए"},

    # Case panel
    "cases.title": {"en": "Violation Cases", "hi": "उल्लंघन मामले"},
    "cases.loading": {"en": "Loading cases...", "hi": "मामले लोड हो रहे हैं..."},
    "cases.empty": {"en": "No cases found", "hi": "कोई मामला नहीं मिला"},
    "cases.failed": {"en": "Failed to load cases", "hi": "मामले लोड करने में विफल"},

    # Map legend
    "legend.critical": {"en": "Critical", "hi": "गंभीर"},
    "legend.high": {"en": "High", "hi": "उच्च"},
    "legend.medium": {"en": "Medium", "hi": "मध्यम"},
    "legend.low": {"en": "Low", "hi": "निम्न"},

    # Detail modal
    "detail.violation_class": {"en": "Violation Class", "hi": "उल्लंघन वर्ग"},
    "detail.zone_type": {"en": "Zone Type", "hi": "क्षेत्र प्रकार"},
    "detail.area": {"en": "Area (m²)", "hi": "क्षेत्रफल (m²)"},
    "detail.confidence": {"en": "Confidence", "hi": "विश्वसनीयता"},
    "detail.status": {"en": "Status", "hi": "स्थिति"},
    "detail.assigned_to": {"en": "Assigned To", "hi": "को सौंपा गया"},
    "detail.description": {"en": "Description", "hi": "विवरण"},
    "detail.created": {"en": "Created", "hi": "बनाया गया"},
    "detail.unassigned": {"en": "Unassigned", "hi": "असाइन नहीं किया गया"},
    "detail.no_description": {"en": "No description", "hi": "कोई विवरण नहीं"},

    # Actions
    "action.update_status": {"en": "Update Status", "hi": "स्थिति अपडेट करें"},
    "action.download_notice": {"en": "Download Notice", "hi": "नोटिस डाउनलोड करें"},
    "action.status_updated": {"en": "Case status updated to", "hi": "मामले की स्थिति अपडेट की गई"},
    "action.notice_downloaded": {"en": "Notice PDF downloaded", "hi": "नोटिस पीडीएफ डाउनलोड हुआ"},

    # Status labels
    "status.detected": {"en": "Detected", "hi": "पहचाना गया"},
    "status.assigned": {"en": "Assigned", "hi": "सौंपा गया"},
    "status.field_verified": {"en": "Field Verified", "hi": "क्षेत्र सत्यापित"},
    "status.enforcement_ready": {"en": "Enforcement Ready", "hi": "प्रवर्तन के लिए तैयार"},
    "status.notice_issued": {"en": "Notice Issued", "hi": "नोटिस जारी"},
    "status.resolved": {"en": "Resolved", "hi": "हल किया गया"},
    "status.escalated": {"en": "Escalated", "hi": "बढ़ाया गया"},

    # Severity labels
    "severity.low": {"en": "Low", "hi": "निम्न"},
    "severity.medium": {"en": "Medium", "hi": "मध्यम"},
    "severity.high": {"en": "High", "hi": "उच्च"},
    "severity.critical": {"en": "Critical", "hi": "गंभीर"},

    # Zone types
    "zone.residential": {"en": "Residential", "hi": "आवासीय"},
    "zone.commercial": {"en": "Commercial", "hi": "वाणिज्यिक"},
    "zone.heritage": {"en": "Heritage", "hi": "विरासत"},
    "zone.green_belt": {"en": "Green Belt", "hi": "हरित पट्टी"},
    "zone.riverfront": {"en": "Riverfront", "hi": "नदी तट"},
    "zone.industrial": {"en": "Industrial", "hi": "औद्योगिक"},
    "zone.other": {"en": "Other", "hi": "अन्य"},

    # Violation classes
    "violation.new_construction": {"en": "New Construction", "hi": "नया निर्माण"},
    "violation.horizontal_expansion": {"en": "Horizontal Expansion", "hi": "क्षैतिज विस्तार"},
    "violation.vertical_expansion": {"en": "Vertical Expansion", "hi": "ऊर्ध्वाधर विस्तार"},
    "violation.encroachment": {"en": "Encroachment", "hi": "अतिक्रमण"},
    "violation.unauthorized_paving": {"en": "Unauthorized Paving", "hi": "अनधिकृत पेविंग"},
    "violation.vegetation_clearance": {"en": "Vegetation Clearance", "hi": "वनस्पति हटाना"},
    "violation.other_change": {"en": "Other Change", "hi": "अन्य परिवर्तन"},

    # Toast messages
    "toast.welcome": {"en": "Welcome", "hi": "स्वागत है"},
    "toast.failed": {"en": "Failed", "hi": "विफल"},

    # Analytics
    "analytics.severity": {"en": "Cases by Severity", "hi": "गंभीरता के अनुसार मामले"},
    "analytics.zone": {"en": "Cases by Zone", "hi": "क्षेत्र के अनुसार मामले"},
    "analytics.status": {"en": "Cases by Status", "hi": "स्थिति के अनुसार मामले"},
    "analytics.trend": {"en": "Weekly Trend", "hi": "साप्ताहिक रुझान"},

    # Loading / empty / error
    "common.loading": {"en": "Loading cases...", "hi": "मामले लोड हो रहे हैं..."},
    "common.empty": {"en": "No cases found", "hi": "कोई मामला नहीं मिला"},
    "common.load_error": {"en": "Failed to load cases", "hi": "मामले लोड करने में विफल"},
    "common.signing_in": {"en": "Signing in...", "hi": "साइन इन हो रहा है..."},
    "common.unassigned": {"en": "Unassigned", "hi": "असाइन नहीं किया गया"},
    "common.no_description": {"en": "No description", "hi": "कोई विवरण नहीं"},
    "common.analytics": {"en": "Analytics", "hi": "विश्लेषण"},
    "common.close": {"en": "Close", "hi": "बंद करें"},

    # Toast
    "toast.status_updated_to": {"en": "Case status updated to", "hi": "मामले की स्थिति अपडेट की गई"},
    "toast.failed": {"en": "Failed", "hi": "विफल"},

    # Language
    "language.en": {"en": "English", "hi": "अंग्रेज़ी"},
    "language.hi": {"en": "Hindi", "hi": "हिन्दी"},
}

# ── Current language state ──────────────────────────────────────────

_current_language = "en"


def set_language(lang: str) -> None:
    """Set the current language. Falls back to 'en' if not available."""
    global _current_language
    if lang in ("en", "hi"):
        _current_language = lang
    else:
        _current_language = "en"


def get_language() -> str:
    """Get the current language code."""
    return _current_language


def get_available_languages() -> list[dict]:
    """Return available languages with codes and native names."""
    return [
        {"code": "en", "name": "English", "native_name": "English"},
        {"code": "hi", "name": "Hindi", "native_name": "हिन्दी"},
    ]


def t(key: str, lang: Optional[str] = None) -> str:
    """Translate a key to the current (or specified) language.

    Args:
        key: Dot-separated translation key (e.g. "login.title").
        lang: Optional language override. Uses current language if not set.

    Returns:
        Translated string, or the key itself if not found.

    Usage:
        from src.services.translation_service import t
        print(t("login.title"))  # "ADA Enforcement"
    """
    lang = lang or _current_language
    entry = TRANSLATIONS.get(key)
    if entry is None:
        return key
    return entry.get(lang, entry.get("en", key))


def get_all_translations(lang: str) -> dict:
    """Get all translations flattened for a given language.

    Returns a dict of key -> translated string, for sending to the
    frontend so the dashboard can do client-side translation.
    """
    result = {}
    for key, values in TRANSLATIONS.items():
        result[key] = values.get(lang, values.get("en", key))
    return result
