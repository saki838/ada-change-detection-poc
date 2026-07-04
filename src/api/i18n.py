"""FastAPI router for multi-language (i18n) support.

Endpoints:
    GET /api/v1/i18n/translations?lang=hi  — Get all translations for a language
    GET /api/v1/i18n/languages              — List available languages
"""

from __future__ import annotations

from fastapi import APIRouter, Query
from pydantic import BaseModel

from src.services.translation_service import (
    get_all_translations,
    get_available_languages,
)

router = APIRouter(prefix="/api/v1/i18n", tags=["i18n"])


class LanguageInfo(BaseModel):
    code: str
    name: str
    native_name: str


class LanguagesResponse(BaseModel):
    languages: list[LanguageInfo]
    default: str = "en"


class TranslationsResponse(BaseModel):
    lang: str
    translations: dict


@router.get(
    "/languages",
    response_model=LanguagesResponse,
    summary="List available dashboard languages",
)
def list_languages():
    """Return the list of supported languages."""
    langs = get_available_languages()
    return LanguagesResponse(
        languages=[LanguageInfo(**l) for l in langs],
        default="en",
    )


@router.get(
    "/translations",
    response_model=TranslationsResponse,
    summary="Get all translations for a given language",
)
def get_translations(
    lang: str = Query("en", description="Language code (en, hi)"),
):
    """Return all UI strings translated into the requested language."""
    translations = get_all_translations(lang)
    return TranslationsResponse(lang=lang, translations=translations)
