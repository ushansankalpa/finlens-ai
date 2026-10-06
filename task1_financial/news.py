from __future__ import annotations

import hashlib
from datetime import datetime, timedelta, timezone
from email.utils import parsedate_to_datetime
from urllib.parse import urlencode

import feedparser
import requests
import yfinance as yf

from shared.io import utc_now
from task1_financial.data import validate_ticker
from task1_financial.schemas import Headline


def get_news(ticker: str, n: int = 10, company_name: str | None = None,
             max_age_days: int = 30) -> tuple[list[Headline], list[str]]:
    ticker = validate_ticker(ticker)
    if not 1 <= n <= 50:
        raise ValueError("Headline count must be between 1 and 50")
    warnings, items = [], []
    try:
        raw = yf.Ticker(ticker).get_news(count=max(n * 2, 20))
        for item in raw or []:
            content = item.get("content") or item
            provider = content.get("provider") or {}
            canonical = content.get("canonicalUrl") or {}
            timestamp = content.get("pubDate") or content.get("providerPublishTime")
            if isinstance(timestamp, (int, float)):
                timestamp = datetime.fromtimestamp(timestamp, timezone.utc).isoformat()
            items.append({"title": content.get("title"), "source": provider.get("displayName") or
                          content.get("publisher") or "Yahoo Finance", "url": canonical.get("url") or
                          content.get("link"), "published_at": timestamp})
    except Exception:
        warnings.append("Yahoo news retrieval failed; trying Google News RSS")
    # Fallback is a real source, not a fabricated fixture.
    try:
        query = f'"{company_name or ticker}" stock when:{max_age_days}d'
        response = requests.get("https://news.google.com/rss/search?" + urlencode(
            {"q": query, "hl": "en-US", "gl": "US", "ceid": "US:en"}), timeout=20)
        response.raise_for_status()
        for entry in feedparser.parse(response.content).entries:
            source = entry.get("source", {}).get("title", "Google News RSS")
            items.append({"title": entry.get("title"), "source": source, "url": entry.get("link"),
                          "published_at": entry.get("published")})
    except Exception:
        warnings.append("RSS news retrieval unavailable")
    found, seen = [], set()
    cutoff = datetime.now(timezone.utc) - timedelta(days=max_age_days)
    for item in items:
        title = (item.get("title") or "").strip()
        key = " ".join(title.casefold().split())
        if not key or key in seen:
            continue
        timestamp = item.get("published_at")
        try:
            stamp = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
        except (ValueError, AttributeError):
            try:
                stamp = parsedate_to_datetime(timestamp)
            except (TypeError, ValueError):
                # Recency cannot be established; do not count toward requirement.
                continue
        if stamp.tzinfo is None:
            stamp = stamp.replace(tzinfo=timezone.utc)
        if not cutoff <= stamp <= datetime.now(timezone.utc) + timedelta(hours=1):
            continue
        seen.add(key)
        found.append(Headline(id=hashlib.sha256(key.encode()).hexdigest()[:16], title=title,
                              source=item["source"], url=item.get("url"),
                              published_at=stamp.isoformat(), retrieved_at=utc_now()))
    found.sort(key=lambda h: h.published_at or "", reverse=True)
    if len(found) < n:
        warnings.append(f"Only {len(found)} unique recent headlines available; required {n}")
    return found[:n], warnings
