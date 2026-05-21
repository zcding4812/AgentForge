from __future__ import annotations

from bs4 import BeautifulSoup


class HtmlHandler:
    name = "html"
    priority = 25

    def matches(self, mime: str, filename: str) -> bool:
        fn = filename.lower()
        return mime in ("text/html", "application/xhtml+xml") or fn.endswith((".html", ".htm"))

    def extract(self, raw: bytes, mime: str, filename: str) -> str:
        soup = BeautifulSoup(raw, "lxml")
        for tag in soup(["script", "style", "noscript"]):
            tag.decompose()
        text = soup.get_text("\n", strip=True)
        return "\n".join(line for line in text.splitlines() if line)
