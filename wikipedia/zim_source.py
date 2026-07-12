"""Read articles straight from the Kiwix ZIM via libzim.

Iterates entries by id so a single integer cursor can resume the whole job.
Skips redirects and non-article assets, strips HTML, and drops the appendix
sections (References / External links / ...) which are mostly citation noise —
cutting them roughly halves chunk count with little loss of answerable content.
"""
import re
from libzim.reader import Archive
from config import cfg

_APPENDIX = re.compile(
    r'id="(References|External_links|See_also|Notes|Further_reading|'
    r'Bibliography|Citations|Sources)"', re.I)


def _strip_html(h):
    h = re.sub(r'(?is)<(script|style).*?</\1>', ' ', h)
    h = re.sub(r'(?s)<[^>]+>', ' ', h)
    h = re.sub(r'&[a-z#0-9]+;', ' ', h)
    return re.sub(r'\s+', ' ', h).strip()


def clean_article(html):
    m = _APPENDIX.search(html)
    if m:
        html = html[:m.start()]
    return _strip_html(html)


class ZimSource:
    def __init__(self, path=None):
        self.archive = Archive(path or cfg.zim_path)
        self.entry_count = self.archive.entry_count

    def iter_from(self, start_id):
        """Yield (entry_id, title, cleaned_text) for real articles from start_id."""
        i = int(start_id)
        n = self.entry_count
        while i < n:
            eid = i
            i += 1
            try:
                e = self.archive._get_entry_by_id(eid)
                if e.is_redirect:
                    continue
                item = e.get_item()
                if item.mimetype != "text/html":
                    continue
                html = bytes(item.content).decode("utf-8", "ignore")
                title = e.title
            except Exception:
                continue
            text = clean_article(html)
            if len(text) < cfg.min_article_chars:
                continue
            yield eid, title, text
