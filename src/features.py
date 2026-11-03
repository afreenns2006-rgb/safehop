"""Extract numeric, URL-only indicators without contacting the host."""

import ipaddress
import math
import re
from collections import Counter
from types import SimpleNamespace
from urllib.parse import urlsplit

FEATURE_NAMES = [
    "url_length", "host_length", "path_length", "query_length", "num_dots",
    "num_hyphens", "num_underscores", "num_at", "num_question", "num_equals",
    "num_ampersand", "num_percent", "num_slashes", "num_digits", "digit_ratio",
    "num_subdomains", "has_ip", "tld_suspicious", "is_shortener", "has_punycode",
    "has_https", "has_port", "keyword_count", "entropy", "path_depth",
]

TWO_LEVEL_SUFFIXES = {"co.uk", "org.uk", "ac.uk", "gov.uk", "co.in", "ac.in", "org.in", "gov.in", "net.in", "com.au", "net.au", "co.jp", "com.br", "co.za"}
SUSPICIOUS_TLDS = {"zip", "xyz", "top", "tk", "ml", "ga", "cf", "gq", "click", "work", "support"}
SHORTENERS = {"bit.ly", "tinyurl.com", "goo.gl", "t.co", "ow.ly", "is.gd", "buff.ly", "cutt.ly", "rb.gy", "shorturl.at"}
KEYWORDS = {"login", "signin", "verify", "secure", "account", "update", "bank", "confirm", "password", "free", "bonus", "wallet", "paypal", "invoice"}


def _parse(url):
    value = str(url or "")
    source = value if "://" in value else "http://" + value
    try:
        parsed = urlsplit(source)
    except ValueError:
        # urlsplit rejects malformed bracketed hosts. Preserve enough parsed
        # structure for feature extraction without rejecting the URL string.
        scheme_match = re.match(r"^([A-Za-z][A-Za-z0-9+.-]*)://", source)
        scheme = scheme_match.group(1) if scheme_match else ""
        remainder = source[scheme_match.end():] if scheme_match else source
        host_end = min((remainder.find(char) for char in "/?#" if char in remainder), default=len(remainder))
        host = remainder[:host_end]
        tail = remainder[host_end:]
        path_end = min((tail.find(char) for char in "?#" if char in tail), default=len(tail))
        path = tail[:path_end]
        query = ""
        if "?" in tail:
            query = tail.split("?", 1)[1].split("#", 1)[0]
        parsed = SimpleNamespace(scheme=scheme, hostname=host, path=path, query=query)
        return value, parsed, host.lower(), None
    try:
        port = parsed.port
    except ValueError:
        port = None
    return value, parsed, (parsed.hostname or "").lower(), port


def _subdomain_count(host):
    try:
        ipaddress.ip_address(host)
        return 0
    except ValueError:
        pass
    labels = [part for part in host.rstrip(".").split(".") if part]
    suffix_len = 2 if ".".join(labels[-2:]) in TWO_LEVEL_SUFFIXES else 1
    return max(0, len(labels) - suffix_len - 1)


def _canonicalize_url(value):
    """Remove common presentation-only URL differences before feature counts."""
    canonical = re.sub(r"^[A-Za-z][A-Za-z0-9+.-]*://", "", value, count=1)
    boundary = min((canonical.find(char) for char in "/?#" if char in canonical), default=len(canonical))
    authority, remainder = canonical[:boundary], canonical[boundary:]
    userinfo, separator, host_port = authority.rpartition("@")
    host_port = host_port.lower()
    if host_port.startswith("www."):
        host_port = host_port[4:]
    authority = (userinfo + separator if separator else "") + host_port
    canonical = authority + remainder
    if canonical.endswith("/"):
        canonical = canonical[:-1]
    return canonical


def extract_features(url):
    """Return 25 ordered numeric features, canonicalizing formatting artifacts."""
    original = str(url or "")
    original_parsed = _parse(original)[1]
    value = _canonicalize_url(original)
    _, parsed, host, port = _parse(value)
    lowered = value.lower()
    try:
        ipaddress.ip_address(host)
        has_ip = 1
    except ValueError:
        has_ip = 0
    labels = [part for part in host.rstrip(".").split(".") if part]
    tld = labels[-1] if labels else ""
    digits = sum(ch.isdigit() for ch in value)
    counts = Counter(value)
    entropy = -sum((n / len(value)) * math.log2(n / len(value)) for n in counts.values()) if value else 0.0
    result = {
        "url_length": len(value), "host_length": len(host), "path_length": len(parsed.path), "query_length": len(parsed.query),
        "num_dots": value.count("."), "num_hyphens": value.count("-"), "num_underscores": value.count("_"),
        "num_at": value.count("@"), "num_question": value.count("?"), "num_equals": value.count("="),
        "num_ampersand": value.count("&"), "num_percent": value.count("%"), "num_slashes": value.count("/"),
        "num_digits": digits, "digit_ratio": digits / len(value) if value else 0.0,
        "num_subdomains": _subdomain_count(host), "has_ip": has_ip, "tld_suspicious": int(tld in SUSPICIOUS_TLDS),
        "is_shortener": int(host in SHORTENERS), "has_punycode": int("xn--" in host),
        "has_https": int(original_parsed.scheme.lower() == "https"), "has_port": int(port is not None),
        "keyword_count": sum(bool(re.search(r"(?<![a-z])" + re.escape(word) + r"(?![a-z])", lowered)) for word in KEYWORDS),
        "entropy": entropy, "path_depth": sum(bool(part) for part in parsed.path.split("/")),
    }
    return {name: result[name] for name in FEATURE_NAMES}
