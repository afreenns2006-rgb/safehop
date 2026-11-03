import pytest

from src.features import FEATURE_NAMES, extract_features


def test_feature_shape_and_order():
    result = extract_features("https://example.com")
    assert list(result) == FEATURE_NAMES
    assert len(result) == 25


def test_ip_host():
    assert extract_features("http://192.168.1.1/a")["has_ip"] == 1
    assert extract_features("http://192.168.1.1/a")["num_subdomains"] == 0


def test_missing_scheme_keeps_https_false():
    assert extract_features("example.com")["has_https"] == 0


def test_at_symbol():
    assert extract_features("https://user@example.com")["num_at"] == 1


def test_shortener():
    assert extract_features("https://bit.ly/abc")["is_shortener"] == 1


def test_punycode():
    assert extract_features("https://xn--bcher-kva.example")["has_punycode"] == 1


def test_co_uk_subdomains():
    assert extract_features("https://login.mail.shop.example.co.uk")["num_subdomains"] == 3


def test_malformed_port():
    assert extract_features("https://example.com:bad/path")["has_port"] == 0


def test_empty_string():
    result = extract_features("")
    assert len(result) == 25
    assert result["url_length"] == 0


@pytest.mark.parametrize("url", [
    "http://[bad",
    "http://a]b/login",
    "http://[::1",
    "",
    "https://example.com/a\x00 b?q=hello world",
])
def test_malformed_and_unusual_urls_always_return_all_features(url):
    result = extract_features(url)
    assert list(result) == FEATURE_NAMES
    assert len(result) == 25


def test_canonical_features_ignore_scheme_www_and_trailing_slash():
    variants = [
        "wikipedia.org",
        "https://wikipedia.org",
        "https://www.wikipedia.org",
        "https://www.wikipedia.org/",
    ]
    feature_rows = [extract_features(url) for url in variants]
    canonical_names = [name for name in FEATURE_NAMES if name != "has_https"]
    assert all(
        {name: row[name] for name in canonical_names}
        == {name: feature_rows[0][name] for name in canonical_names}
        for row in feature_rows[1:]
    )
    assert [row["has_https"] for row in feature_rows] == [0, 1, 1, 1]
