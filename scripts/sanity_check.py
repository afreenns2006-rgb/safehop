"""Compare model scores across common URL formatting variants."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.predict import predict_url

KNOWN_SITES = [
    "google.com", "github.com", "wikipedia.org", "python.org", "bbc.com",
    "amazon.com", "microsoft.com", "apple.com", "nytimes.com", "stackoverflow.com",
]
PHISHING_EXAMPLES = [
    "http://192.0.2.10/login",
    "http://verify-account.login.secure.account.example.xyz/signin/confirm-details-now",
    "http://bit.ly/suspicious-login",
    "http://paypal.com@evil.example/login",
    "http://xn--pple-43d.com/login",
]


def main():
    print("Well-known sites by URL form:")
    for domain in KNOWN_SITES:
        variants = (
            ("bare", domain),
            ("https", f"https://{domain}"),
            ("https-www", f"https://www.{domain}"),
            ("https-www-slash", f"https://www.{domain}/"),
        )
        for form, url in variants:
            score = predict_url(url)["score"]
            print(f"{domain:20} {form:16} {score:5.1f}%")
            if score >= 35:
                print(f"WARNING: {domain} ({form}) scores {score:.1f}% phishing risk")

    print("\nObvious phishing examples:")
    for url in PHISHING_EXAMPLES:
        result = predict_url(url)
        print(f"{url} -> {result['score']:.1f}% ({result['level']})")


if __name__ == "__main__":
    main()
