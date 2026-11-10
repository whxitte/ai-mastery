# Scope guard and recon pipeline

This lesson builds the foundation everything else stands on: a scope file, a Scope Guard that decides what may be touched (with unit tests that prove it), a SQLite store, and a recon pipeline that runs subfinder, dnsx, httpx and katana with the program's rate limit and headers. No AI yet. The LLM modules in the next lesson only ever read what this pipeline stored, so getting this layer right is what makes the rest safe.

> [!GOALS]
> - Write a `scope.yaml` that captures a program's machine-enforceable rules
> - Build `scope.py`: host, wildcard, IP and CIDR matching with deny-by-default and out-of-scope-wins semantics
> - Prove it with pytest, including the parser tricks you already know from SSRF work
> - Create the SQLite schema the whole project uses
> - Run a scope-checked, rate-limited recon pipeline against Juice Shop

## The scope file

Start with the data. Here is the scope for the practice lab:

```yaml title="programs/juice-shop/scope.yaml"
program: juice-shop
platform: local-lab
in_scope:
  - localhost
out_of_scope: []
ports: [3000]                 # only these ports may be probed
excluded_paths:
  - /socket.io/               # websocket polling noise, not worth crawling
rate_limit: 10                # requests per second, applied to every tool
headers:
  X-Bug-Bounty: lab-sethu     # programs often require an identifying header
banned_tests:
  - dos
  - brute_force
```

And here is what a real program's scope looks like in the same format (using reserved documentation domains and the TEST-NET-3 range, so nothing here points at a real company):

```yaml title="programs/example-corp/scope.yaml"
program: example-corp
platform: hackerone
in_scope:
  - "*.example.com"           # subdomains only, NOT example.com itself
  - example.com               # so the apex is listed separately
  - api.example.net
  - 203.0.113.0/24
out_of_scope:
  - blog.example.com          # third-party hosted blog
  - "*.corp.example.com"      # internal SSO, explicitly excluded
  - 203.0.113.50
ports: [80, 443, 8443]
excluded_paths:
  - /logout                   # crawling it kills your session
rate_limit: 5
headers:
  X-Bug-Bounty: h1-yourhandle
banned_tests:
  - dos
  - brute_force
  - vuln_scan                 # "no automated scanners" on this program
```

The rules the Guard applies:

| Entry | Matches | Does not match |
|---|---|---|
| `example.com` | `example.com` | `www.example.com`, `evilexample.com` |
| `*.example.com` | `www.example.com`, `a.b.example.com` | `example.com` (the apex), `example.com.evil.net` |
| `203.0.113.0/24` | `203.0.113.7` | any hostname, even one that resolves into the range |
| `203.0.113.50` | that single IP | anything else |
| `https://app.example.com` | treated as `app.example.com` | entries with a path (`https://x/api`) are rejected |

> [!CONCEPT] Deny by default, out-of-scope wins
> **Deny by default** means anything not explicitly matched by an in-scope rule is refused, including things the code does not understand (a decimal IP like `2130706433`, a URL with a backslash). **Out-of-scope wins** means exclusions are checked first, so `*.example.com` in scope plus `admin.example.com` out of scope gives a clear answer for `admin.example.com`: no. These are the same rules as a good firewall policy: default drop, specific deny before broad allow.

Two choices here are deliberately conservative:

- **Wildcards do not include the apex.** Most programs write `*.example.com` and list `example.com` separately when they mean it. If a program's wording is ambiguous, ask them; don't let code guess wider.
- **A hostname is not in scope just because its IP is in an in-scope range.** Programs that list a CIDR usually mean "these IPs", not "every name that resolves here". The reverse check is enforced, though: a name that matches `*.example.com` but resolves into an out-of-scope range is denied. That catches the classic "subdomain CNAMEs to a third-party SaaS" case.

`banned_tests` uses a fixed vocabulary of test types. Every tool wrapper declares its type, and the Guard refuses banned ones. An unknown word in `banned_tests` (a typo like `dso`) makes the file fail to load, because a typo in a ban list should never silently allow something.

| Test type | Used by |
|---|---|
| `passive_recon` | subfinder |
| `dns_resolution` | dnsx |
| `http_probe` | httpx |
| `crawl` | katana and our own JS downloads |
| `vuln_scan` | nuclei |
| `content_discovery` | ffuf |
| `brute_force`, `dos` | nothing in this project; listed so programs can ban them explicitly |

## `scope.py`: the Scope Guard

The whole module is standard library except for YAML loading. Read it slowly; it is the most security-critical file in the project.

```python title="scope.py"
"""Scope Guard: the single authority on what this project may touch.

Three rules: deny by default, out-of-scope always wins, fail closed on anything odd.
"""
from __future__ import annotations

import ipaddress
import re
import sys
from dataclasses import dataclass, field
from typing import Iterable
from urllib.parse import unquote, urlsplit

# Every action our code can take is labelled with one of these test types.
TEST_TYPES = {
    "passive_recon",      # third-party sources only (subfinder)
    "dns_resolution",     # dnsx
    "http_probe",         # httpx
    "crawl",              # katana and our own JS downloads
    "vuln_scan",          # nuclei
    "content_discovery",  # ffuf with a wordlist
    "brute_force",        # credential guessing: we never implement it, but programs ban it
    "dos",                # load or stress testing: same
}

_LABEL = r"[a-z0-9_](?:[a-z0-9_-]{0,61}[a-z0-9_])?"
_HOST_RE = re.compile(rf"{_LABEL}(?:\.{_LABEL})*")
_BAD_URL_CHARS = re.compile(r"[\s\\\x00-\x1f\x7f]")


class ScopeError(ValueError):
    """The scope file itself is invalid. We refuse to run rather than guess."""


class OutOfScope(PermissionError):
    """Code tried to touch something the scope does not allow."""


@dataclass(frozen=True)
class Decision:
    allowed: bool
    reason: str


def _as_ip(text: str):
    try:
        return ipaddress.ip_address(text)
    except ValueError:
        return None


def normalize_host(host: str) -> str:
    """Lowercase, strip the trailing dot and IPv6 brackets, IDNA-encode. Raise ValueError if odd."""
    h = str(host).strip().lower().rstrip(".")
    if h.startswith("[") and h.endswith("]"):
        h = h[1:-1]
    ip = _as_ip(h)
    if ip is not None:
        return str(ip)
    try:
        h = h.encode("idna").decode("ascii")
    except UnicodeError as exc:
        raise ValueError(f"bad host {host!r}") from exc
    if not h or len(h) > 253 or not _HOST_RE.fullmatch(h):
        raise ValueError(f"bad host {host!r}")
    return h


@dataclass(frozen=True)
class Rule:
    kind: str   # "exact" | "wildcard" | "network"
    value: str  # host name, wildcard root, or network in CIDR form
    raw: str    # the line as written in scope.yaml, for log messages

    def matches(self, host: str, ip) -> bool:
        if self.kind == "network":
            net = ipaddress.ip_network(self.value)
            return ip is not None and ip.version == net.version and ip in net
        if ip is not None:
            return False                         # name rules never match IP literals
        if self.kind == "exact":
            return host == self.value
        return host.endswith("." + self.value)   # wildcard: subdomains only, not the apex


def parse_rule(entry: str) -> Rule:
    raw = str(entry).strip()
    text = raw.lower()
    if "://" in text:
        parts = urlsplit(text)
        if parts.path not in ("", "/") or parts.query:
            raise ScopeError(f"{raw!r}: path-level scope is not supported; "
                             "scope the host and use excluded_paths")
        text = parts.hostname or ""
    if "/" in text:
        try:
            net = ipaddress.ip_network(text, strict=False)
        except ValueError as exc:
            raise ScopeError(f"bad CIDR {raw!r}") from exc
        if net.prefixlen == 0:
            raise ScopeError(f"{raw!r} matches the whole internet")
        return Rule("network", str(net), raw)
    if text.startswith("*."):
        root = text[2:]
        if "*" in root:
            raise ScopeError(f"unsupported wildcard {raw!r}")
        try:
            root = normalize_host(root)
        except ValueError as exc:
            raise ScopeError(f"bad wildcard {raw!r}") from exc
        if "." not in root or _as_ip(root) is not None:
            raise ScopeError(f"wildcard {raw!r} is too broad")
        return Rule("wildcard", root, raw)
    if "*" in text:
        raise ScopeError(f"unsupported wildcard {raw!r}: only '*.domain' is allowed")
    try:
        host = normalize_host(text)
    except ValueError as exc:
        raise ScopeError(f"bad scope entry {raw!r}") from exc
    if _as_ip(host) is not None:
        return Rule("network", str(ipaddress.ip_network(host)), raw)
    return Rule("exact", host, raw)


@dataclass
class Scope:
    program: str
    in_scope: list[Rule]
    out_of_scope: list[Rule] = field(default_factory=list)
    ports: frozenset[int] | None = None
    excluded_paths: tuple[str, ...] = ()
    rate_limit: int = 5
    headers: dict[str, str] = field(default_factory=dict)
    banned_tests: frozenset[str] = frozenset()

    # ---------- loading ----------
    @classmethod
    def from_dict(cls, d: dict) -> "Scope":
        if not d.get("in_scope"):
            raise ScopeError("in_scope is empty: refusing to run")
        banned = frozenset(d.get("banned_tests") or [])
        if unknown := banned - TEST_TYPES:
            raise ScopeError(f"unknown test types in banned_tests: {sorted(unknown)}")
        rate = int(d.get("rate_limit", 5))
        if not 1 <= rate <= 100:
            raise ScopeError("rate_limit must be 1..100 requests per second")
        headers = {str(k).strip(): str(v).strip() for k, v in (d.get("headers") or {}).items()}
        for k, v in headers.items():
            if not k or any(c in k + v for c in "\r\n") or ":" in k:
                raise ScopeError(f"bad header {k!r}")
        ports = d.get("ports")
        return cls(
            program=str(d.get("program", "unnamed")),
            in_scope=[parse_rule(e) for e in d["in_scope"]],
            out_of_scope=[parse_rule(e) for e in d.get("out_of_scope") or []],
            ports=frozenset(int(p) for p in ports) if ports else None,
            excluded_paths=tuple(str(p).lower() for p in d.get("excluded_paths") or ()),
            rate_limit=rate,
            headers=headers,
            banned_tests=banned,
        )

    @classmethod
    def load(cls, path) -> "Scope":
        import yaml  # only needed when reading files
        with open(path, encoding="utf-8") as f:
            return cls.from_dict(yaml.safe_load(f) or {})

    # ---------- decisions ----------
    def check_host(self, host: str, resolved_ips: Iterable[str] = ()) -> Decision:
        try:
            h = normalize_host(host)
        except ValueError:
            return Decision(False, f"unparseable host {host!r}")
        ip = _as_ip(h)
        for rule in self.out_of_scope:                       # out-of-scope wins
            if rule.matches(h, ip):
                return Decision(False, f"out-of-scope rule {rule.raw!r}")
        for addr in resolved_ips:                            # e.g. CNAME to a third party
            rip = _as_ip(str(addr).strip())
            if rip is None:
                return Decision(False, f"unparseable resolved IP {addr!r}")
            for rule in self.out_of_scope:
                if rule.kind == "network" and rule.matches("", rip):
                    return Decision(False, f"resolves to {addr}, out-of-scope rule {rule.raw!r}")
        for rule in self.in_scope:
            if rule.matches(h, ip):
                return Decision(True, f"in-scope rule {rule.raw!r}")
        return Decision(False, "no in-scope rule matched (deny by default)")

    def check_url(self, url: str) -> Decision:
        if not isinstance(url, str) or _BAD_URL_CHARS.search(url):
            return Decision(False, "whitespace, backslash or control character in URL")
        try:
            parts = urlsplit(url)
            port = parts.port
        except ValueError:
            return Decision(False, "unparseable URL")
        if parts.scheme not in ("http", "https"):
            return Decision(False, f"scheme {parts.scheme!r} not allowed")
        if "@" in parts.netloc:
            return Decision(False, "userinfo in URL (classic parser-confusion trick)")
        if not parts.hostname:
            return Decision(False, "URL has no host")
        port = port or (443 if parts.scheme == "https" else 80)
        if self.ports is not None and port not in self.ports:
            return Decision(False, f"port {port} not in allowed ports {sorted(self.ports)}")
        path = unquote(parts.path or "/").lower()
        for prefix in self.excluded_paths:
            if path.startswith(prefix):
                return Decision(False, f"path excluded by {prefix!r}")
        return self.check_host(parts.hostname)

    # ---------- enforcement helpers ----------
    def require_url(self, url: str) -> str:
        d = self.check_url(url)
        if not d.allowed:
            raise OutOfScope(f"{url}: {d.reason}")
        return url

    def require_host(self, host: str, resolved_ips: Iterable[str] = ()) -> str:
        d = self.check_host(host, resolved_ips)
        if not d.allowed:
            raise OutOfScope(f"{host}: {d.reason}")
        return normalize_host(host)

    def require_test(self, test_type: str) -> None:
        if test_type not in TEST_TYPES:
            raise OutOfScope(f"unknown test type {test_type!r}")
        if test_type in self.banned_tests:
            raise OutOfScope(f"program rules ban {test_type!r}")

    def partition_urls(self, urls: Iterable[str]) -> tuple[list[str], list[tuple[str, str]]]:
        allowed, denied = [], []
        for u in urls:
            d = self.check_url(u)
            (allowed.append(u) if d.allowed else denied.append((u, d.reason)))
        return allowed, denied

    def partition_hosts(self, hosts: Iterable[str]) -> tuple[list[str], list[tuple[str, str]]]:
        allowed, denied = [], []
        for h in hosts:
            d = self.check_host(h)
            (allowed.append(normalize_host(h)) if d.allowed else denied.append((h, d.reason)))
        return allowed, denied

    def wildcard_roots(self) -> list[str]:
        return [r.value for r in self.in_scope if r.kind == "wildcard"]

    def seed_hosts(self) -> list[str]:
        """Exact names and single IPs from in_scope: safe starting points."""
        seeds = [r.value for r in self.in_scope if r.kind == "exact"]
        for r in self.in_scope:
            net = ipaddress.ip_network(r.value) if r.kind == "network" else None
            if net is not None and net.num_addresses == 1:
                seeds.append(str(net.network_address))
        return [s for s in seeds if self.check_host(s).allowed]


if __name__ == "__main__":
    if len(sys.argv) < 3:
        sys.exit("usage: python scope.py programs/<name>/scope.yaml URL_OR_HOST [...]")
    scope = Scope.load(sys.argv[1])
    for item in sys.argv[2:]:
        d = scope.check_url(item) if "://" in item else scope.check_host(item)
        print(f"{'ALLOW' if d.allowed else 'DENY ':5}  {item}  ({d.reason})")
```

Some details worth calling out, because they are exactly the bugs you hunt for in other people's URL allowlists:

- **Userinfo**: in `http://example.com@evil.net/` the real host is `evil.net`. Python's parser gets that right, but we refuse any `@` in the authority anyway, because different parsers disagree about edge cases.
- **Backslashes**: `http://evil.net\@example.com/` is parsed as host `example.com` by Python's `urlsplit`, while browsers and many HTTP clients treat `\` as `/` and connect to `evil.net`. A parser differential like that is an SSRF classic; the Guard refuses backslashes, whitespace and control characters outright.
- **Odd IP encodings**: `2130706433` is `127.0.0.1` to many HTTP clients. Python's `ipaddress` does not accept it, so it is treated as a hostname, matches nothing, and is denied. Deny by default handles encodings you never thought of.
- **Excluded paths** are compared after percent-decoding and lowercasing, so `/%6Cogout` and `/LOGOUT` are still excluded.
- **Header values** are checked for CR/LF so a scope file cannot smuggle extra headers into tool arguments.

> [!WARNING] What the Guard can and cannot control
> The Guard controls what we **feed** to tools and what we **keep** from their output. Once katana is running, katana's own HTTP client decides which links it follows. That is why the crawl wrapper also passes katana's scope flags (`-fs`, `-cs`, `-cos`) and disables redirect following, and why we prefer tools with strict scope options. For an extra layer on real programs, route tools through Burp with a target scope and its option to drop out-of-scope requests enabled (the setting's location varies by Burp version), so the network path enforces scope too.

Try the CLI:

```bash
python scope.py programs/juice-shop/scope.yaml http://localhost:3000/ http://localhost:8080/ http://127.0.0.1:3000/ "http://localhost:3000/socket.io/?EIO=4"
```

```text
ALLOW  http://localhost:3000/  (in-scope rule 'localhost')
DENY   http://localhost:8080/  (port 8080 not in allowed ports [3000])
DENY   http://127.0.0.1:3000/  (no in-scope rule matched (deny by default))
DENY   http://localhost:3000/socket.io/?EIO=4  (path excluded by '/socket.io/')
```

`127.0.0.1` is denied even though it is the same machine: the scope says `localhost`, and the Guard does not reason about equivalence. That is the behaviour you want on a real program.

## Prove it with tests

A scope checker without tests is a liability. These run in milliseconds, so run them before every hunt.

```ini title="pytest.ini"
[pytest]
pythonpath = .
markers =
    llm: needs LM Studio running with the model loaded
```

```python title="tests/test_scope.py"
import pytest

from scope import OutOfScope, Scope, ScopeError

PROGRAM = {
    "program": "example-corp",
    "in_scope": ["*.example.com", "example.com", "api.example.net",
                 "203.0.113.0/24", "2001:db8::/32"],
    "out_of_scope": ["blog.example.com", "*.corp.example.com", "203.0.113.50"],
    "ports": [80, 443, 8443],
    "excluded_paths": ["/logout"],
    "rate_limit": 5,
    "headers": {"X-Bug-Bounty": "h1-yourhandle"},
    "banned_tests": ["dos", "brute_force"],
}


@pytest.fixture
def scope() -> Scope:
    return Scope.from_dict(PROGRAM)


@pytest.mark.parametrize("host,allowed", [
    ("www.example.com", True),
    ("a.b.example.com", True),
    ("example.com", True),             # listed separately
    ("EXAMPLE.COM.", True),            # case and trailing dot normalized
    ("evilexample.com", False),        # suffix trick
    ("example.com.evil.net", False),   # prefix trick
    ("blog.example.com", False),       # out-of-scope wins over the wildcard
    ("x.corp.example.com", False),     # out-of-scope wildcard
    ("api.example.net", True),
    ("www.example.net", False),        # deny by default
    ("203.0.113.7", True),             # CIDR
    ("203.0.113.50", False),           # excluded single IP inside the CIDR
    ("198.51.100.1", False),
    ("2001:db8::1", True),
    ("[2001:db8::1]", True),
    ("localhost", False),
    ("", False),
    ("exa mple.com", False),
])
def test_hosts(scope, host, allowed):
    assert scope.check_host(host).allowed is allowed


@pytest.mark.parametrize("url,allowed", [
    ("https://www.example.com/", True),
    ("https://www.example.com:8443/x", True),
    ("https://www.example.com:8080/", False),        # port not allowed
    ("http://example.com@evil.net/", False),        # userinfo
    ("http://evil.net\\@example.com/", False),      # backslash parser differential
    ("ftp://example.com/", False),
    ("javascript:alert(1)", False),
    ("https://www.example.com/LOGOUT", False),      # excluded path, case-insensitive
    ("https://www.example.com/%6cogout", False),    # excluded path, percent-encoded
    ("http://2130706433/", False),                  # decimal IP
    ("https://[2001:db8::1]/", True),
    ("https://www.example.com/a b", False),         # whitespace
    ("https://www.example.com:99999/", False),      # invalid port
])
def test_urls(scope, url, allowed):
    assert scope.check_url(url).allowed is allowed


def test_resolved_ip_out_of_scope(scope):
    assert not scope.check_host("cdn.example.com", ["203.0.113.50"]).allowed
    assert scope.check_host("cdn.example.com", ["198.51.100.9"]).allowed


@pytest.mark.parametrize("bad", [
    {"in_scope": []},
    {"in_scope": ["*example.com"]},
    {"in_scope": ["*.com"]},
    {"in_scope": ["0.0.0.0/0"]},
    {"in_scope": ["https://example.com/api"]},
    {"in_scope": ["a.example"], "banned_tests": ["dso"]},
    {"in_scope": ["a.example"], "headers": {"X-Test": "a\r\nInjected: 1"}},
])
def test_bad_scope_files_fail_closed(bad):
    with pytest.raises(ScopeError):
        Scope.from_dict(bad)


def test_banned_tests(scope):
    with pytest.raises(OutOfScope):
        scope.require_test("dos")
    scope.require_test("crawl")  # allowed, no exception


def test_require_url_raises(scope):
    with pytest.raises(OutOfScope):
        scope.require_url("https://blog.example.com/")


def test_lab_scope():
    lab = Scope.from_dict({"program": "juice-shop", "in_scope": ["localhost"], "ports": [3000]})
    assert lab.seed_hosts() == ["localhost"]
    assert lab.check_url("http://localhost:3000/rest/products/search?q=x").allowed
    assert not lab.check_url("http://localhost:8080/").allowed
```

```bash
pytest -q tests/test_scope.py
```

All tests should pass. When you add a feature to `scope.py`, add the attack you are worried about to these lists first.

## `db.py`: the SQLite store

One database per program at `data/<program>/hunt.db`. SQLite needs no server, is a single file you can copy as a snapshot, and you can poke at it with the `sqlite3` command that ships with macOS.

```python title="db.py"
"""SQLite store: one database per program under data/<program>/hunt.db."""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from config import DATA_DIR

SCHEMA = """
CREATE TABLE IF NOT EXISTS hosts (
    id INTEGER PRIMARY KEY,
    name TEXT UNIQUE NOT NULL,          -- normalized host name or IP
    source TEXT NOT NULL,               -- scope | subfinder
    ips TEXT DEFAULT '[]',              -- JSON list from dnsx
    resolved_at TEXT,
    probed_at TEXT,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS urls (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    host TEXT NOT NULL,
    method TEXT DEFAULT 'GET',
    source TEXT NOT NULL,               -- httpx | katana | js | ffuf
    is_service INTEGER DEFAULT 0,       -- 1 = root of a live web service (from httpx)
    status INTEGER,
    title TEXT,
    content_type TEXT,
    length INTEGER,
    webserver TEXT,
    location TEXT,
    tech TEXT DEFAULT '[]',             -- JSON list
    params TEXT DEFAULT '[]',           -- JSON list of parameter names
    sensitive INTEGER DEFAULT 0,        -- set by the JS analyzer
    notes TEXT,                         -- short LLM explanation
    crawled_at TEXT,                    -- services only
    score INTEGER,                      -- host triage, services only
    score_json TEXT,
    first_seen TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS js_files (
    id INTEGER PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    sha256 TEXT,
    size INTEGER,
    path TEXT,                          -- local copy under data/<program>/js/
    fetched_at TEXT,
    analyzed_at TEXT
);
CREATE TABLE IF NOT EXISTS findings (
    id INTEGER PRIMARY KEY,
    source TEXT NOT NULL,               -- nuclei | js-secret | ffuf | manual
    dedupe_key TEXT UNIQUE NOT NULL,
    name TEXT,
    severity TEXT,
    url TEXT,
    detail TEXT,                        -- JSON: trimmed raw record
    hits INTEGER DEFAULT 1,
    triage_json TEXT,                   -- LLM triage output
    status TEXT DEFAULT 'new',          -- new | triaged | verified | rejected
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS hypotheses (
    id INTEGER PRIMARY KEY,
    url_id INTEGER REFERENCES urls(id),
    param TEXT DEFAULT '',
    wstg_id TEXT NOT NULL,
    title TEXT NOT NULL,
    rationale TEXT,
    evidence TEXT,
    test_plan TEXT,                     -- JSON list of manual steps
    test_risk TEXT,
    priority INTEGER,
    status TEXT DEFAULT 'open',         -- open | confirmed | rejected | skipped
    prompt_version TEXT,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(url_id, param, wstg_id)
);
CREATE TABLE IF NOT EXISTS param_classes (
    name TEXT PRIMARY KEY,
    class TEXT,
    confidence TEXT,
    source TEXT                         -- rule | llm
);
CREATE TABLE IF NOT EXISTS llm_runs (
    id INTEGER PRIMARY KEY,
    module TEXT NOT NULL,
    prompt_version TEXT,
    model TEXT,
    input_sha256 TEXT,
    output TEXT,
    valid INTEGER,
    error TEXT,
    attempts INTEGER,
    latency_ms INTEGER,
    prompt_tokens INTEGER,
    completion_tokens INTEGER,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
"""

URL_FIELDS = {"method", "is_service", "status", "title", "content_type", "length",
              "webserver", "location", "tech", "params", "sensitive", "notes"}


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def program_dir(program: str) -> Path:
    d = DATA_DIR / program
    d.mkdir(parents=True, exist_ok=True)
    return d


def connect(program: str) -> sqlite3.Connection:
    conn = sqlite3.connect(program_dir(program) / "hunt.db")
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.executescript(SCHEMA)
    return conn


def upsert_host(conn: sqlite3.Connection, name: str, source: str) -> None:
    conn.execute("INSERT OR IGNORE INTO hosts(name, source) VALUES (?, ?)", (name, source))


def set_host_ips(conn: sqlite3.Connection, name: str, ips: list[str]) -> None:
    conn.execute("UPDATE hosts SET ips=?, resolved_at=? WHERE name=?",
                 (json.dumps(sorted(set(ips))), now(), name))


def upsert_url(conn: sqlite3.Connection, url: str, host: str, source: str, **fields) -> int:
    """Insert a URL if new, then update whichever known fields were given."""
    conn.execute("INSERT OR IGNORE INTO urls(url, host, source) VALUES (?, ?, ?)",
                 (url, host, source))
    sets = {k: json.dumps(v) if isinstance(v, (list, dict)) else v
            for k, v in fields.items() if k in URL_FIELDS and v is not None}
    if sets:  # column names come from the URL_FIELDS allowlist, never from input
        cols = ", ".join(f"{k}=?" for k in sets)
        conn.execute(f"UPDATE urls SET {cols} WHERE url=?", (*sets.values(), url))
    return conn.execute("SELECT id FROM urls WHERE url=?", (url,)).fetchone()[0]


def add_finding(conn: sqlite3.Connection, *, source: str, dedupe_key: str, name: str,
                severity: str, url: str, detail: dict) -> None:
    """Insert a finding, or bump its hit counter if we have seen the same one before."""
    conn.execute(
        """INSERT INTO findings(source, dedupe_key, name, severity, url, detail)
           VALUES (?, ?, ?, ?, ?, ?)
           ON CONFLICT(dedupe_key) DO UPDATE SET hits = hits + 1""",
        (source, dedupe_key, name, severity, url, json.dumps(detail)[:20000]))


def counts(conn: sqlite3.Connection) -> dict[str, int]:
    """Numbers only: this is what the planner is allowed to see."""
    q = lambda sql: conn.execute(sql).fetchone()[0]
    return {
        "hosts": q("SELECT COUNT(*) FROM hosts"),
        "hosts_unprobed": q("SELECT COUNT(*) FROM hosts WHERE probed_at IS NULL"),
        "services": q("SELECT COUNT(*) FROM urls WHERE is_service=1"),
        "services_uncrawled": q("SELECT COUNT(*) FROM urls WHERE is_service=1 AND crawled_at IS NULL"),
        "services_unscored": q("SELECT COUNT(*) FROM urls WHERE is_service=1 AND score IS NULL"),
        "urls": q("SELECT COUNT(*) FROM urls"),
        "js_files": q("SELECT COUNT(*) FROM js_files"),
        "js_unfetched": q("SELECT COUNT(*) FROM js_files WHERE fetched_at IS NULL"),
        "js_unanalyzed": q("SELECT COUNT(*) FROM js_files WHERE path IS NOT NULL AND analyzed_at IS NULL"),
        "findings": q("SELECT COUNT(*) FROM findings"),
        "findings_untriaged": q("SELECT COUNT(*) FROM findings WHERE triage_json IS NULL"),
        "hypotheses": q("SELECT COUNT(*) FROM hypotheses"),
    }
```

> [!TIP] Why `counts()` lives here
> The planner in the [agent loop lesson](#/bugbounty/agent-loop) decides based on these numbers alone. Keeping the function next to the schema makes it obvious that it returns integers and nothing a target could write.

## `net.py`: the only HTTP client our code uses

The recon tools make their own requests, but our Python code also needs to fetch things (JS files to analyze). All of that goes through one client that checks scope on **every hop**, including redirects, applies the program rate limit and headers, and caps response size.

```python title="net.py"
"""Scope-checked, rate-limited, size-capped HTTP client for our own requests."""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from urllib.parse import urljoin

import requests

from scope import OutOfScope, Scope


class RateLimiter:
    """Spaces requests evenly: at most `per_second` per second across threads."""

    def __init__(self, per_second: int):
        self.interval = 1.0 / per_second
        self._next = 0.0
        self._lock = threading.Lock()

    def wait(self) -> None:
        with self._lock:
            now = time.monotonic()
            delay = self._next - now
            self._next = max(now, self._next) + self.interval
        if delay > 0:
            time.sleep(delay)


@dataclass
class Fetched:
    url: str
    status: int
    headers: dict
    body: bytes


class ScopedClient:
    def __init__(self, scope: Scope, test_type: str = "crawl", max_bytes: int = 5_000_000):
        scope.require_test(test_type)
        self.scope = scope
        self.max_bytes = max_bytes
        self.limiter = RateLimiter(scope.rate_limit)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "bb-lab/0.1", **scope.headers})

    def get(self, url: str, max_redirects: int = 3) -> Fetched:
        for _ in range(max_redirects + 1):
            self.scope.require_url(url)                    # checked on every hop
            self.limiter.wait()
            r = self.session.get(url, timeout=15, allow_redirects=False, stream=True)
            if r.is_redirect:
                nxt = urljoin(url, r.headers["location"])
                r.close()
                url = nxt                                  # loop re-checks scope
                continue
            body = b""
            for chunk in r.iter_content(65536):
                body += chunk
                if len(body) > self.max_bytes:
                    r.close()
                    raise ValueError(f"{url}: response larger than {self.max_bytes} bytes")
            return Fetched(url, r.status_code, dict(r.headers), body)
        raise OutOfScope(f"too many redirects from {url}")
```

> [!NOTE] Why handle redirects by hand
> With `allow_redirects=True`, `requests` would follow a `302` to anywhere. An open redirect on the target (Juice Shop has one, as do many real apps) would then bounce your downloader out of scope. Following redirects manually lets the Guard check each `Location`.

## `recon.py`: the pipeline

Each stage reads its input from the database, filters it through the Guard, writes targets to a temporary file, runs the tool with an argument list, and re-checks every output line before storing it. Raw tool output is kept under `data/<program>/raw/` for audit, and every denial is appended to `data/<program>/denied.jsonl`, so you can always answer "why didn't it test X?"

```python title="recon.py"
"""Recon pipeline: seed -> subfinder -> dnsx -> httpx -> katana -> JS download."""
from __future__ import annotations

import argparse
import hashlib
import ipaddress
import json
import os
import re
import tempfile
from contextlib import contextmanager
from urllib.parse import parse_qsl, urlsplit

import requests

import db
from net import ScopedClient
from scope import OutOfScope, Scope
from tools import run

LOCAL_NAMES = {"localhost"}   # never sent to public DNS


# ---------- shared helpers (active.py reuses these) ----------
@contextmanager
def targets_file(lines):
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(lines) + "\n")
    try:
        yield f.name
    finally:
        os.unlink(f.name)


def header_args(scope: Scope) -> list[str]:
    args: list[str] = []
    for k, v in scope.headers.items():
        args += ["-H", f"{k}: {v}"]
    return args


def jsonl(text: str):
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("{"):
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def save_raw(scope: Scope, stage: str, text: str) -> None:
    folder = db.program_dir(scope.program) / "raw"
    folder.mkdir(exist_ok=True)
    (folder / f"{stage}-{db.now().replace(':', '')}.jsonl").write_text(text)


def log_denied(scope: Scope, stage: str, denied: list[tuple[str, str]]) -> None:
    if not denied:
        return
    with open(db.program_dir(scope.program) / "denied.jsonl", "a") as f:
        for item, reason in denied:
            f.write(json.dumps({"ts": db.now(), "stage": stage, "item": item, "reason": reason}) + "\n")


def _is_ip(text: str) -> bool:
    try:
        ipaddress.ip_address(text)
        return True
    except ValueError:
        return False


def _root(url: str) -> str:
    p = urlsplit(url)
    return f"{p.scheme}://{p.netloc}/"


def _scope_regex(scope: Scope) -> str:
    """katana -cs: in-scope URL regex built from exact and wildcard rules."""
    alts = []
    for r in scope.in_scope:
        if r.kind == "exact":
            alts.append(re.escape(r.value))
        elif r.kind == "wildcard":
            alts.append(r"(?:[a-z0-9_-]+\.)+" + re.escape(r.value))
        else:
            net = ipaddress.ip_network(r.value)
            if net.num_addresses == 1:
                addr = str(net.network_address)
                alts.append(re.escape(addr) if net.version == 4 else r"\[" + re.escape(addr) + r"\]")
    return rf"^https?://(?:{'|'.join(alts)})(?::\d+)?(?:[/?#]|$)"


def _exclude_regex(scope: Scope) -> str:
    """katana -cos: excluded paths on any host."""
    paths = "|".join(re.escape(p) for p in scope.excluded_paths)
    return rf"(?i)^https?://[^/?#]+(?:{paths})"


# ---------- stages ----------
def stage_seed(scope: Scope, conn) -> str:
    seeds = scope.seed_hosts()
    for h in seeds:
        db.upsert_host(conn, h, "scope")
    conn.commit()
    return f"seeded {len(seeds)} hosts from scope"


def stage_subdomains(scope: Scope, conn) -> str:
    roots = scope.wildcard_roots()
    if not roots:
        return "no wildcard roots in scope; subfinder skipped"
    scope.require_test("passive_recon")
    with targets_file(roots) as path:
        p = run(["subfinder", "-dL", path, "-oJ", "-silent", "-duc"], timeout=1800)
    save_raw(scope, "subfinder", p.stdout)
    names = {rec.get("host", "") for rec in jsonl(p.stdout)}
    allowed, denied = scope.partition_hosts(n for n in names if n)
    for h in allowed:
        db.upsert_host(conn, h, "subfinder")
    conn.commit()
    log_denied(scope, "subfinder", denied)
    return f"subfinder: {len(allowed)} hosts in scope, {len(denied)} denied"


def stage_resolve(scope: Scope, conn) -> str:
    names = [r["name"] for r in conn.execute("SELECT name FROM hosts WHERE resolved_at IS NULL")]
    literal = [n for n in names if _is_ip(n) or n in LOCAL_NAMES]
    for n in literal:
        db.set_host_ips(conn, n, [n] if _is_ip(n) else ["127.0.0.1"])
    to_resolve = [n for n in names if n not in literal]
    denied: list[tuple[str, str]] = []
    if to_resolve:
        scope.require_test("dns_resolution")
        # DNS goes to public resolvers, not the target; 50/s is polite to them
        with targets_file(to_resolve) as path:
            p = run(["dnsx", "-l", path, "-a", "-aaaa", "-resp", "-json", "-silent",
                     "-duc", "-rl", "50"], timeout=1800)
        save_raw(scope, "dnsx", p.stdout)
        resolved = {rec.get("host", ""): (rec.get("a") or []) + (rec.get("aaaa") or [])
                    for rec in jsonl(p.stdout)}
        for n in to_resolve:
            ips = resolved.get(n, [])
            d = scope.check_host(n, ips)
            if d.allowed:
                db.set_host_ips(conn, n, ips)
            else:
                conn.execute("DELETE FROM hosts WHERE name=?", (n,))
                denied.append((n, d.reason))
    conn.commit()
    log_denied(scope, "dnsx", denied)
    return f"resolved {len(names)} hosts, dropped {len(denied)} resolving out of scope"


def stage_probe(scope: Scope, conn) -> str:
    scope.require_test("http_probe")
    rows = conn.execute("SELECT name, ips FROM hosts WHERE probed_at IS NULL").fetchall()
    targets = [r["name"] for r in rows if scope.check_host(r["name"], json.loads(r["ips"])).allowed]
    if not targets:
        return "nothing to probe"
    ports = ",".join(str(p) for p in sorted(scope.ports)) if scope.ports else "80,443"
    with targets_file(targets) as path:
        argv = ["httpx", "-l", path, "-p", ports,
                "-sc", "-title", "-td", "-server", "-ip", "-cl", "-ct", "-location",
                "-json", "-silent", "-duc", "-rl", str(scope.rate_limit), "-t", "10",
                *header_args(scope)]
        p = run(argv, timeout=3600)
    save_raw(scope, "httpx", p.stdout)
    live, denied = 0, []
    for rec in jsonl(p.stdout):
        url = rec.get("url", "")
        d = scope.check_url(url)
        if not d.allowed:
            denied.append((url, d.reason))
            continue
        db.upsert_url(conn, _root(url), urlsplit(url).hostname, "httpx", is_service=1,
                      status=rec.get("status_code"), title=rec.get("title"),
                      content_type=rec.get("content_type"), length=rec.get("content_length"),
                      webserver=rec.get("webserver"), location=rec.get("location"),
                      tech=rec.get("tech") or [])
        live += 1
    for t in targets:
        conn.execute("UPDATE hosts SET probed_at=? WHERE name=?", (db.now(), t))
    conn.commit()
    log_denied(scope, "httpx", denied)
    return f"httpx: {live} live services from {len(targets)} hosts"


def stage_crawl(scope: Scope, conn, depth: int = 3, duration: str = "10m") -> str:
    scope.require_test("crawl")
    rows = conn.execute("SELECT url FROM urls WHERE is_service=1 AND crawled_at IS NULL").fetchall()
    seeds = [r["url"] for r in rows if scope.check_url(r["url"]).allowed]
    if not seeds:
        return "nothing to crawl"
    with targets_file(seeds) as path:
        argv = ["katana", "-list", path, "-d", str(depth), "-jc",
                "-fs", "fqdn", "-cs", _scope_regex(scope), "-dr",
                "-ct", duration, "-rl", str(scope.rate_limit), "-c", "5", "-p", "2",
                "-jsonl", "-or", "-ob", "-silent", "-duc", *header_args(scope)]
        if depth >= 3:
            argv += ["-kf", "robotstxt,sitemapxml"]   # needs depth >= 3
        if scope.excluded_paths:
            argv += ["-cos", _exclude_regex(scope)]
        p = run(argv, timeout=3600)
    save_raw(scope, "katana", p.stdout)
    kept, js, denied = 0, 0, []
    for rec in jsonl(p.stdout):
        req, resp = rec.get("request") or {}, rec.get("response") or {}
        url = req.get("endpoint", "")
        d = scope.check_url(url)
        if not d.allowed:
            denied.append((url, d.reason))
            continue
        parts = urlsplit(url)
        params = sorted({k for k, _ in parse_qsl(parts.query, keep_blank_values=True)})
        db.upsert_url(conn, url, parts.hostname, "katana", method=req.get("method", "GET"),
                      status=resp.get("status_code"), params=params)
        kept += 1
        if parts.path.lower().endswith(".js"):
            conn.execute("INSERT OR IGNORE INTO js_files(url) VALUES (?)", (url,))
            js += 1
    for s in seeds:
        conn.execute("UPDATE urls SET crawled_at=? WHERE url=?", (db.now(), s))
    conn.commit()
    log_denied(scope, "katana", denied)
    return f"katana: {kept} URLs in scope ({js} JS), {len(denied)} denied"


def stage_fetch_js(scope: Scope, conn, limit: int = 50) -> str:
    client = ScopedClient(scope, "crawl")
    folder = db.program_dir(scope.program) / "js"
    folder.mkdir(exist_ok=True)
    rows = conn.execute("SELECT id, url FROM js_files WHERE fetched_at IS NULL LIMIT ?",
                        (limit,)).fetchall()
    ok = 0
    for r in rows:
        try:
            res = client.get(r["url"])
        except (OutOfScope, requests.RequestException, ValueError):
            res = None
        if res is None or res.status != 200:
            conn.execute("UPDATE js_files SET fetched_at=? WHERE id=?", (db.now(), r["id"]))
            continue
        digest = hashlib.sha256(res.body).hexdigest()
        path = folder / f"{digest[:16]}.js"
        path.write_bytes(res.body)
        conn.execute("UPDATE js_files SET sha256=?, size=?, path=?, fetched_at=? WHERE id=?",
                     (digest, len(res.body), str(path), db.now(), r["id"]))
        ok += 1
    conn.commit()
    return f"downloaded {ok}/{len(rows)} JS files"


STAGES = {"seed": stage_seed, "subdomains": stage_subdomains, "resolve": stage_resolve,
          "probe": stage_probe, "crawl": stage_crawl, "js": stage_fetch_js}


def main() -> None:
    ap = argparse.ArgumentParser(description="Scope-checked recon pipeline")
    ap.add_argument("scope_file")
    ap.add_argument("--stages", default=",".join(STAGES))
    args = ap.parse_args()
    scope = Scope.load(args.scope_file)
    conn = db.connect(scope.program)
    for name in args.stages.split(","):
        print(f"[{name}] {STAGES[name](scope, conn)}")


if __name__ == "__main__":
    main()
```

How each tool is called, and why:

| Stage | Key flags | Why |
|---|---|---|
| subfinder | `-dL file -oJ -silent -duc` | Roots from a file, JSON lines out; only runs for `*.domain` rules |
| dnsx | `-l file -a -aaaa -resp -json -rl 50` | A and AAAA records so the Guard can check resolved IPs |
| httpx | `-p <ports> -sc -title -td -server -ip -cl -ct -location -json -rl N -H ...` | One JSON record per live service with what triage needs |
| katana | `-d 3 -jc -kf robotstxt,sitemapxml -fs fqdn -cs <regex> -cos <regex> -dr -ct 10m -rl N -jsonl -or -ob -H ...` | JS parsing, known files, katana-side scope, no redirect following, time cap, lean output |

> [!WARNING] Flags change between versions
> These flags were checked against the current READMEs, but ProjectDiscovery tools evolve fast. When you upgrade, run `httpx -h`, `katana -h` and so on, and rerun the practice below. If a flag disappears, the tool usually prints an error and the stage returns zero results, which the agent will notice as "nothing changed".

`localhost` and IP literals skip dnsx: public resolvers know nothing about your loopback name, and sending internal names to public DNS is a small information leak on real engagements too.

## Practice run against Juice Shop

> [!LAB] First recon run
> 1. Make sure Juice Shop is running (`docker ps` shows `juice-shop`) and your venv is active.
> 2. Create `programs/juice-shop/scope.yaml` from the top of this lesson and run the tests: `pytest -q`.
> 3. Check a few decisions by hand:
>    ```bash
>    python scope.py programs/juice-shop/scope.yaml http://localhost:3000/ http://127.0.0.1:3000/ http://localhost:3001/
>    ```
> 4. Run the pipeline:
>    ```bash
>    python recon.py programs/juice-shop/scope.yaml
>    ```
>    You should see one line per stage: one seeded host, subfinder skipped (no wildcard rules), one live service from httpx, a few dozen to a few hundred URLs from katana, and a handful of JS files downloaded. Exact numbers depend on the Juice Shop version.
> 5. Look at what you collected:
>    ```bash
>    sqlite3 -header -column data/juice-shop/hunt.db "SELECT url, status, title, tech FROM urls WHERE is_service=1"
>    sqlite3 data/juice-shop/hunt.db "SELECT url FROM urls WHERE url LIKE '%/rest/%' OR url LIKE '%/api/%' LIMIT 30"
>    sqlite3 data/juice-shop/hunt.db "SELECT url, params FROM urls WHERE params != '[]' LIMIT 20"
>    sqlite3 data/juice-shop/hunt.db "SELECT url, size FROM js_files"
>    ```
> 6. Read the audit trail: `cat data/juice-shop/denied.jsonl | jq -c '{stage, item, reason}' | head`. Every line is something a tool found and the Guard refused, with the reason.
> 7. Experiment: add `/rest/` to `excluded_paths`, delete `data/juice-shop/hunt.db`, rerun, and confirm no `/rest/` URLs are stored. Then remove it again.

> [!TIP] Snapshots for later
> Copy the database after a good run: `cp data/juice-shop/hunt.db data/juice-shop/recon-snapshot.db`. The [evaluation lesson](#/bugbounty/evaluate) reruns analysis against a frozen snapshot so prompt changes can be compared fairly, without re-scanning.

## Recap

- `scope.yaml` holds the machine-enforceable rules: in/out-of-scope assets, ports, excluded paths, rate limit, headers, banned test types.
- The Guard denies by default, checks exclusions first, refuses odd input (userinfo, backslashes, control characters, unknown test types) and fails closed on a bad scope file.
- Tests encode the attacks you worry about; run them before every hunt.
- Every stage filters input and output through the Guard, passes the program rate limit and headers, and logs denials with reasons.
- Our own HTTP requests go through `ScopedClient`, which re-checks scope on every redirect hop.
- Tools' own scope flags and `-dr` matter because the Guard cannot see requests a running tool makes by itself.

## Check yourself

1. The scope lists `*.shop.example.com` and nothing else. Is `shop.example.com` in scope under our Guard? Should you test it?
   <details><summary>Answer</summary>

   No: wildcards match subdomains only, not the apex. Whether the program meant to include it is a question for the program, not for your code. Ask, or leave it alone.

   </details>
2. Why does `check_url` reject any URL containing a backslash instead of trying to parse it correctly?
   <details><summary>Answer</summary>

   Because parsers disagree. Python's `urlsplit` reads `http://evil.net\@example.com/` as host `example.com`, while browsers and many HTTP clients treat `\` as `/` and connect to `evil.net`. If the Guard and the tool parse differently, the Guard approves one host and the tool contacts another. Refusing the ambiguous input removes the differential.

   </details>
3. A subdomain matches `*.example.com` but resolves to `203.0.113.50`, which is listed out of scope. What happens, and where?
   <details><summary>Answer</summary>

   In `stage_resolve`, dnsx returns the A record, `check_host(name, ips)` finds the IP in an out-of-scope rule and denies it. The host is deleted from the database and the denial is logged, so no later stage probes it.

   </details>
4. Why does the crawl stage pass `-dr` to katana?
   <details><summary>Answer</summary>

   To stop katana from following redirects. An open redirect on the target could send katana's HTTP client to an out-of-scope host, and the Guard only sees katana's output afterwards. The redirecting URL itself is still recorded, which is useful as a lead.

   </details>
5. Why is `banned_tests: [dso]` an error rather than a warning?
   <details><summary>Answer</summary>

   A typo in a ban list would otherwise silently allow the thing you meant to ban (`dos`). Failing to load forces you to fix the file before any traffic is sent.

   </details>

## Go deeper

- [katana](https://github.com/projectdiscovery/katana) — scope options (`-cs`, `-cos`, `-fs`), JS parsing and JSONL fields
- [dnsx](https://github.com/projectdiscovery/dnsx) — resolution flags and JSON output
- [PortSwigger: SSRF](https://portswigger.net/web-security/ssrf) — URL parsing tricks that apply directly to allowlist code like the Scope Guard
- [Python `ipaddress` module](https://docs.python.org/3/library/ipaddress.html) — the exact parsing rules the Guard relies on
- [SQLite documentation](https://www.sqlite.org/docs.html) — `INSERT ... ON CONFLICT`, WAL mode and the `sqlite3` CLI
