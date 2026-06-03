#!/usr/bin/env python3
from __future__ import annotations

import json
import smtplib
import subprocess
from email.message import EmailMessage
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False))


def get_keychain_secret(account: str, service: str, label: str) -> str:
    result = subprocess.run(
        ["security", "find-generic-password", "-a", account, "-s", service, "-w"],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Unable to load {label} from Keychain: {detail}")
    return result.stdout.strip()


def send_via_smtp(signal_config: dict[str, Any], recipient: str, subject: str, body: str) -> None:
    sender = str(signal_config.get("smtp_sender") or "").strip()
    host = str(signal_config.get("smtp_host") or "").strip()
    port = int(signal_config.get("smtp_port") or 465)
    use_ssl = bool(signal_config.get("smtp_ssl", True))
    keychain_service = str(signal_config.get("smtp_keychain_service") or "").strip()
    if not sender or not host or not keychain_service:
        raise RuntimeError("Missing SMTP config: smtp_sender / smtp_host / smtp_keychain_service")
    password = get_keychain_secret(sender, keychain_service, "SMTP credential")

    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)

    if use_ssl:
        with smtplib.SMTP_SSL(host, port, timeout=30) as server:
            server.login(sender, password)
            server.send_message(message)
    else:
        with smtplib.SMTP(host, port, timeout=30) as server:
            server.starttls()
            server.login(sender, password)
            server.send_message(message)


def send_via_mailapp(signal_config: dict[str, Any], recipient: str, subject: str, body: str) -> None:
    sender = str(signal_config.get("mailapp_sender") or "").strip()
    script_lines = [
        "on run argv",
        '  if (count of argv) is less than 3 then error "Missing Mail arguments"',
        "  set theRecipient to item 1 of argv",
        "  set theSubject to item 2 of argv",
        "  set theContent to item 3 of argv",
        '  tell application "Mail"',
        "    set newMessage to make new outgoing message with properties {subject:theSubject, content:theContent & return & return, visible:false}",
        "    tell newMessage",
        "      make new to recipient at end of to recipients with properties {address:theRecipient}",
    ]
    if sender:
        script_lines.append(f'      set sender to "{sender}"')
    script_lines.extend(
        [
            "      send",
            "    end tell",
            "  end tell",
            "end run",
        ]
    )
    result = subprocess.run(
        ["osascript", "-l", "AppleScript", "-e", "\n".join(script_lines), recipient, subject, body],
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        raise RuntimeError(f"Mail.app send failed: {detail}")


def send_email(signal_config: dict[str, Any], recipient: str, subject: str, body: str) -> None:
    provider = str(signal_config.get("provider") or "smtp").strip().lower()
    if provider == "smtp":
        send_via_smtp(signal_config, recipient, subject, body)
        return
    if provider == "mailapp":
        send_via_mailapp(signal_config, recipient, subject, body)
        return
    raise RuntimeError(f"Unsupported email provider: {provider}")
