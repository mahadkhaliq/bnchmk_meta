#!/usr/bin/env python3
"""Export visible user/assistant messages from Codex JSONL sessions."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


SKIP_PREFIXES = (
    "<environment_context>",
    "<recommended_plugins>",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--codex-home", type=Path, default=Path.home() / ".codex"
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("migration_state") / "codex_chat_transcripts",
    )
    return parser.parse_args()


def load_titles(index_path):
    titles = {}
    if not index_path.exists():
        return titles
    with index_path.open() as handle:
        for line in handle:
            record = json.loads(line)
            titles[record["id"]] = record.get("thread_name", record["id"])
    return titles


def content_text(content):
    parts = []
    for item in content or []:
        if item.get("type") in {"input_text", "output_text", "text"}:
            text = item.get("text", "").strip()
            if text:
                parts.append(text)
    return "\n\n".join(parts)


def export_session(path, title, output):
    session_id = None
    started_at = None
    messages = []
    with path.open() as handle:
        for line in handle:
            record = json.loads(line)
            if record.get("type") == "session_meta":
                payload = record.get("payload", {})
                session_id = payload.get("session_id") or payload.get("id")
                started_at = payload.get("timestamp")
                continue
            if record.get("type") != "response_item":
                continue
            payload = record.get("payload", {})
            if payload.get("type") != "message":
                continue
            role = payload.get("role")
            if role not in {"user", "assistant"}:
                continue
            text = content_text(payload.get("content"))
            if not text or text.startswith(SKIP_PREFIXES):
                continue
            messages.append((role, text))

    if not session_id:
        session_id = path.stem.rsplit("-", 5)[-1]
    safe_title = re.sub(r"[^A-Za-z0-9._-]+", "-", title).strip("-")
    destination = output / f"{safe_title}-{session_id}.md"
    lines = [
        f"# {title}",
        "",
        f"- Session ID: `{session_id}`",
        f"- Started: `{started_at or 'unknown'}`",
        f"- Raw source: `{path.name}`",
        "",
    ]
    for role, text in messages:
        lines.extend((f"## {role.title()}", "", text, ""))
    destination.write_text("\n".join(lines), encoding="utf-8")
    return destination, len(messages)


def main():
    args = parse_args()
    sessions = args.codex_home / "sessions"
    titles = load_titles(args.codex_home / "session_index.jsonl")
    args.output.mkdir(parents=True, exist_ok=True)
    results = []
    for path in sorted(sessions.rglob("*.jsonl")):
        session_id = path.stem.rsplit("-", 5)[-1]
        for known_id in titles:
            if known_id in path.name:
                session_id = known_id
                break
        title = titles.get(session_id, session_id)
        destination, count = export_session(path, title, args.output)
        results.append((destination, count))
    for destination, count in results:
        print(f"{destination}: {count} visible messages")


if __name__ == "__main__":
    main()
