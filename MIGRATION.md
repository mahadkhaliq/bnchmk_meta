# Project Migration Guide

Backup created 2026-08-06 for `/Users/mkfqm/malof_lab`.

Read this file first on the new computer. It explains what is in the archive,
how to restore it, and where the Codex and Claude Code chats live.

## Quick Start

```bash
# 1. Verify the archive before extracting it.
shasum -a 256 -c malof_lab_complete_2026-08-06.zip.sha256

# 2. Extract it wherever you want the project to live.
unzip malof_lab_complete_2026-08-06.zip -d ~/

# 3. Restore the environment and run the smoke test.
cd ~/malof_lab
bash migration/restore_on_new_mac.sh

# 4. Optional: also load the Codex and Claude chats into those tools.
bash migration/restore_on_new_mac.sh --with-chats
```

Step 3 creates `.venv`, installs `mlp_pipeline/requirements.txt`, and runs
`python -m lorentz.test_1x1`. Step 4 is separate because it writes to
`~/.codex` and `~/.claude`; it moves any existing directories aside first and
prompts before touching anything.

## What Is In The Archive

The ZIP extracts to a single `malof_lab/` directory containing:

| Path | Contents |
| --- | --- |
| `.git/` | Full Git history, branches, and the dirty working tree as it stood at backup time |
| `mlp_pipeline/` | Primary MLP data, models, training, evaluation, plots, checkpoints |
| `lorentz/` | Geometry-to-Lorentz MLP, differentiable finite-slab physics, ablation campaigns |
| `power_tx_data/` | v3 measured/simulated datasets (1x1, 2x2, 3x3) |
| `version_4_codes_all/` | Synthetic v4 generation code |
| `reproducing_benchmark/` | Benchmark reproduction material |
| `references/` | Source documents behind the physics integration |
| `mlp_pipeline 2/` | Older working copy, kept as-is |
| `migration/` | Backup and restore scripts, chat exporters |
| `migration_state/` | Codex and Claude state archives, chat transcripts, checksums |
| `AGENTS.md` | The project guide both Codex and Claude Code read automatically |
| `CLAUDE.md`, `CODEX.md` | Pointers to `AGENTS.md` for each tool |

Loose ZIPs (`README.zip`, `dataset_version_3.zip`, `rep_benchmark_code.zip`,
`vf_rel_ablation_stable_v3_results.zip`) are carried across untouched.

The previous backup, `malof_lab_complete_2026-08-03.zip`, is **not** nested
inside this one. This archive supersedes it.

## Restoring Agent Context

### The part that always works

`AGENTS.md` is the durable, documented way to give a new Codex or Claude Code
installation this project's conventions and scientific context. It travels
with the repository, and both tools pick it up automatically once you open the
project. `CLAUDE.md` points Claude Code at it; `CODEX.md` does the same for
Codex. Nothing needs to be restored for this to work.

The readable chat transcripts in `migration_state/codex_chat_transcripts/` and
`migration_state/claude_chat_transcripts/` are plain Markdown and are readable
immediately, with no restore step and no particular tool version.

See `migration_state/CHATS.md` for the index of all eight conversations.

### The best-effort part

Loading old chats back into the tools' own UIs means copying local state
directories. Neither vendor documents raw session-file copying as a supported
migration interface, so treat it as best effort — which is exactly why the
Markdown transcripts exist as the durable fallback.

1. Install Codex/ChatGPT and Claude Code on the new Mac and **sign in
   normally**. Credentials are deliberately excluded from this backup.
2. Check whether account-backed chats already sync down on their own.
3. Fully quit Codex, the ChatGPT app, Claude Code, and any IDE extensions.
4. Run `bash migration/restore_on_new_mac.sh --with-chats`.

Codex also supports `/import` for Claude Code setup and recent chats; its
documented limit is 50 chats from the last 30 days, so it will not cover the
older threads here.

If the new Mac uses a different username, Claude Code encodes project paths
differently and will create a new directory under `~/.claude/projects/`. Copy
the restored contents of `-Users-mkfqm-malof-lab/` into that new directory
after checking its name.

## What Was Deliberately Excluded

The state archives carry conversations and settings, not identity:

- Codex: `auth.json`, `installation_id`, `logs_2.sqlite`, caches, IPC files,
  temporary files, shell snapshots, and WAL/SHM sidecars.
- Claude Code: `~/.claude.json`, login/account files, IDE lock files,
  telemetry, and settings backups that can carry account metadata.
- The unrelated `-Users-mkfqm-qic-internship-qfdia-rl` Claude project.

Codex SQLite databases (`state_5`, `memories_1`, `goals_1`) were copied with
SQLite's online backup command and passed `PRAGMA integrity_check`.

## Security Notes

- These archives contain private conversation content, datasets, model
  outputs, and full Git history. Treat them as sensitive.
- Transfer by direct means — external drive, AirDrop, or an access-controlled
  share. Do not put them behind a public link.
- Sign in fresh on the new computer rather than copying credentials.

## Refreshing This Backup

From the repository root, on the source machine:

```bash
bash migration/backup_agent_state.sh          # re-snapshot ~/.codex and ~/.claude
python3 migration/export_codex_chats.py       # refresh Codex transcripts
python3 migration/export_claude_chats.py      # refresh Claude transcripts
bash migration/create_complete_backup.sh      # rebuild the full ZIP + checksum
```
