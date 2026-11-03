# AGENTS.md

## Secrets / API keys

- Do **not** store API keys in this repository, prompts, source code, logs, reports, or Git.
- `DEEPSEEK_API_KEY` for this Windows machine is stored **outside the repo** at:
  `C:\Users\36266\.secrets\deepseek.key`
- Runtime precedence:
  1. `DEEPSEEK_API_KEY` environment variable, if set;
  2. otherwise read the external key file above.
- Read the key only at runtime. Never print, echo, log, screenshot, or commit its contents.
- If neither source exists, stop and ask the user to provide it; do not use a mock as an online substitute.
- On a remote server or another machine, set `DEEPSEEK_API_KEY` in that environment or copy the key securely; the local Windows path is not portable.
- `.gitignore` excludes `.secrets/`, `secrets/`, `.env*` (except `.env.example`), `*.key`, and `**/deepseek.key` as a safety net. This does not make it safe to keep a real key inside the repo; keep it outside.
