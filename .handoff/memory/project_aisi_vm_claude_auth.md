---
name: AISI VM Claude Code authentication
description: VM image now handles Claude auth + token rotation natively; claudeup no longer needed
type: project
originSessionId: 5d48bfc0-9134-4a7c-814f-10a9bf5fc5ea
---
As of ~2026-05-07, AISI dev-VM images authenticate Claude Code natively at the platform level. The user-facing implications:

- **`claudeup` is no longer needed.** Just run `claude`; auth is handled behind the scenes by the VM image.
- **Token rotation is automatic.** TTL is 50 minutes; Claude fetches a new token when the current one expires. No daily-refresh ritual required.
- **`aisitools` is what fetches the API key.** It's installed by the AISI VM image itself (cloud-init), so it's already on PATH on every fresh AISI dev-VM — `good_morning.py`'s bootstrap script does NOT need to install it. If you ever run Claude Code on a non-AISI-image machine, install with `uv tool install git+ssh://git@github.com/AI-Safety-Institute/aisi-inspect-tools` to avoid a ~2 s `uvx` fallback delay.
- **`agentup` provides `codexup` (and `claudeup`).** The bootstrap script's `uv tool install ... agentup` is load-bearing because we still call `codexup` directly. Don't read it as "kept for ad-hoc claudeup use" — claudeup is the obsolete sibling, codexup is the active one. (A platform-side codexup replacement is reportedly in flight but not landed yet.)
- **Custom `~/.claude.json` should be merged, not overwritten.** Setup scripts that touch this file need to read-modify-write rather than clobber.
- **Test-key override** is possible by overriding the default AWS secret (per the platform docs).

**Why:** the change was shipped in [aisi-research-platform#1034](https://github.com/AI-Safety-Institute/aisi-research-platform/pull/1034). It removes per-user-per-day key-refresh chores that scripts like `scripts/good_morning.py` previously had to do.

**How to apply:** when modifying VM-bootstrap or daily-maintenance code, don't add `claudeup` invocations. If touching `~/.claude.json`, merge. If on a fresh VM and Claude startup feels slow, install `aisitools`. Note that this is platform-team behaviour — if Claude Code stops authenticating despite this, the failure is more likely platform-side than script-side.
