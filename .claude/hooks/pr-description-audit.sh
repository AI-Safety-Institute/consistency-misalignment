#!/usr/bin/env bash
# PostToolUse hook: after a `git push` in this repo, emit a system reminder
# to audit the PR description against the current diff.
#
# Why: on a multi-commit PR the body goes stale within hours as scope-expanding
# commits land (architectural refactors alongside features, follow-on fixes
# after review feedback). API claims, file lists, verification numbers, and
# scope-expansions are the most common drift surfaces. A reviewer reading a
# stale body builds a wrong model of what the PR does. This hook fires after
# every push so the audit happens at the moment the new commits become
# visible to reviewers.
#
# Triggered as a PostToolUse hook on Bash. Reads the tool-call JSON from
# stdin and only emits on `git push` invocations from this repo's clone on a
# non-main branch.

set -uo pipefail

input_command=$(jq -r '.tool_input.command // empty')
echo "$input_command" | grep -qE '^git push' || exit 0

remote_url=$(git config --get remote.origin.url 2>/dev/null)
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [[ -z "$remote_url" || -z "$branch" ]]; then
  exit 0
fi
if [[ "$branch" == "main" || "$branch" == "HEAD" ]]; then
  exit 0
fi
if [[ "$remote_url" != *consistency-em* && "$remote_url" != *consistency-misalignment* ]]; then
  exit 0
fi

jq -n --arg branch "$branch" '
  {
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: (
        "PR-description audit: you just pushed to \($branch). Before finishing " +
        "this turn, compare the latest PR body against `git log main..HEAD` and " +
        "the new commit summaries. Patch the PR body via the GitHub REST API if " +
        "anything has drifted: API names (column names, placeholders, kwargs), " +
        "file lists (new files / relocations), verification numbers (test count, " +
        "smoke status), and missing scope-expansions (architectural refactors or " +
        "review-driven fixes that landed alongside the feature). Reviewers reading " +
        "a stale body build a wrong model of what the PR does."
      )
    }
  }
'
