#!/usr/bin/env bash
# PostToolUse hook: after a `git push` in the consistency-em repo, emit a
# system reminder to audit the PR description against the current diff.
#
# Triggered as a PostToolUse hook on Bash. Reads the tool-call JSON from stdin
# and only emits on `git push` invocations from the consistency-em repo on a
# non-main branch. See memory: pr-description-drifts-after-many-commits.

input_command=$(jq -r '.tool_input.command // empty')
echo "$input_command" | grep -qE '^git push' || exit 0

url=$(git config --get remote.origin.url 2>/dev/null)
branch=$(git rev-parse --abbrev-ref HEAD 2>/dev/null)

if [[ -z "$url" || -z "$branch" ]]; then
  exit 0
fi
if [[ "$branch" == "main" || "$branch" == "HEAD" ]]; then
  exit 0
fi
if [[ "$url" != *consistency-em* && "$url" != *consistency-misalignment* ]]; then
  exit 0
fi

jq -n --arg b "$branch" '
  {
    hookSpecificOutput: {
      hookEventName: "PostToolUse",
      additionalContext: (
        "PR-description audit: you just pushed to \($b). Before finishing this turn, " +
        "compare the latest PR body against `git log main..HEAD` and the new commit " +
        "summaries — patch the PR body via the GitHub REST API if anything has drifted " +
        "(API / column names / placeholders, file lists, verification numbers, missing " +
        "scope-expansions). See feedback memory `pr-description-drifts-after-many-commits` " +
        "for the rationale."
      )
    }
  }
'
