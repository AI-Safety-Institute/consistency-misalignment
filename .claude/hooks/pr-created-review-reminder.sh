#!/usr/bin/env bash
# PostToolUse hook: after a PR is created (via `gh pr create` or a REST API
# POST to /pulls), emit a system reminder to run two review skills in
# parallel before requesting human review.
#
# Why: the standing convention in this repo is to fan-out review before
# inviting a human reviewer. The global `/review` skill flags generic
# correctness/style issues; the project-local `/arathi-review` skill applies
# repo-specific conventions (docstring discipline, test layout, naming, etc.)
# that have built up over earlier PRs. Both produce small, actionable lists
# that are cheaper to apply pre-review than to round-trip with a human.
#
# Triggered as a PostToolUse hook on Bash. Reads the tool-call JSON from
# stdin and only emits when the command looks like a PR creation.

set -uo pipefail

input_command=$(jq -r '.tool_input.command // empty' | tr -d '\n')

is_pr_create=0
echo "$input_command" | grep -qE 'gh pr create' && is_pr_create=1
echo "$input_command" | grep -qE 'POST.*/pulls( |$|\?|")' && is_pr_create=1
echo "$input_command" | grep -qE '/pulls( |$|\?|").*POST' && is_pr_create=1

[ "$is_pr_create" -eq 1 ] || exit 0

cat <<'EOF'
{
  "hookSpecificOutput": {
    "hookEventName": "PostToolUse",
    "additionalContext": "A pull request was just created. Standing rule for this repo: invoke /review (generic correctness + style) and /arathi-review (project-specific conventions) in parallel on the new PR. Report findings, then apply fixes before requesting human review."
  }
}
EOF
