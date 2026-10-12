# Code Review Standards

Every pull request requires at least one approval before merge.
Release-blocking changes (anything touching payments, auth, or data
migrations) require two approvals.

`code_review_pass_rate` in sprint metrics measures the share of PRs
approved without a "changes requested" round — not just whether something
eventually got merged. A low pass rate usually means more rework cycles,
which is one of the leading indicators the sprint-performance model
weighs: teams with `code_review_pass_rate` below ~80% tend to see more
`bugs_escaped_prod` and more `ci_build_failures` the same sprint, since
review quality and defect rate are tightly linked.

Reviewers are expected to respond within one business day. Repeated
blocked PRs due to slow review should be raised in retro, not silently
absorbed into the next sprint's carryover.
