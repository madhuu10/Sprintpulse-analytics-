# Definition of Done

A story is only counted as "completed" toward a sprint's committed points
if all of the following hold:

1. Code is merged to the main branch.
2. Unit tests cover the new logic, and the team's overall test-coverage
   percentage for the sprint is reported alongside completion rate.
3. The change has passed code review (tracked as `code_review_pass_rate`
   in sprint metrics).
4. QA has verified the story in the staging environment with no open
   critical bugs (`bugs_found_qa`).
5. CI is green — repeated `ci_build_failures` on a story block it from
   counting as done until resolved.

Stories that miss any of these at sprint close are **not** counted as
completed, even if the code is technically merged — they roll into the
next sprint as `carryover_points_in`.
