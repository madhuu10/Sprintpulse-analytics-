"""
Synthetic data generator for SprintPulse.

No public dataset combines employee attendance with agile sprint performance
(that's internal company data at any real org), so this generates a
realistic synthetic one instead — clearly labeled as such everywhere it's
used. The generation isn't random noise: attendance behavior and sprint
metrics are built from the same underlying per-employee/per-team "quality"
signals, so the relationships a model or a chatbot finds in it (e.g. "lower
attendance teams tend to miss sprint commitments more") are real patterns
in the data, not fabricated after the fact.

Produces:
  data/employees.csv
  data/attendance_monthly.csv      one row per employee per month
  data/sprints.csv                 one row per team per sprint
  docs/retros/*.md                 one retro note per sprint, generated
                                    from that sprint's own numbers
"""

import numpy as np
import pandas as pd
from datetime import date, timedelta

rng = np.random.default_rng(42)

TEAMS = ["Phoenix", "Falcon", "Orion", "Nimbus", "Vertex", "Atlas", "Comet", "Lumen", "Zephyr", "Halcyon",
         "Aurora", "Griffin", "Juniper", "Kestrel", "Meridian", "Solstice"]
DEPARTMENTS = ["Platform", "Product", "Growth", "Data", "Infra"]
TEAM_DEPT = {team: DEPARTMENTS[i % len(DEPARTMENTS)] for i, team in enumerate(TEAMS)}
ROLES = ["Engineer", "QA", "Designer", "Product Manager", "DevOps"]
ROLE_P = [0.50, 0.15, 0.10, 0.10, 0.15]
SENIORITY = ["Junior", "Mid", "Senior"]
SENIORITY_P = [0.35, 0.40, 0.25]

FIRST_NAMES = ["Aarav", "Maya", "Liam", "Zara", "Noah", "Priya", "Ethan", "Sofia", "Ravi", "Elena",
               "Omar", "Grace", "Kenji", "Nina", "Leo", "Amara", "Finn", "Ingrid", "Diego", "Hana",
               "Tariq", "Wei", "Sana", "Lucas", "Mei", "Jonas", "Fatima", "Erik", "Nadia", "Theo"]
LAST_NAMES = ["Shah", "Kowalski", "Nguyen", "Okafor", "Rossi", "Mueller", "Tanaka", "Silva", "Patel", "Novak",
              "Larsen", "Costa", "Haddad", "Kim", "Dubois", "Santos", "Varga", "Ibrahim", "Choi", "Fischer"]


def make_name(used):
    while True:
        n = f"{rng.choice(FIRST_NAMES)} {rng.choice(LAST_NAMES)}"
        if n not in used:
            used.add(n)
            return n


# ---------------------------------------------------------------------
# 1. EMPLOYEES — each gets a hidden "reliability" trait that drives both
#    their attendance behavior AND (aggregated to team level) contributes
#    to their team's sprint performance.
# ---------------------------------------------------------------------
def build_employees():
    used_names = set()
    rows = []
    emp_id = 1
    for team in TEAMS:
        team_size = rng.integers(6, 11)
        for _ in range(team_size):
            seniority = rng.choice(SENIORITY, p=SENIORITY_P)
            tenure = {"Junior": rng.integers(1, 24), "Mid": rng.integers(12, 60), "Senior": rng.integers(36, 120)}[seniority]
            reliability = np.clip(rng.normal(0.90 if seniority != "Junior" else 0.84, 0.07), 0.55, 0.99)
            rows.append(dict(
                employee_id=emp_id, name=make_name(used_names), team=team, department=TEAM_DEPT[team],
                role=rng.choice(ROLES, p=ROLE_P), seniority=seniority, tenure_months=int(tenure),
                _reliability=reliability,
            ))
            emp_id += 1
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 2. MONTHLY ATTENDANCE — one row per employee per month. Winter months
#    (flu season) and high overtime in the prior month both push risk up.
# ---------------------------------------------------------------------
MONTHS = pd.date_range("2025-01-01", "2026-09-01", freq="MS")


def build_attendance(employees):
    rows = []
    overtime_prev = {e: 0.0 for e in employees["employee_id"]}
    for month in MONTHS:
        season_effect = 0.04 if month.month in (12, 1, 2) else 0.0  # winter flu season
        for _, emp in employees.iterrows():
            working_days = 21
            burnout = 0.015 * (overtime_prev[emp.employee_id] / 20)  # last month's overtime -> this month's absences
            new_hire_effect = 0.015 if emp.tenure_months < 6 else 0.0
            unplanned_rate = np.clip(
                (1 - emp._reliability) * 0.35 + season_effect * 0.5 + burnout + new_hire_effect + rng.normal(0, 0.015),
                0, 0.5,
            )
            unplanned_absences = rng.binomial(working_days, unplanned_rate)
            sick_days = rng.binomial(working_days, np.clip((1 - emp._reliability) * 0.15 + season_effect * 0.5, 0, 0.3))
            approved_leave = rng.binomial(3, 0.15)
            wfh_days = rng.binomial(working_days, 0.2)
            late_arrivals = rng.poisson(max(0.3, (1 - emp._reliability) * 6))
            overtime_hours = max(0, rng.normal(8 if emp.seniority != "Junior" else 4, 5))
            overtime_prev[emp.employee_id] = overtime_hours
            checkin_delay = max(0, rng.normal((1 - emp._reliability) * 20, 5))

            days_present = working_days - unplanned_absences - sick_days - approved_leave
            attendance_rate = max(0, days_present) / working_days

            # Hidden risk score -> observed binary label (what the model learns to predict)
            latent_risk = (
                -2.6 + 9.0 * (0.9 - attendance_rate) + 5.0 * unplanned_rate + 0.05 * overtime_hours
                + 1.5 * new_hire_effect + rng.normal(0, 0.25)
            )
            prob_high_risk = 1 / (1 + np.exp(-latent_risk))
            high_risk = rng.random() < prob_high_risk

            rows.append(dict(
                employee_id=emp.employee_id, month=month.strftime("%Y-%m"), team=emp.team,
                department=emp.department, role=emp.role, seniority=emp.seniority,
                tenure_months=emp.tenure_months, attendance_rate=round(attendance_rate, 3),
                unplanned_absences=int(unplanned_absences), sick_days=int(sick_days),
                approved_leave_days=int(approved_leave), wfh_days=int(wfh_days),
                late_arrivals=int(late_arrivals), overtime_hours=round(overtime_hours, 1),
                avg_checkin_delay_min=round(checkin_delay, 1),
                high_absenteeism_risk=int(high_risk),
            ))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 3. SPRINTS — one row per team per sprint. Completion rate depends on
#    QA quality, scope discipline, and that team's average attendance
#    for the month the sprint falls in (the real cross-domain link).
# ---------------------------------------------------------------------
def build_sprints(employees, attendance):
    team_attendance_by_month = attendance.groupby(["team", "month"])["attendance_rate"].mean()
    team_size = employees.groupby("team").size()

    start = date(2025, 1, 6)
    rows = []
    for team in TEAMS:
        sprint_start = start
        for sprint_num in range(1, 41):
            sprint_end = sprint_start + timedelta(days=13)
            month_key = sprint_start.strftime("%Y-%m")
            avg_attendance = team_attendance_by_month.get((team, month_key), 0.9)

            tsize = int(team_size[team])
            velocity_base = tsize * rng.normal(4.5, 0.6)
            scope_change_pct = np.clip(rng.normal(0.06, 0.05), 0, 0.35)
            test_coverage = np.clip(rng.normal(72, 12), 30, 98)
            code_review_pass_rate = np.clip(rng.normal(0.88, 0.08), 0.4, 1.0)
            bugs_found_qa = max(0, int(rng.poisson(3 - test_coverage / 50)))
            bugs_escaped_prod = max(0, int(rng.poisson(max(0.1, 1.2 - code_review_pass_rate))))
            ci_build_failures = max(0, int(rng.poisson(2 - code_review_pass_rate)))
            carryover_points_in = max(0, int(rng.normal(3, 3)))
            standup_attendance_pct = np.clip(avg_attendance + rng.normal(0, 0.03), 0.5, 1.0)

            committed = max(10, int(velocity_base + carryover_points_in))
            # Non-linear effects a plain linear model can't represent: test coverage's
            # benefit saturates (an S-curve, not a straight line), scope creep past ~15%
            # triggers a disproportionate "tipping point" penalty, and CI failures plus
            # escaped bugs compound each other rather than adding independently.
            coverage_effect = 0.22 * np.tanh((test_coverage - 68) / 14)
            scope_cliff = -14.0 * max(0, scope_change_pct - 0.15) ** 2
            crisis_interaction = -0.05 * ci_build_failures * bugs_escaped_prod
            effect = (
                coverage_effect
                - 0.22 * scope_change_pct
                - 0.015 * bugs_found_qa
                - 0.03 * bugs_escaped_prod
                - 0.02 * ci_build_failures
                + scope_cliff
                + crisis_interaction
                + 0.7 * (avg_attendance - 0.9)
                + rng.normal(0, 0.05)
            )
            completion_rate = float(np.clip(0.95 + effect, 0.25, 1.15))
            completed = int(round(committed * completion_rate))

            rows.append(dict(
                team=team, department=TEAM_DEPT[team], sprint_number=sprint_num,
                sprint_start=sprint_start.isoformat(), sprint_end=sprint_end.isoformat(),
                team_size=tsize, avg_team_attendance_rate=round(avg_attendance, 3),
                standup_attendance_pct=round(standup_attendance_pct, 3),
                story_points_committed=committed, story_points_completed=completed,
                carryover_points_in=carryover_points_in, scope_change_pct=round(scope_change_pct, 3),
                test_coverage_pct=round(test_coverage, 1), code_review_pass_rate=round(code_review_pass_rate, 3),
                bugs_found_qa=bugs_found_qa, bugs_escaped_prod=bugs_escaped_prod,
                ci_build_failures=ci_build_failures, completion_rate=round(completion_rate, 3),
            ))
            sprint_start = sprint_end + timedelta(days=1)
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# 4. RETRO NOTES — one short markdown note per sprint, generated from
#    that sprint's own numbers so the RAG corpus stays consistent with
#    the dashboards (nothing in the text contradicts the data).
# ---------------------------------------------------------------------
ACTIONS_QA = ["add regression tests around the affected module", "require a second reviewer on release-blocking PRs",
              "schedule a QA pairing session before code-freeze"]
ACTIONS_SCOPE = ["push late requests to the next sprint by default", "tighten sprint-planning acceptance criteria",
                 "flag scope additions in standup for visibility"]
ACTIONS_ATTEND = ["check in 1:1 with affected team members", "review on-call/rotation load for burnout",
                   "revisit WFH flexibility for the team"]


def retro_note(row):
    rate_pct = row.completion_rate * 100
    lines = [f"# {row.team} — Sprint {row.sprint_number} Retro\n",
             f"**Dates**: {row.sprint_start} to {row.sprint_end}  ",
             f"**Result**: completed {row.story_points_completed}/{row.story_points_committed} story points "
             f"({rate_pct:.0f}%)  ",
             f"**Team attendance this period**: {row.avg_team_attendance_rate*100:.0f}%\n"]

    issues = []
    if row.scope_change_pct > 0.15:
        issues.append((row.scope_change_pct, f"Scope grew mid-sprint ({row.scope_change_pct*100:.0f}% added after "
                       f"planning), which crowded out committed work.", rng.choice(ACTIONS_SCOPE)))
    if row.bugs_escaped_prod > 1 or row.bugs_found_qa > 4:
        issues.append((row.bugs_escaped_prod + row.bugs_found_qa * 0.3,
                       f"QA found {row.bugs_found_qa} bugs and {row.bugs_escaped_prod} escaped to production, "
                       f"pulling engineers into firefighting.", rng.choice(ACTIONS_QA)))
    if row.avg_team_attendance_rate < 0.85:
        issues.append((1 - row.avg_team_attendance_rate,
                       f"Attendance dipped to {row.avg_team_attendance_rate*100:.0f}% this sprint, reducing "
                       f"available capacity.", rng.choice(ACTIONS_ATTEND)))
    if row.test_coverage_pct < 60:
        issues.append((1 - row.test_coverage_pct / 100,
                       f"Test coverage sat at {row.test_coverage_pct:.0f}%, below the team's usual range, "
                       "making regressions more likely.", rng.choice(ACTIONS_QA)))

    body = []
    if rate_pct >= 95 and not issues:
        body.append("Clean sprint — commitment met with no notable blockers. Keep the current cadence.")
    elif issues:
        issues.sort(key=lambda t: -t[0])
        for _, desc, action in issues[:2]:
            body.append(desc + f" **Action item**: {action}.")
    else:
        body.append("Completion landed close to plan; no single factor stood out as a blocker this sprint.")

    return "\n".join(lines) + "\n" + "\n\n".join(body) + "\n"


def main():
    employees = build_employees()
    attendance = build_attendance(employees)
    sprints = build_sprints(employees, attendance)

    employees.drop(columns=["_reliability"]).to_csv("data/employees.csv", index=False)
    attendance.to_csv("data/attendance_monthly.csv", index=False)
    sprints.to_csv("data/sprints.csv", index=False)

    for _, row in sprints.iterrows():
        fname = f"docs/retros/{row.team.lower()}_sprint{row.sprint_number:02d}.md"
        with open(fname, "w", encoding="utf-8") as f:
            f.write(retro_note(row))

    print(f"employees: {len(employees)}")
    print(f"attendance_monthly: {len(attendance)} rows, high-risk rate {attendance['high_absenteeism_risk'].mean():.1%}")
    print(f"sprints: {len(sprints)} rows, mean completion rate {sprints['completion_rate'].mean():.1%}")
    print(f"retro notes written: {len(sprints)}")
    corr = sprints["avg_team_attendance_rate"].corr(sprints["completion_rate"])
    print(f"attendance <-> completion_rate correlation: {corr:.2f}")


if __name__ == "__main__":
    main()
