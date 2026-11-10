"""The registry of approved policies, looked up by their version name.

WHAT THIS FILE IS
-----------------
A run doesn't carry its scientific limits around; it names a POLICY VERSION
(D1: `policy_version`, e.g. "placeholder-0"). This file maps each version name
to the actual Policy object (state.py): the OPV-likeness minimum, the
uncertainty limits and the time budget.

WHY A REGISTRY: D2 says policies are approved by the team and IMMUTABLE (they
never change once registered). A run started today and one started next month
with the same version name use exactly the same limits. To change a limit,
register a NEW version name; never edit an existing one.

The web API checks a request's `policy_version` here: an unknown name is
refused with 422 (D1). The worker looks the policy up again when it runs the
run, since only the NAME is stored in the database.

!!! NO REAL POLICY IS APPROVED YET (peer review PR-06) !!!
The only entry is a placeholder whose numbers suit the FAKE predictor.
"""

from .state import Policy

# PLACEHOLDER limits, made up to suit the fake predictor (stubs.py). Moved here
# from __main__.py (where it was called DEMO_POLICY) so the API can use it too.
PLACEHOLDER_POLICY = Policy(
    policy_version="placeholder-0",
    opv_min=0.5,  # OPV-likeness below 0.5 -> Stop
    # A value whose uncertainty is above these limits (or unknown) -> Escalate.
    uncertainty_max_by_target={"avt_fraction": 0.05, "pce_percent": 1.0, "lue_percent": 0.8},
    # max_runtime_seconds keeps its default: 1800 s (D2's 30-minute budget) of working time.
)

# All registered policies: version name -> Policy. Add new versions to this list.
# (A dictionary comprehension: {key: value for item in list}.)
POLICIES: dict[str, Policy] = {p.policy_version: p for p in [PLACEHOLDER_POLICY]}
