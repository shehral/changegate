# Never-claim register

## What changegate does not do

Neither the tool nor its documentation claims to:

- stop or detect prompt injection, jailbreaks or prompt extraction (RM-21, RM-22);
- vet MCP servers, skills or plugins (RM-16);
- contain agents, manage permissions, or stop destructive actions or sandbox escapes
  (RM-23, RM-24);
- secure memory or retrieval (RM-25);
- keep long sessions on-persona (RM-26);
- make outputs grounded (RM-29);
- make robots safe (RM-35);
- staff oversight or promise a response (RM-37);
- attest to or assess conformity with any regulation or standard (RM-38);
- detect or stop misuse (RM-40);
- set a builder's safety-capability trade-off (RM-07).

## Wording

The banned-phrase list is `RULES` in [`tools/never_claim_lint.py`](../tools/never_claim_lint.py);
CI fails on any match in the README, these docs, the examples, and every non-docstring string
in `src/changegate`. The lint catches words, not implications.

<!-- never-claim: quoted -->
| Instead of | Say |
|---|---|
| "clean", "no issues found" | "not detected at recall r" (r measured on a known-bad library, or "not measured") |
| "certified", "guaranteed", "ensures", "prevents" | the measurement and its envelope: "PASS against the user-declared envelope" |
| "the model is misaligned", "X% of fine-tunes are misaligned" | "lost refusal behavior on battery B at FPR ≤ z" |
| "detects deception" | the measured quantity, named |
| "automatic repair", "fixes misalignment" | "repair recipe R; re-verified on held-out endpoints" (only when the independence rule is met) |
<!-- never-claim: end -->
