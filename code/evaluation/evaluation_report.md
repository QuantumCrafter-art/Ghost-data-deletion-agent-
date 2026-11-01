# Public sample evaluation

The deterministic local fallback checkpointed 21 of 32 public samples. Metrics below are conditional on those 21 scored cases; coverage and validation are separate. The predictor does not read or use the public reference file.

- Exact-match accuracy: 0.0476 (1/21 scored cases)
- Status accuracy: 0.9524 (20/21)
- Step accuracy: 0.1915
- Field accuracy: 0.3658
- Sequence accuracy: 0.0476
- Coverage: 21/32 (0.6563)
- Invalid output/citation rows: 0 among scored predictions
- Duplicate sample rows: 0

Per-field mismatch counts among scored cases:

- deletion_status: 1
- step_no: 62
- phase: 80
- system_id: 104
- resource_id: 104
- operation: 107
- state: 100
- evidence_id: 69

Largest measured patterns: missing steps (20 cases), sequence mismatch (20), wrong system/resource (20), phase/operation/state mismatch (16), and citation mismatch (7).

## Improvements and regressions

- Added policy-category mappings for collection names present in the SQL corpus, including support conversations, analytics reports, connector configurations, backup catalogs, exports, and retained orders.
- Treated completed identifier-scoped lookup results as state evidence without treating query keys as stored PII.
- Added explicit subject/resource link statements as discovery candidates only; activity records are never accepted as proof of resulting state.
- Enforced shutdown of active recreation flows under the project contract even though the supplied policy table has no separate ACTIVE_DATA_FLOW rule.
- Classified records with explicit marketing opt-in fields as MARKETING_PROFILE and enforced deletion for subject-linked rows even when only a stable reference/hash is exposed.
- Added handling for already-observed policy-compliant verifications in incomplete cases.
- Added focused tests for link boundaries, completed lookups, query-key/PII separation, corpus collection mappings, active-flow treatment, explicit non-deletion intent, and evidence-gap safety.

## Full request batch

The latest clean local pass visited all 518 requests and checkpointed 389 contract-valid, evidence-grounded predictions. It left 130 unresolved; no complete root output.csv was created.

Diagnostic groups: 61 cases had no subject-linked resource/state record; 69 had linked candidate records but no policy/state-supported decision. The unresolved records included 111 security-audit candidates, 35 support-message candidates, 13 support-attachment candidates, and 11 uncategorized candidates; those record counts can overlap by case. Conditional retention/shared-context/ownership facts remain unknown where the supplied evidence does not establish them. Candidate activity links were not treated as state proof.

## Per-case reference comparison

“—” means no prediction was safely produced, so that case is unscored. Field names below identify every mismatch against the public reference.

| Case | Status (reference → prediction) | Steps (reference/prediction) | Exact | Mismatched fields |
|---|---|---:|:---:|---|
| case_01 | COMPLETE → — | 1/— | — | unscored |
| case_02 | COMPLETE → — | 1/— | — | unscored |
| case_03 | COMPLETE → COMPLETE | 3/1 | no | system_id[1], resource_id[1], operation[1], state[1], missing_step_2, missing_step_3 |
| case_04 | COMPLETE → — | 2/— | — | unscored |
| case_05 | INCOMPLETE → INCOMPLETE | 3/3 | yes | — |
| case_06 | INCOMPLETE → INCOMPLETE | 4/3 | no | system_id[3], resource_id[3], missing_step_4 |
| case_07 | INCOMPLETE → — | 4/— | — | unscored |
| case_08 | INCOMPLETE → INCOMPLETE | 4/3 | no | system_id[3], resource_id[3], missing_step_4 |
| case_09 | INCOMPLETE → INCOMPLETE | 5/3 | no | system_id[3], resource_id[3], operation[3], state[3], missing_step_4, missing_step_5 |
| case_10 | INCOMPLETE → — | 4/— | — | unscored |
| case_11 | INCOMPLETE → — | 14/— | — | unscored |
| case_12 | INCOMPLETE → INCOMPLETE | 5/3 | no | system_id[3], resource_id[3], missing_step_4, missing_step_5 |
| case_13 | INCOMPLETE → — | 8/— | — | unscored |
| case_14 | INCOMPLETE → — | 5/— | — | unscored |
| case_15 | INCOMPLETE → COMPLETE | 5/1 | no | deletion_status, phase[1], system_id[1], resource_id[1], operation[1], state[1], missing_step_2, missing_step_3, missing_step_4, missing_step_5 |
| case_16 | INCOMPLETE → INCOMPLETE | 7/3 | no | phase[2], system_id[2], resource_id[2], operation[2], state[2], evidence_id[2], phase[3], operation[3], missing_step_4, missing_step_5, missing_step_6, missing_step_7 |
| case_17 | INCOMPLETE → — | 5/— | — | unscored |
| case_18 | INCOMPLETE → INCOMPLETE | 14/9 | no | system_id[3], resource_id[3], operation[3], state[3], phase[4], operation[4], state[4], evidence_id[4], system_id[5], resource_id[5], operation[5], state[5], system_id[6], resource_id[6], operation[6], state[6], phase[7], system_id[7], resource_id[7], operation[7], state[7], phase[8], operation[8], system_id[9], resource_id[9], operation[9], state[9], missing_step_10, missing_step_11, missing_step_12, missing_step_13, missing_step_14 |
| case_19 | INCOMPLETE → INCOMPLETE | 8/3 | no | phase[2], system_id[2], resource_id[2], operation[2], state[2], evidence_id[2], phase[3], operation[3], missing_step_4, missing_step_5, missing_step_6, missing_step_7, missing_step_8 |
| case_20 | INCOMPLETE → INCOMPLETE | 8/6 | no | system_id[5], resource_id[5], system_id[6], resource_id[6], operation[6], state[6], missing_step_7, missing_step_8 |
| case_21 | COMPLETE → COMPLETE | 3/1 | no | system_id[1], resource_id[1], operation[1], state[1], missing_step_2, missing_step_3 |
| case_22 | INCOMPLETE → INCOMPLETE | 8/3 | no | system_id[1], resource_id[1], operation[1], phase[2], operation[2], state[2], evidence_id[2], phase[3], system_id[3], resource_id[3], operation[3], state[3], missing_step_4, missing_step_5, missing_step_6, missing_step_7, missing_step_8 |
| case_23 | INCOMPLETE → INCOMPLETE | 8/6 | no | system_id[5], resource_id[5], system_id[6], resource_id[6], missing_step_7, missing_step_8 |
| case_24 | INCOMPLETE → INCOMPLETE | 7/3 | no | system_id[1], resource_id[1], phase[2], operation[2], state[2], evidence_id[2], phase[3], system_id[3], resource_id[3], operation[3], missing_step_4, missing_step_5, missing_step_6, missing_step_7 |
| case_25 | COMPLETE → COMPLETE | 3/1 | no | system_id[1], resource_id[1], operation[1], state[1], missing_step_2, missing_step_3 |
| case_26 | COMPLETE → COMPLETE | 6/3 | no | system_id[1], resource_id[1], operation[1], state[1], system_id[2], resource_id[2], operation[2], state[2], system_id[3], resource_id[3], operation[3], state[3], missing_step_4, missing_step_5, missing_step_6 |
| case_27 | COMPLETE → COMPLETE | 6/3 | no | system_id[1], resource_id[1], operation[1], state[1], system_id[2], resource_id[2], operation[2], state[2], system_id[3], resource_id[3], operation[3], state[3], missing_step_4, missing_step_5, missing_step_6 |
| case_28 | COMPLETE → COMPLETE | 6/3 | no | system_id[1], resource_id[1], operation[1], state[1], system_id[2], resource_id[2], operation[2], state[2], system_id[3], resource_id[3], operation[3], state[3], missing_step_4, missing_step_5, missing_step_6 |
| case_29 | INCOMPLETE → INCOMPLETE | 14/9 | no | system_id[3], resource_id[3], operation[3], state[3], phase[4], operation[4], state[4], evidence_id[4], system_id[5], resource_id[5], operation[5], state[5], system_id[6], resource_id[6], operation[6], state[6], phase[7], system_id[7], resource_id[7], operation[7], state[7], phase[8], operation[8], system_id[9], resource_id[9], operation[9], state[9], missing_step_10, missing_step_11, missing_step_12, missing_step_13, missing_step_14 |
| case_30 | INCOMPLETE → INCOMPLETE | 14/9 | no | system_id[3], resource_id[3], operation[3], state[3], phase[4], operation[4], state[4], evidence_id[4], system_id[5], resource_id[5], operation[5], state[5], system_id[6], resource_id[6], operation[6], state[6], phase[7], system_id[7], resource_id[7], operation[7], state[7], phase[8], operation[8], system_id[9], resource_id[9], operation[9], state[9], missing_step_10, missing_step_11, missing_step_12, missing_step_13, missing_step_14 |
| case_31 | COMPLETE → — | 3/— | — | unscored |
| case_32 | INCOMPLETE → — | 5/— | — | unscored |
