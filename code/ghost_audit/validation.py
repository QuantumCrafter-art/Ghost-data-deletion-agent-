"""Strict challenge output and cross-field validation."""
class ContractError(ValueError):
    pass

_FIELDS = {"step_no", "phase", "system_id", "resource_id", "operation", "state", "evidence_id"}
_PHASE_ORDER = {"FINDING": 0, "REMEDIATION": 1, "VERIFICATION": 2}
_OPERATIONS = {
    "REPORT_REMAINING_RECORD": ("FINDING", "PRESENT", 0),
    "REPORT_EXPOSED_PII": ("FINDING", "PRESENT", 1),
    "REPORT_ACTIVE_DATA_FLOW": ("FINDING", "ACTIVE", 2),
    "REPORT_RESTORE_RISK": ("FINDING", "ACTIVE", 3),
    "DISABLE_DATA_FLOW": ("REMEDIATION", "DISABLED", 0),
    "BLOCK_RESTORE": ("REMEDIATION", "BLOCKED", 1),
    "DELETE_RECORD": ("REMEDIATION", "ABSENT", 2),
    "REDACT_RECORD": ("REMEDIATION", "REDACTED", 3),
    "ANONYMIZE_RECORD": ("REMEDIATION", "ANONYMIZED", 4),
    "VERIFY_RECORD_ABSENT": ("VERIFICATION", "ABSENT", 0),
    "VERIFY_RECORD_PRESENT": ("VERIFICATION", "PRESENT", 1),
    "VERIFY_RECORD_REDACTED": ("VERIFICATION", "REDACTED", 2),
    "VERIFY_RECORD_ANONYMIZED": ("VERIFICATION", "ANONYMIZED", 3),
    "VERIFY_DATA_FLOW_DISABLED": ("VERIFICATION", "DISABLED", 4),
    "VERIFY_RESTORE_BLOCKED": ("VERIFICATION", "BLOCKED", 5),
}
_REMEDIATION_VERIFICATION = {
    "DISABLE_DATA_FLOW": "VERIFY_DATA_FLOW_DISABLED",
    "BLOCK_RESTORE": "VERIFY_RESTORE_BLOCKED",
    "DELETE_RECORD": "VERIFY_RECORD_ABSENT",
    "REDACT_RECORD": "VERIFY_RECORD_REDACTED",
    "ANONYMIZE_RECORD": "VERIFY_RECORD_ANONYMIZED",
}
_REMEDIATION_FINDINGS = {
    "DISABLE_DATA_FLOW": {"REPORT_ACTIVE_DATA_FLOW"},
    "BLOCK_RESTORE": {"REPORT_RESTORE_RISK"},
    "DELETE_RECORD": {"REPORT_REMAINING_RECORD"},
    "REDACT_RECORD": {"REPORT_REMAINING_RECORD", "REPORT_EXPOSED_PII"},
    "ANONYMIZE_RECORD": {"REPORT_REMAINING_RECORD", "REPORT_EXPOSED_PII"},
}
_FINDING_REMEDIATIONS = {
    "REPORT_REMAINING_RECORD": {"DELETE_RECORD", "REDACT_RECORD", "ANONYMIZE_RECORD"},
    "REPORT_EXPOSED_PII": {"REDACT_RECORD", "ANONYMIZE_RECORD"},
    "REPORT_ACTIVE_DATA_FLOW": {"DISABLE_DATA_FLOW"},
    "REPORT_RESTORE_RISK": {"BLOCK_RESTORE"},
}

def _sort_key(step):
    phase, _, op_order = _OPERATIONS[step["operation"]]
    return (_PHASE_ORDER[phase], op_order, step["system_id"], step["resource_id"])

def validate_steps(status, steps, *, available_citations):
    citations = set(available_citations)
    if status not in {"COMPLETE", "INCOMPLETE", "NOT_APPLICABLE"}:
        raise ContractError(f"invalid deletion_status: {status}")
    if not isinstance(steps, list):
        raise ContractError("steps_json must be a JSON array")
    if status == "NOT_APPLICABLE":
        if steps:
            raise ContractError("NOT_APPLICABLE requires an empty steps array")
        return
    if not steps:
        raise ContractError(f"{status} requires evidence/remediation steps")

    seen_phases, phase_positions = set(), []
    seen_steps = set()
    for expected_no, step in enumerate(steps, 1):
        if not isinstance(step, dict) or set(step) != _FIELDS:
            actual = sorted(step) if isinstance(step, dict) else []
            raise ContractError(f"step {expected_no} must contain exactly the required fields; allowed keys are {sorted(_FIELDS)}, received keys are {actual}")
        if type(step["step_no"]) is not int or step["step_no"] != expected_no:
            raise ContractError("step numbers must be consecutive positive integers")
        for key in ("system_id", "resource_id", "operation", "phase", "state", "evidence_id"):
            if not isinstance(step[key], str):
                raise ContractError(f"{key} must be a string")
        if not step["system_id"] or not step["resource_id"]:
            raise ContractError("system_id and resource_id must be non-empty strings")
        rule = _OPERATIONS.get(step["operation"])
        if rule is None:
            raise ContractError(f"unsupported operation at step {expected_no}: {step['operation']!r}; use one of {sorted(_OPERATIONS)}")
        if (step["phase"], step["state"]) != rule[:2]:
            raise ContractError(f"step {expected_no} operation {step['operation']} requires phase={rule[0]} and state={rule[1]}; received phase={step['phase']!r}, state={step['state']!r}")
        duplicate_key = tuple(step[key] for key in ("phase", "system_id", "resource_id", "operation", "state", "evidence_id"))
        if duplicate_key in seen_steps:
            raise ContractError(f"duplicate step at position {expected_no}")
        seen_steps.add(duplicate_key)
        phase_positions.append(_sort_key(step))
        seen_phases.add(step["phase"])
        evidence_id = step["evidence_id"]
        if step["phase"] == "FINDING":
            if not evidence_id or evidence_id not in citations:
                raise ContractError(f"finding cites unsupported evidence: {evidence_id!r}")
        elif status == "COMPLETE":
            if step["phase"] != "VERIFICATION" or not evidence_id or evidence_id not in citations:
                raise ContractError("COMPLETE requires cited verification steps only")
        elif evidence_id:
            raise ContractError("INCOMPLETE remediation/future verification evidence_id must be empty")

    if phase_positions != sorted(phase_positions):
        raise ContractError("steps are not in required phase/operation/system/resource order")
    if status == "COMPLETE":
        if seen_phases != {"VERIFICATION"}:
            raise ContractError("COMPLETE requires verification steps only")
        return
    if seen_phases != {"FINDING", "REMEDIATION", "VERIFICATION"}:
        raise ContractError("INCOMPLETE requires findings, remediation, and verification")

    by_pair = {}
    for step in steps:
        pair = (step["system_id"], step["resource_id"])
        by_pair.setdefault(pair, []).append(step["operation"])
    for step in steps:
        pair = (step["system_id"], step["resource_id"])
        operations = set(by_pair[pair])
        if step["phase"] == "FINDING" and not (operations & _FINDING_REMEDIATIONS[step["operation"]]):
            raise ContractError(f"finding has no matching remediation for {pair}")
        if step["phase"] == "REMEDIATION":
            required = _REMEDIATION_VERIFICATION[step["operation"]]
            if required not in operations:
                raise ContractError(f"remediation has no matching verification for {pair}")
            if not any(finding in operations for finding in _REMEDIATION_FINDINGS[step["operation"]]):
                raise ContractError(f"remediation has no matching finding for {pair}")

