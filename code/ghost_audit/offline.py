"""Conservative, fully local prediction path using participant evidence only."""
from __future__ import annotations
import csv, io, json, re
from collections import defaultdict
from pathlib import Path
from .intent import explicitly_declines_deletion, explicitly_requests_other_action_only
from .identity import IdentityResolver, normalize_identifier
from .index import EvidenceEngine
from .policy import DeletionPolicy, PolicyRule
from .timeutil import parse_time
from .validation import validate_steps, ContractError

_SQL_INSERT = re.compile(r'(?is)INSERT\s+INTO\s+["`\[]?([A-Za-z0-9_.-]+)["`\]]?\s*(?:\((.*?)\))?\s+VALUES\s*\((.*)\)\s*;?\s*$')
_KV = re.compile(r'''([A-Za-z_][A-Za-z0-9_]*)=("[^"]*"|'[^']*'|[^\s]+)''')
_WORD = re.compile(r'(?<![A-Za-z0-9_]){}(?![A-Za-z0-9_])')
_TABLE_SYSTEM = {
    "customer_profiles":"account_service", "customer_search_documents":"search_service",
    "customer_analytics_profiles":"analytics_platform", "support_records":"support_platform",
    "support_attachments":"support_platform", "customer_exports":"export_service",
    "connector_subscriptions":"integration_service", "backup_restore_entries":"backup_service",
    "security_audit_records":"security_service",
}
_TABLE_CATEGORY = {
    "customer_profiles":"USER_PROFILE", "customer_search_documents":"SEARCH_INDEX_ENTRY",
    "search_documents_primary":"SEARCH_INDEX_ENTRY", "customer_analytics_profiles":"ANALYTICS_PROFILE",
    "support_records":"SUPPORT_MESSAGE", "support_attachments":"SUPPORT_ATTACHMENT",
    "customer_exports":"USER_EXPORT", "connector_subscriptions":"ACTIVE_DATA_FLOW",
    "backup_restore_entries":"BACKUP_ARCHIVE", "security_audit_records":"SECURITY_AUDIT_LOG",
}
_COLLECTION_CATEGORY = {
    "customer_profiles":"USER_PROFILE", "profiles":"USER_PROFILE",
    "customer_search_documents":"SEARCH_INDEX_ENTRY", "search_documents":"SEARCH_INDEX_ENTRY",
    "customer_analytics_profiles":"ANALYTICS_PROFILE", "analytics_profiles":"ANALYTICS_PROFILE",
    "customer_analytics_reports":"ANALYTICS_PROFILE",
    "support_conversations":"SUPPORT_MESSAGE", "support_records":"SUPPORT_MESSAGE",
    "ticket_attachments":"SUPPORT_ATTACHMENT", "support_attachments":"SUPPORT_ATTACHMENT",
    "customer_export_jobs":"USER_EXPORT", "customer_exports":"USER_EXPORT",
    "connector_configurations":"ACTIVE_DATA_FLOW", "connector_subscriptions":"ACTIVE_DATA_FLOW",
    "backup_catalog":"BACKUP_ARCHIVE", "backup_restore_entries":"BACKUP_ARCHIVE",
    "security_audit_events":"SECURITY_AUDIT_LOG", "security_audit_records":"SECURITY_AUDIT_LOG",
    "retained_order_records":"LEGAL_HOLD_RECORD", "deleted_customer_references":"ACCOUNT_TOMBSTONE",
}
_SCHEMA = {
    "customer_profiles":["resource_id","customer_reference","full_name","email","phone","email_hash","account_status","updated_at"],
    "customer_search_documents":["resource_id","customer_reference","display_name","email","phone","email_hash","index_name","updated_at"],
    "search_documents_primary":["resource_id","display_name","email","customer_reference","updated_at"],
    "customer_analytics_profiles":["resource_id","customer_reference","email","phone","email_hash","audience_cohort","updated_at"],
    "support_records":["resource_id","customer_reference","requester_name","requester_email","requester_phone","message_excerpt","updated_at"],
    "customer_exports":["resource_id","customer_reference","owner_email","owner_phone","object_key","updated_at"],
    "connector_subscriptions":["resource_id","customer_reference","customer_email","connector_name","enabled","last_delivery_status","updated_at"],
    "backup_restore_entries":["resource_id","customer_reference","customer_email","archive_key","restore_allowed","updated_at"],
    "security_audit_records":["resource_id","subject_reference","event_code","retention_until","updated_at"],
    "customer_data_primary":["resource_id","origin_system","collection_name","record_json","updated_at"],
    "customer_data_read_replica":["resource_id","origin_system","collection_name","record_json","updated_at"],
    "customer_data_reporting_replica":["resource_id","origin_system","collection_name","record_json","updated_at"],
    "customer_data_failover_replica":["resource_id","origin_system","collection_name","record_json","updated_at"],
    "customer_data_replica_eu_west":["resource_id","origin_system","collection_name","record_json","updated_at"],
    "customer_data_replica_ap_south":["resource_id","origin_system","collection_name","record_json","updated_at"],
}
_PII_FIELDS = {"email","phone","email_hash","full_name","requester_name","requester_email","requester_phone","owner_email","owner_phone","customer_email","name"}
_QUERY_FIELDS = {"lookup_key","query","search_query","old_value","new_value","previous_value","changed_value"}


def _truth(v):
    return str(v).strip().casefold() in {"true","1","yes","active","enabled","present","identifiable"}
def _nonnull(v):
    return v is not None and str(v).strip().casefold() not in {"","null","none","false"}
def _parse_record(text, schemas=None):
    text=(text or "").strip()
    m=_SQL_INSERT.match(text)
    if m:
        table, explicit, raw=m.groups(); table=table.rsplit('.',1)[-1]
        cols=[x.strip().strip('"`[]').casefold() for x in explicit.split(',')] if explicit else (schemas or {}).get(table,_SCHEMA.get(table))
        if not cols:return {"table":table,"raw":text}
        try: values=next(csv.reader(io.StringIO(raw),delimiter=',',quotechar="'",skipinitialspace=True))
        except (csv.Error,StopIteration): return {"table":table,"raw":text}
        if len(cols)!=len(values):return {"table":table,"raw":text}
        data={k:v.strip().strip('"') for k,v in zip(cols,values)}
        if data.get("record_json"):
            try:data.update(json.loads(data["record_json"]))
            except (json.JSONDecodeError,TypeError):pass
        data["_table"]=table
        return data
    try:
        obj=json.loads(text)
        if isinstance(obj,dict):return obj
    except (json.JSONDecodeError,TypeError):pass
    data={}
    for key,val in _KV.findall(text):
        value=val[1:-1] if val[:1] in {'"',"'"} else val
        data[key.casefold()]=value
    if data:return data
    return {"_raw":text}

def _category(data, text, policy, location=""):
    table=str(data.get("_table", "")).casefold()
    cat=str(data.get("data_category",data.get("category",data.get("record_type","")))).upper()
    collection=str(data.get("collection_name", data.get("table_name", ""))).casefold()
    if cat in policy.categories():return cat
    if any(key in data for key in ("email_opt_in", "marketing_opt_in", "marketing_consent")):
        return "MARKETING_PROFILE"
    if collection in _COLLECTION_CATEGORY:return _COLLECTION_CATEGORY[collection]
    for t,c in _TABLE_CATEGORY.items():
        if table==t or table.startswith(t+"_") or collection==t:return c
    if "requester_email" in data or "requester_phone" in data:return "SUPPORT_MESSAGE"
    if "file_name" in data or "download_available" in data:return "USER_EXPORT"
    if re.search(r"(?i)attachment|\.(?:pdf|jpg|png|zip)\b",text) and ("belongs only" in text.casefold() or "attachment" in text.casefold()):return "SUPPORT_ATTACHMENT"
    system=str(data.get("system_id",data.get("system",data.get("service","")))).casefold()
    system_categories={"account_service":"USER_PROFILE","search_service":"SEARCH_INDEX_ENTRY","analytics_platform":"ANALYTICS_PROFILE","support_platform":"SUPPORT_MESSAGE","backup_service":"BACKUP_ARCHIVE","security_service":"SECURITY_AUDIT_LOG","export_service":"USER_EXPORT","integration_service":"ACTIVE_DATA_FLOW"}
    if system in system_categories:return system_categories[system]
    if "owner_email" in data or "owner_phone" in data:return "USER_EXPORT"
    if "restore_allowed" in data or "archive_present" in data:return "BACKUP_ARCHIVE"
    if "enabled" in data and ("connector_name" in data or "last_delivery_status" in data):return "ACTIVE_DATA_FLOW"
    if collection=="security_audit_events":return "SECURITY_AUDIT_LOG"
    if collection in {"customer_profiles","profiles"}:return "USER_PROFILE"
    if collection in {"customer_search_documents","search_documents"}:return "SEARCH_INDEX_ENTRY"
    if collection in {"customer_analytics_profiles","analytics_profiles"}:return "ANALYTICS_PROFILE"
    loc=str(location).casefold()
    if "analytics_platform" in loc or ".profiles[" in loc:return "ANALYTICS_PROFILE"
    if "security_service" in loc or "audit_records" in loc:return "SECURITY_AUDIT_LOG"
    if "search_service" in loc or "index_queries" in loc or "hits.documents" in loc:return "SEARCH_INDEX_ENTRY"
    if "account_service" in loc or "profile_lookups" in loc:return "USER_PROFILE"
    if "support_platform" in loc or "tickets" in loc:return "SUPPORT_MESSAGE"
    if "connector" in loc or "integration_service" in loc:return "ACTIVE_DATA_FLOW"
    if "backup_service" in loc or "restore" in loc:return "BACKUP_ARCHIVE"
    if "export_service" in loc or "exports" in loc:return "USER_EXPORT"
    cats=policy.evidence_categories(text)
    if len(cats)==1:return next(iter(cats))
    return None

def _system(data, category, path="", location=""):
    for key in ("target_system","system_id","system","service","origin_system","source_system"):
        value=data.get(key)
        if isinstance(value,str) and value.strip():return value.strip()
    table=str(data.get("_table","" )).casefold()
    loc=str(location).casefold()
    for name in ("account_service","search_service","analytics_platform","support_platform","backup_service","security_service","export_service","integration_service","identity_service","object_storage"):
        if name in loc:return name
    if table.startswith("customer_data_"):
        return str(data.get("origin_system", "")) or None
    if table in _TABLE_SYSTEM:return _TABLE_SYSTEM[table]
    if table.startswith("customer_profiles_"):return "account_service"
    if table.startswith("customer_search_documents_") or table.startswith("search_documents_"):return "search_service"
    path=str(path).casefold()
    for token,system in (("account","account_service"),("profile","account_service"),("search","search_service"),("analytics","analytics_platform"),("support","support_platform"),("backup","backup_service"),("export","export_service"),("security","security_service"),("audit","security_service"),("connector","integration_service")):
        if token in path:return system
    return None

def _resource(data):
    for key in ("resource_id","resource","record_reference","record_ref","object_id"):
        value=data.get(key)
        if isinstance(value,str) and value.strip():return value.strip()
    return None

def _matches_subject(data, identifiers):
    for key,value in data.items():
        if key.startswith("_") or key.casefold() in _QUERY_FIELDS or not isinstance(value,(str,int,float)):continue
        norm=str(value).strip().casefold()
        if not norm:continue
        for field,alias in identifiers:
            if normalize_identifier(field,str(alias))==normalize_identifier(field,norm):return True
    return False

def _lookup_result_matches(data, identifiers):
    """Link a completed lookup to a subject without mistaking its query key for record PII."""
    lookup = data.get("lookup_key")
    if not isinstance(lookup, str) or not lookup.strip():
        return False
    linked = any(normalize_identifier(kind, lookup) == normalize_identifier(kind, alias)
                 for kind, alias in identifiers)
    if not linked:
        return False
    event = str(data.get("event", data.get("event_type", ""))).casefold()
    completed = any(token in event for token in ("_completed", "_complete", "_returned", "_readback"))
    has_outcome = any(_nonnull(data.get(key)) for key in
                      ("rows", "http_status", "status_code", "result", "enabled", "restore_allowed", "email", "profile"))
    return completed or has_outcome


def _subject_resource_link(data, identifiers):
    """Recognize explicit participant-evidence resource-link statements as candidates only."""
    fields=(str(data.get("details", "")), str(data.get("description", "")))
    relation=re.compile(r"\b(?:linked|links|mapped|maps|associated|belongs|attached)\b",re.I)
    for text in fields:
        if not relation.search(text):
            continue
        for kind,value in identifiers:
            if not value:
                continue
            if kind=="phone":
                digits="".join(ch for ch in text if ch.isdigit())
                normalized=normalize_identifier(kind,str(value))
                if normalized and normalized in digits:return True
            else:
                if re.search(r"(?<![A-Za-z0-9_.+%\-])"+re.escape(str(value))+r"(?![A-Za-z0-9_.\-])",text,re.I):return True
    return False


def _pii_present(data):
    return any(_nonnull(data.get(k)) for k in _PII_FIELDS)


def _rules_for_category(policy, category):
    rules = policy.lookup(category)
    if rules or category != "ACTIVE_DATA_FLOW":
        return rules
    # The project contract explicitly requires stopping recreating flows; the
    # supplied policy table does not define a separate ACTIVE_DATA_FLOW row.
    return (PolicyRule("contract_active_flow", category, "Always", "DISABLE_DATA_FLOW"),)

class OfflinePredictor:
    """Evidence-first deterministic fallback. Raises when no support permits a valid output."""
    def __init__(self,dataset_dir,engine=None,identities=None,policy=None):
        self.dataset=Path(dataset_dir)
        self.engine=engine or EvidenceEngine(self.dataset)
        self.identities=identities or IdentityResolver(self.dataset)
        self.policy=policy or DeletionPolicy(self.dataset)
        self.schemas={}
        by_source=defaultdict(list)
        for record in self.engine.records:
            if record.path.casefold().endswith(".sql"):
                by_source[record.source_id].append(record.text)
        for lines in by_source.values():
            sql="\n".join(lines)
            for match in re.finditer(r'(?is)CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?["`\[]?([A-Za-z0-9_.-]+)["`\]]?\s*\((.*?)\)\s*;',sql):
                table,body=match.groups(); cols=[]
                for piece in body.split(","):
                    m=re.match(r'\s*["`\[]?([A-Za-z_][A-Za-z0-9_]*)',piece)
                    if m and m.group(1).casefold() not in {"constraint","primary","unique","foreign","check"}:cols.append(m.group(1).casefold())
                if cols:self.schemas[table.rsplit(".",1)[-1]]=cols

    def _direct_records(self,request):
        resolved=self.identities.resolve_email(request["email"],request["audit_at"])
        if resolved.status!="RESOLVED":return resolved,None,set(),[]
        candidate=resolved.candidates[0]
        identifiers=[(field,value) for field,values in candidate.identifiers.items() for value in values]
        identifiers.append(("user_id",candidate.user_id))
        found=[]; sources=set(); seen=set()
        for field,value in identifiers:
            for hit in self.engine.search(value,request["audit_at"],limit=100):
                from .agent import DeletionAuditAgent
                if not DeletionAuditAgent._contains_identifier(hit.text,value,field):continue
                key=(hit.source_id,hit.location)
                if key in seen:continue
                seen.add(key);sources.add(hit.source_id)
                if hit.source_kind=="audit":continue
                data=_parse_record(hit.text,self.schemas); cat=_category(data,hit.text,self.policy,hit.location)
                system=_system(data,cat,hit.path,hit.location); resource=_resource(data)
                if not system or not resource:continue
                # Search hits that only repeat the input as a lookup key are not state evidence.
                subject=_matches_subject(data,identifiers) or _lookup_result_matches(data,identifiers)
                is_link_event=(str(data.get("event_type",data.get("event_name",""))).upper()=="LINK_RESOURCE_IDENTITY"
                               and _subject_resource_link(data,identifiers)) or _subject_resource_link(data,identifiers)
                if not subject and not is_link_event:continue
                found.append({"system":system,"resource":resource,"category":cat,"data":data,
                              "hit":hit,"subject_linked":True})
        return resolved,candidate,set(sources),found

    def predict(self,request):
        from .agent import DeletionAuditAgent
        if (explicitly_declines_deletion(request.get("request_text","")) or
                explicitly_requests_other_action_only(request.get("request_text",""))):
            return {"case_id":request["case_id"],"deletion_status":"NOT_APPLICABLE","steps_json":[]},set(),"explicit refusal"
        resolved,candidate,sources,records=self._direct_records(request)
        if resolved is None or resolved.status!="RESOLVED":
            raise ValueError("identity is ambiguous or unavailable")
        aliases=[(f,v) for f,values in candidate.identifiers.items() for v in values]+[("user_id",candidate.user_id)]
        # Enrich only resource pairs already tied to this subject by exact participant data.
        by_pair=defaultdict(list)
        for rec in records:by_pair[(rec["system"],rec["resource"])].append(rec)
        for (system,resource) in list(by_pair):
            for hit in self.engine.search(f"{system} {resource}",request["audit_at"],limit=100):
                if not re.search(r"(?<![A-Za-z0-9_])"+re.escape(resource)+r"(?![A-Za-z0-9_])",hit.text):continue
                data=_parse_record(hit.text,self.schemas);cat=_category(data,hit.text,self.policy,hit.location)
                rec_system=_system(data,cat,hit.path,hit.location);rec_resource=_resource(data)
                if rec_system==system and rec_resource==resource:
                    sources.add(hit.source_id)
                    by_pair[(system,resource)].append({"system":system,"resource":resource,"category":cat,"data":data,"hit":hit,"subject_linked":True})
        findings=[]; remediations=[]; verifications=[]
        for pair,items in sorted(by_pair.items()):
            # Keep the latest snapshot/observation for each category/table, retaining independent systems.
            newest={}
            for item in items:
                hit=item["hit"]
                key=(item.get("category"),str(item["data"].get("_table","")))
                rank=(hit.captured_at,hit.event_at or "",hit.location)
                if key not in newest or rank>newest[key][0]:newest[key]=(rank,item)
            for _,item in newest.values():
                data=item["data"];cat=item.get("category");hit=item["hit"]
                # Activity can link a subject to a candidate resource, but its result
                # never establishes resource state; require a readback or snapshot.
                if hit.source_kind=="activity":continue
                if not cat:
                    continue
                rules=_rules_for_category(self.policy,cat)
                if not rules:continue
                rule=rules[0]
                assessment_text=re.sub(r"(?i)\blookup_key\s*[=:]\s*[\"']?[^\s,}]+", "", hit.text)
                assessment=self.policy.assess(rule,[(hit.source_id,assessment_text)],aliases,request["audit_at"])
                action=rule.required_action.upper()
                table=str(data.get("_table","" )).casefold()
                status=str(data.get("status",data.get("record_state",data.get("account_status","")))).casefold()
                pii=_pii_present(data)
                explicit_absent=(str(data.get("rows",""))=="0" or str(data.get("result","" )).casefold() in {"miss","not_found","absent"}
                                 or str(data.get("http_status",""))=="404")
                if cat=="SECURITY_AUDIT_LOG":
                    if assessment.applicability=="APPLIES":
                        verifications.append((pair,"VERIFY_RECORD_PRESENT","PRESENT",hit.source_id));continue
                    if assessment.applicability=="UNKNOWN":continue
                if cat=="ACCOUNT_TOMBSTONE":
                    stable_only=(str(data.get("account_status","" )).upper()=="DELETED" and not pii)
                    if stable_only:
                        verifications.append((pair,"VERIFY_RECORD_PRESENT","PRESENT",hit.source_id));continue
                    if pii: action="REDACT"
                if cat=="ANALYTICS_PROFILE" and not pii:
                    verifications.append((pair,"VERIFY_RECORD_ANONYMIZED","ANONYMIZED",hit.source_id));continue
                if cat=="BACKUP_ARCHIVE":
                    if _truth(data.get("restore_allowed")):
                        action="BLOCK_RESTORE"
                        assessment=type(assessment)("APPLIES",assessment.evidence_ids,"explicit restore_allowed flag")
                    elif str(data.get("restore_allowed","")).casefold() in {"false","0","no"}:
                        verifications.append((pair,"VERIFY_RESTORE_BLOCKED","BLOCKED",hit.source_id));continue
                    else:continue
                if cat=="ACTIVE_DATA_FLOW":
                    if _truth(data.get("enabled")):action="DISABLE_DATA_FLOW"
                    elif str(data.get("enabled","")).casefold() in {"false","0","no"}:
                        verifications.append((pair,"VERIFY_DATA_FLOW_DISABLED","DISABLED",hit.source_id));continue
                    else:continue
                if assessment.applicability=="UNKNOWN" and rule.condition.strip().casefold()!="always":continue
                if assessment.applicability=="DOES_NOT_APPLY":continue
                if action=="RETAIN":
                    verifications.append((pair,"VERIFY_RECORD_PRESENT","PRESENT",hit.source_id));continue
                if explicit_absent or status in {"absent","deleted","redacted","anonymized","disabled","blocked"}:
                    op="VERIFY_RECORD_ABSENT" if status in {"absent","deleted"} or explicit_absent else {
                        "redacted":"VERIFY_RECORD_REDACTED","anonymized":"VERIFY_RECORD_ANONYMIZED","disabled":"VERIFY_DATA_FLOW_DISABLED","blocked":"VERIFY_RESTORE_BLOCKED"}[status]
                    state={"VERIFY_RECORD_ABSENT":"ABSENT","VERIFY_RECORD_REDACTED":"REDACTED","VERIFY_RECORD_ANONYMIZED":"ANONYMIZED","VERIFY_DATA_FLOW_DISABLED":"DISABLED","VERIFY_RESTORE_BLOCKED":"BLOCKED"}[op]
                    verifications.append((pair,op,state,hit.source_id));continue
                # A subject-linked record can violate an unconditional DELETE rule
                # even when its visible fields are only a stable reference/hash.
                # Conditional categories have already failed closed above if their
                # supplied predicate is unknown.
                if not pii and not item.get("subject_linked") and cat not in {"BACKUP_ARCHIVE","ACTIVE_DATA_FLOW"}:continue
                report_op={"DISABLE_DATA_FLOW":"REPORT_ACTIVE_DATA_FLOW","BLOCK_RESTORE":"REPORT_RESTORE_RISK"}.get(action, "REPORT_EXPOSED_PII" if pii and action in {"REDACT", "ANONYMIZE"} else "REPORT_REMAINING_RECORD")
                report_state="ACTIVE" if action in {"DISABLE_DATA_FLOW","BLOCK_RESTORE"} else "PRESENT"
                remediation={"DELETE":"DELETE_RECORD","REDACT":"REDACT_RECORD","ANONYMIZE":"ANONYMIZE_RECORD","DISABLE_DATA_FLOW":"DISABLE_DATA_FLOW","BLOCK_RESTORE":"BLOCK_RESTORE"}.get(action)
                if not remediation:continue
                findings.append((pair,report_op,report_state,hit.source_id))
                remediations.append((pair,remediation))
        # Deduplicate evidence-backed facts before contract serialization.
        findings=list(dict.fromkeys(findings)); remediations=list(dict.fromkeys(remediations)); verifications=list(dict.fromkeys(verifications))
        if findings:
            # Every remediation gets its matching finding and future verification.
            steps=[]
            for pair,op,state,eid in findings:
                steps.append({"system_id":pair[0],"resource_id":pair[1],"operation":op,"phase":"FINDING","state":state,"evidence_id":eid})
            for pair,op in remediations:
                steps.append({"system_id":pair[0],"resource_id":pair[1],"operation":op,"phase":"REMEDIATION","state":{"DELETE_RECORD":"ABSENT","REDACT_RECORD":"REDACTED","ANONYMIZE_RECORD":"ANONYMIZED","DISABLE_DATA_FLOW":"DISABLED","BLOCK_RESTORE":"BLOCKED"}[op],"evidence_id":""})
            # In an INCOMPLETE audit, retain checks for already-observed compliant
            # resources too; the contract requires verification coverage for the case,
            # while evidence_id is empty for every future/post-remediation check.
            verify_state={"VERIFY_RECORD_ABSENT":"ABSENT","VERIFY_RECORD_PRESENT":"PRESENT",
                          "VERIFY_RECORD_REDACTED":"REDACTED","VERIFY_RECORD_ANONYMIZED":"ANONYMIZED",
                          "VERIFY_DATA_FLOW_DISABLED":"DISABLED","VERIFY_RESTORE_BLOCKED":"BLOCKED"}
            verification_specs={(pair,op,state) for pair,op,state,_ in verifications}
            remediation_verify={"DELETE_RECORD":"VERIFY_RECORD_ABSENT","REDACT_RECORD":"VERIFY_RECORD_REDACTED",
                                "ANONYMIZE_RECORD":"VERIFY_RECORD_ANONYMIZED","DISABLE_DATA_FLOW":"VERIFY_DATA_FLOW_DISABLED",
                                "BLOCK_RESTORE":"VERIFY_RESTORE_BLOCKED"}
            for pair,op in remediations:
                verify=remediation_verify[op]
                verification_specs.add((pair,verify,verify_state[verify]))
            for (system_id,resource_id),operation,state in sorted(verification_specs):
                steps.append({"system_id":system_id,"resource_id":resource_id,"operation":operation,
                              "phase":"VERIFICATION","state":state,"evidence_id":""})
            status="INCOMPLETE"
        elif verifications:
            steps=[{"system_id":p[0],"resource_id":p[1],"operation":op,"phase":"VERIFICATION","state":st,"evidence_id":eid} for p,op,st,eid in verifications]
            status="COMPLETE"
        else:
            raise ValueError("no sufficiently specific subject-linked state evidence")
        order={"FINDING":0,"REMEDIATION":1,"VERIFICATION":2}
        oporder={"REPORT_REMAINING_RECORD":0,"REPORT_EXPOSED_PII":1,"REPORT_ACTIVE_DATA_FLOW":2,"REPORT_RESTORE_RISK":3,"DISABLE_DATA_FLOW":0,"BLOCK_RESTORE":1,"DELETE_RECORD":2,"REDACT_RECORD":3,"ANONYMIZE_RECORD":4,"VERIFY_RECORD_ABSENT":0,"VERIFY_RECORD_PRESENT":1,"VERIFY_RECORD_REDACTED":2,"VERIFY_RECORD_ANONYMIZED":3,"VERIFY_DATA_FLOW_DISABLED":4,"VERIFY_RESTORE_BLOCKED":5}
        steps.sort(key=lambda x:(order[x["phase"]],oporder[x["operation"]],x["system_id"],x["resource_id"],x["evidence_id"]))
        for n,step in enumerate(steps,1):step["step_no"]=n
        # Restrict citations to source IDs actually read by this deterministic pass and visible at audit_at.
        citations=self.engine.valid_citations(request["audit_at"],sources)
        validate_steps(status,steps,available_citations=citations)
        return {"case_id":request["case_id"],"deletion_status":status,"steps_json":steps},sources,"deterministic evidence/policy pass"
