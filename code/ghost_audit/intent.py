"""Conservative affirmative evidence check for NOT_APPLICABLE decisions."""
import re

# This is intentionally one-sided: it detects explicit refusals/cancellations but
# never infers authorization from an unknown or empty request.
_EXPLICIT_NEGATION_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in (
    r"\b(?:do not|don['’]?t|never|no)\s+(?:want|wish|need|ask|request)?\s*(?:you\s+to\s+)?(?:delete|remove|erase|wipe|close|deactivate)\b",
    r"\bnot\s+(?:asking|requesting|authorizing)\b.{0,60}\b(?:delete|remove|erase|wipe|close|deactivate|deletion)\b",
    r"\b(?:cancel|withdraw|revoke|stop)\b.{0,60}\b(?:deletion|delete|removal|erase|account closure)\b",
    r"\b(?:no longer|never)\s+(?:want|wish)\b.{0,60}\b(?:delete|remove|erase|close)\b",
    r"\b(?:no deletion|without deleting|not for deletion)\b",
    r"\b(?:no quiero|no deseo|no solicito)\b.{0,80}\b(?:borrar|eliminar|suprimir|cerrar)\b",
    r"\b(?:não quero|nao quero|não desejo|nao desejo)\b.{0,80}\b(?:apagar|excluir|deletar|remover)\b",
    r"\b(?:non voglio|non desidero)\b.{0,80}\b(?:cancellare|eliminare|rimuovere)\b",
    r"\b(?:nicht löschen|nicht entfernen|möchte nicht gelöscht)\b",
    r"\b(?:not delete|ne pas supprimer|ne veux pas supprimer)\b",
    r"(?:削除しない|削除を希望しない|削除を望みません|削除依頼を取り消す|削除要求を取り消す)",
    r"(?:삭제하지|삭제를 원하지|삭제 요청을 취소|삭제요청 취소)",
    r"(?:不要删除|不删除|不想删除|撤回删除|取消删除)",
))

def explicitly_declines_deletion(request_text):
    if not isinstance(request_text, str) or not request_text.strip():
        return False
    text = " ".join(request_text.casefold().split())
    return any(pattern.search(text) for pattern in _EXPLICIT_NEGATION_PATTERNS)


_OTHER_ACTION_ONLY_PATTERNS = tuple(re.compile(pattern, re.IGNORECASE | re.DOTALL) for pattern in (
    r"\b(?:download|export|access|view|obtain)\b.{0,100}\b(?:account information|personal data|my data|my information)\b.{0,80}\b(?:keep|leave)\b.{0,30}\b(?:account|records)\b.{0,25}\b(?:as is|unchanged|intact)\b",
    r"\b(?:correct|fix|update)\b.{0,100}\b(?:without|and do not)\b.{0,35}\b(?:deleting|removing|erasing)\b.{0,30}\b(?:records|data|account)\b",
    r"\b(?:remove|unsubscribe|opt[- ]?out)\b.{0,60}\b(?:me )?from\b.{0,40}\b(?:marketing|promotional)\b.{0,30}\bonly\b",
))


def explicitly_requests_other_action_only(request_text):
    """Recognize clear access/correction/marketing-only requests as non-deletion intent."""
    if not isinstance(request_text, str) or not request_text.strip():
        return False
    text = " ".join(request_text.casefold().split())
    return any(pattern.search(text) for pattern in _OTHER_ACTION_ONLY_PATTERNS)
