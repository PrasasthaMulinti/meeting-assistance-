"""
Action item extraction using rule-based keywords and POS tagging.
"""
from typing import List, Dict, Tuple
import nltk
from nltk import pos_tag
from nltk.tokenize import sent_tokenize
from src.utils.helpers import tokenize

nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

# Action-indicating patterns
ACTION_KEYWORDS = {
    "will",
    "should",
    "need to",
    "must",
    "have to",
    "going to",
    "assign",
    "assignee",
    "deadline",
    "follow up",
    "action",
    "task",
    "todo",
    "please",
    "reminder",
    "scheduled",
    "ensure",
    "implement",
    "complete",
    "review",
    "submit",
    "send",
    "prepare",
    "update",
    "schedule",
    "coordinate",
}


def _sentence_contains_action_keyword(sentence: str) -> bool:
    """Check if sentence contains any action-indicating keyword."""
    lower = sentence.lower()
    return any(kw in lower for kw in ACTION_KEYWORDS)


def _has_verb_near_start(
    tokens: List[str], tags: List[Tuple[str, str]], window: int = 5
) -> bool:
    """Check if there is a verb in the first 'window' tokens."""
    vb_tags = {"VB", "VBD", "VBG", "VBN", "VBP", "VBZ"}
    for i, (_, tag) in enumerate(tags[:window]):
        if tag in vb_tags:
            return True
    return False


def extract_action_items(transcript: str) -> List[Dict[str, str]]:
    """
    Extract action items from meeting transcript.
    Returns list of dicts with keys: 'text', 'assignee' (if detected), 'raw_sentence'.
    """
    action_items = []
    sentences = sent_tokenize(transcript)
    for sent in sentences:
        if not _sentence_contains_action_keyword(sent):
            continue
        tokens = tokenize(sent)
        if len(tokens) < 3:
            continue
        tags = pos_tag(tokens)
        if not _has_verb_near_start(tokens, tags):
            continue
        # Simple assignee: look for "X will" or "X to"
        assignee = None
        for i, tok in enumerate(tokens):
            if (
                tok in ("will", "to")
                and i > 0
                and tokens[i - 1] not in ("to", "need", "have", "going")
            ):
                assignee = " ".join(tokens[max(0, i - 2) : i])
                break
        action_items.append(
            {
                "text": sent.strip(),
                "assignee": assignee or "Unassigned",
                "raw_sentence": sent.strip(),
            }
        )
    return action_items
