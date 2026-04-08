"""Task 2 (Medium): Intent Classification — Zero Instructions.

The agent receives a raw message (email, support ticket, chat) and must infer
what action to take — with no instructions at all. It must figure out the
correct action type and produce a structured response.

Actions: reply, forward, archive, flag, escalate, delete
"""

import json
import random
from typing import Any, Dict, List, Tuple

TASK_BANK: List[Dict[str, Any]] = [
    # --- Reply ---
    {
        "input": {
            "from": "client@acme.com",
            "subject": "Quick question about pricing",
            "body": "Hi, could you send me the updated pricing sheet for Q2? Thanks!",
            "priority": "normal",
        },
        "expected": {
            "action": "reply",
            "priority": "normal",
            "summary": "Client requesting Q2 pricing sheet",
        },
        "category": "reply",
    },
    {
        "input": {
            "from": "manager@company.com",
            "subject": "Meeting tomorrow",
            "body": "Can you confirm your availability for the 2pm sync tomorrow?",
            "priority": "normal",
        },
        "expected": {
            "action": "reply",
            "priority": "normal",
            "summary": "Manager asking to confirm meeting availability",
        },
        "category": "reply",
    },
    # --- Escalate ---
    {
        "input": {
            "from": "angry_customer@gmail.com",
            "subject": "URGENT: Service down for 3 hours!!!",
            "body": "Your service has been completely down since 6am. We are losing thousands of dollars per hour. This is unacceptable. I need to speak with a manager NOW.",
            "priority": "urgent",
        },
        "expected": {
            "action": "escalate",
            "priority": "urgent",
            "summary": "Critical service outage causing financial loss, customer demands manager",
        },
        "category": "escalate",
    },
    {
        "input": {
            "from": "legal@partner.com",
            "subject": "Contract violation notice",
            "body": "We have identified a breach of Section 4.2 of our agreement. Please have your legal team respond within 48 hours.",
            "priority": "urgent",
        },
        "expected": {
            "action": "escalate",
            "priority": "urgent",
            "summary": "Legal notice about contract violation requiring legal team response",
        },
        "category": "escalate",
    },
    # --- Archive ---
    {
        "input": {
            "from": "noreply@newsletter.com",
            "subject": "Weekly Tech Digest - March 2025",
            "body": "Here are this week's top stories in tech...",
            "priority": "low",
        },
        "expected": {
            "action": "archive",
            "priority": "low",
            "summary": "Automated newsletter, no action needed",
        },
        "category": "archive",
    },
    {
        "input": {
            "from": "system@jira.com",
            "subject": "JIRA-1234 has been resolved",
            "body": "The issue JIRA-1234 'Fix login timeout' has been marked as resolved by dev-team.",
            "priority": "low",
        },
        "expected": {
            "action": "archive",
            "priority": "low",
            "summary": "Automated JIRA notification, issue already resolved",
        },
        "category": "archive",
    },
    # --- Flag ---
    {
        "input": {
            "from": "security@company.com",
            "subject": "Suspicious login attempt detected",
            "body": "We detected a login attempt from an unrecognized device in Russia at 3:42 AM. If this wasn't you, please secure your account immediately.",
            "priority": "high",
        },
        "expected": {
            "action": "flag",
            "priority": "high",
            "summary": "Security alert about suspicious login from unrecognized device",
        },
        "category": "flag",
    },
    # --- Forward ---
    {
        "input": {
            "from": "vendor@supplies.com",
            "subject": "Invoice #4521 attached",
            "body": "Please find attached the invoice for last month's supplies order. Payment due in 30 days.",
            "priority": "normal",
        },
        "expected": {
            "action": "forward",
            "priority": "normal",
            "summary": "Vendor invoice needs to be forwarded to finance/accounting",
        },
        "category": "forward",
    },
    {
        "input": {
            "from": "recruiter@talent.com",
            "subject": "Strong candidate for ML Engineer role",
            "body": "I have an excellent candidate with 5 years of PyTorch experience. Resume attached. Who should I direct them to?",
            "priority": "normal",
        },
        "expected": {
            "action": "forward",
            "priority": "normal",
            "summary": "Recruiter with ML engineer candidate, forward to hiring manager",
        },
        "category": "forward",
    },
    # --- Delete ---
    {
        "input": {
            "from": "promo@spam-store.com",
            "subject": "🔥 50% OFF EVERYTHING 🔥 LIMITED TIME!!!",
            "body": "CLICK HERE to claim your exclusive discount! Act now before it's too late! Unsubscribe link at bottom.",
            "priority": "low",
        },
        "expected": {
            "action": "delete",
            "priority": "low",
            "summary": "Spam promotional email",
        },
        "category": "delete",
    },
]


def pick_task(seed: int = None) -> Tuple[Any, Any, str]:
    """Pick a random task. Returns (input_data, expected_output, category)."""
    rng = random.Random(seed)
    task = rng.choice(TASK_BANK)
    return task["input"], task["expected"], task["category"]


def grade(response: str, expected: Dict[str, Any]) -> Tuple[float, str]:
    """Grade the agent's response. Returns (score, feedback)."""
    try:
        parsed = json.loads(response)
    except (json.JSONDecodeError, TypeError):
        # Try to extract action from plain text
        parsed = _parse_plaintext(response, expected)

    if not isinstance(parsed, dict):
        return 0.0, "Could not parse response as a structured action. Expected JSON with 'action' and 'priority' fields."

    score = 0.0
    feedback_parts = []

    # Action correctness (50% weight)
    resp_action = str(parsed.get("action", "")).strip().lower()
    exp_action = expected["action"].lower()
    if resp_action == exp_action:
        score += 0.50
        feedback_parts.append("Correct action.")
    else:
        feedback_parts.append(f"Wrong action: got '{resp_action}', expected '{exp_action}'.")

    # Priority correctness (30% weight)
    resp_priority = str(parsed.get("priority", "")).strip().lower()
    exp_priority = expected["priority"].lower()
    if resp_priority == exp_priority:
        score += 0.30
        feedback_parts.append("Correct priority.")
    else:
        feedback_parts.append(f"Wrong priority: got '{resp_priority}', expected '{exp_priority}'.")

    # Summary present and reasonable (20% weight)
    resp_summary = str(parsed.get("summary", "")).strip()
    if len(resp_summary) > 10:
        score += 0.20
        feedback_parts.append("Summary provided.")
    elif resp_summary:
        score += 0.10
        feedback_parts.append("Summary too short.")
    else:
        feedback_parts.append("Missing summary.")

    return round(score, 2), " ".join(feedback_parts)


def _parse_plaintext(text: str, expected: Dict[str, Any]) -> Dict[str, Any]:
    """Try to extract structured fields from plain text response."""
    result = {}
    text_lower = text.lower()

    actions = ["reply", "forward", "archive", "flag", "escalate", "delete"]
    for action in actions:
        if action in text_lower:
            result["action"] = action
            break

    priorities = ["urgent", "high", "normal", "low"]
    for priority in priorities:
        if priority in text_lower:
            result["priority"] = priority
            break

    # Use the full text as summary if we found an action
    if result:
        result["summary"] = text.strip()

    return result if result else text
