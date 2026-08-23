"""
Step 3: Given a user requirement/query, retrieve relevant context
from the vector store and ask phi3:mini (via Ollama) to generate
structured test cases as JSON.
"""

import json
import re
import requests
from config import OLLAMA_URL, LLM_MODEL
from vectorstore import retrieve

PROMPT_TEMPLATE = """You are an experienced QA engineer writing software test cases.

Use the CONTEXT below (retrieved from real requirement documents) to ground your answer.
Only use the REQUIREMENT as the thing you are testing; use CONTEXT for supporting detail.

CONTEXT:
{context}

REQUIREMENT TO TEST:
{requirement}

Generate 3 to 6 test cases covering positive, negative, and edge cases.

Vary the phrasing of each "title" - do not start every title with the same word.
Mix different openings naturally, for example: "Verify...", "Ensure...", "Confirm that...",
"Reject...", "Check whether...", "Validate...", or a direct description of the scenario.

Respond with ONLY a valid JSON array. No preamble, no explanation, no markdown fences,
no trailing commas, no comments.

Example of the exact JSON structure required (structure only - vary your own titles' wording,
don't reuse this example's exact phrasing):
[
  {{
    "id": "TC001",
    "title": "Complaint registration succeeds with valid complainant details",
    "preconditions": "Citizen is on the Registration module login page",
    "steps": ["Enter valid complainant details", "Enter complaint description", "Click Submit"],
    "expected_result": "Complaint is registered and a confirmation reference number is shown",
    "priority": "High"
  }}
]

Each item must have exactly these fields: "id", "title", "preconditions", "steps" (array of strings),
"expected_result", "priority" ("High", "Medium", or "Low").
"""


def build_prompt(requirement: str, context_chunks) -> str:
    context_text = "\n\n---\n".join(
        f"[Source: {c['source']} | Section: {c['section']}]\n{c['text']}" for c in context_chunks
    )
    return PROMPT_TEMPLATE.format(context=context_text, requirement=requirement)


def call_ollama(prompt: str, force_json: bool = True) -> str:
    payload = {
        "model": LLM_MODEL,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0.3,
            "num_predict": 1536,
            "num_ctx": 4096,  # more headroom so context + output doesn't get truncated
        },
    }
    if force_json:
        payload["format"] = "json"  # ask Ollama to constrain output to valid JSON syntax
    resp = requests.post(f"{OLLAMA_URL}/api/generate", json=payload, timeout=300)
    resp.raise_for_status()
    return resp.json()["response"]


MIN_REQUIREMENT_WORDS = 4

VALIDITY_PROMPT_TEMPLATE = """Decide whether the TEXT below is a software requirement, feature
description, or user story that a QA engineer could reasonably write test cases for.

It is NOT valid if it's a name, greeting, random statement, or anything unrelated to
software functionality - even if it superficially resembles a sentence.

TEXT: "{text}"

Respond with ONLY this JSON object and nothing else: {{"is_requirement": true}} or {{"is_requirement": false}}
"""


def is_plausible_requirement(text: str) -> bool:
    """
    Cheap heuristic gate before spending an LLM call: reject obviously
    too-short input (e.g. "hi", "test", a bare name) outright.
    """
    return len(text.strip().split()) >= MIN_REQUIREMENT_WORDS


def classify_is_requirement(text: str) -> bool:
    """
    Ask the LLM itself to judge whether the input is something a QA
    engineer could actually write test cases for. This is more reliable
    than an embedding-distance threshold, since a good distance cutoff
    varies by embedding model and is hard to pick blindly - the model
    can directly reason about what counts as a requirement.

    Uses structured JSON output (format:"json") rather than a plain
    YES/NO text answer - small models often wrap a plain-text answer in
    preamble ("This looks like a valid requirement. YES"), which breaks
    naive prefix matching. A JSON boolean is unambiguous to parse.

    Fails open (treats input as valid) if the check itself can't be
    completed cleanly, so a flaky classification never blocks a
    legitimate requirement - the downstream generation step has its own
    retries and will surface a clearer error if something's actually wrong.
    """
    prompt = VALIDITY_PROMPT_TEMPLATE.format(text=text.strip())
    try:
        raw = call_ollama(prompt, force_json=True)
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            return True
        parsed = json.loads(match.group(0))
        if isinstance(parsed, dict) and "is_requirement" in parsed:
            return bool(parsed["is_requirement"])
        return True
    except (requests.RequestException, json.JSONDecodeError, ValueError):
        return True


class NotARequirementError(ValueError):
    """Raised when the input doesn't look like something test cases can be written for."""
    pass


def repair_common_json_issues(text: str) -> str:
    """Fix small-model JSON slip-ups that a strict parser would otherwise reject."""
    # Trailing commas before a closing ] or }
    text = re.sub(r",\s*([\]}])", r"\1", text)
    return text


REQUIRED_FIELDS = ["id", "title", "preconditions", "steps", "expected_result", "priority"]


def normalize_to_list(parsed):
    """
    format:"json" guarantees valid JSON syntax, but not that the top level
    is an array. The model sometimes returns:
      - a plain array (expected case)
      - a single test case object directly, not wrapped in an array
      - an object wrapping the array under some key, e.g. {"test_cases": [...]}
    Normalize all three into a plain list of test case dicts.
    """
    if isinstance(parsed, list):
        return parsed
    if isinstance(parsed, dict):
        # If this dict itself looks like a single test case (has the fields
        # we expect), treat it as one test case rather than searching its
        # values - otherwise a field like "steps" (itself a list) gets
        # mistaken for the wrapped array of test cases.
        if any(field in parsed for field in REQUIRED_FIELDS):
            return [parsed]
        for value in parsed.values():
            if isinstance(value, list) and value and isinstance(value[0], dict):
                return value
        return [parsed]
    return []


def validate_test_cases(raw_list):
    """
    format:"json" guarantees valid JSON syntax, but not that every element
    matches our schema (e.g. the model can still emit a stray string, or an
    object missing fields, especially if output got truncated mid-array).
    Keep only well-formed test case objects; drop and report anything else
    instead of crashing.
    """
    valid, dropped = [], 0
    for item in raw_list:
        if not isinstance(item, dict):
            dropped += 1
            continue
        if not all(field in item for field in REQUIRED_FIELDS):
            dropped += 1
            continue
        if not isinstance(item.get("steps"), list):
            dropped += 1
            continue
        valid.append(item)

    if dropped:
        print(f"Note: dropped {dropped} malformed test case entr{'y' if dropped == 1 else 'ies'} "
              f"from the model output.")
    return valid


def extract_json_array(raw_text: str):
    """
    phi3:mini sometimes wraps JSON in markdown fences or adds stray text.
    This strips that, finds the JSON block, parses it, and returns a
    validated list of well-formed test case dicts.
    """
    cleaned = raw_text.strip()
    cleaned = re.sub(r"```json|```", "", cleaned).strip()

    match = re.search(r"[\[{].*[\]}]", cleaned, re.DOTALL)
    if not match:
        raise ValueError(f"No JSON found in model output:\n{raw_text}")

    json_str = match.group(0)
    try:
        parsed = json.loads(json_str)
    except json.JSONDecodeError:
        repaired = repair_common_json_issues(json_str)
        parsed = json.loads(repaired)

    raw_list = normalize_to_list(parsed)
    valid_list = validate_test_cases(raw_list)

    if not valid_list:
        raise ValueError("Parsed JSON but found no valid test case objects in it.")

    return valid_list


def generate_test_cases(collection, requirement: str, retries: int = 2, skip_validity_check: bool = False):
    """
    Full RAG call: validate input -> retrieve context -> build prompt ->
    call phi3:mini -> parse JSON.

    Raises NotARequirementError if the input doesn't look like something
    a QA engineer could write test cases for (e.g. a name, greeting, or
    unrelated text), instead of silently generating meaningless output.
    """
    if not skip_validity_check:
        if not is_plausible_requirement(requirement):
            raise NotARequirementError(
                "That doesn't look like a requirement or user story - it's too short. "
                "Try describing a feature or user action, e.g. "
                '"Citizen registers a complaint through the Registration module".'
            )
        if not classify_is_requirement(requirement):
            raise NotARequirementError(
                "That doesn't look like a software requirement or user story, so there's "
                "nothing meaningful to write test cases for. Try describing a feature or "
                'user action instead, e.g. "Citizen registers a complaint through the '
                'Registration module".'
            )

    context_chunks = retrieve(collection, requirement)
    prompt = build_prompt(requirement, context_chunks)

    last_error = None
    last_raw = None
    for attempt in range(1, retries + 2):
        raw = call_ollama(prompt)
        last_raw = raw
        try:
            test_cases = extract_json_array(raw)
            return test_cases, context_chunks
        except (ValueError, json.JSONDecodeError) as e:
            last_error = e
            print(f"Attempt {attempt}: model output wasn't valid JSON, retrying...")

    print("\n--- RAW MODEL OUTPUT (final failed attempt, for debugging) ---")
    print(last_raw)
    print("--- END RAW OUTPUT ---\n")

    raise RuntimeError(f"Failed to get valid JSON after {retries + 1} attempts. Last error: {last_error}")


def print_test_cases(test_cases):
    for tc in test_cases:
        if not isinstance(tc, dict):
            continue  # already filtered in extract_json_array, but stay safe
        print(f"\n[{tc.get('id')}] {tc.get('title')}  (Priority: {tc.get('priority')})")
        print(f"  Preconditions: {tc.get('preconditions')}")
        print("  Steps:")
        for i, step in enumerate(tc.get("steps", []), 1):
            print(f"    {i}. {step}")
        print(f"  Expected Result: {tc.get('expected_result')}")