import json
import os
import re
from string import Template
import time

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))  # Add the parent directory SSRL_simple to the search path

from utils_Qwen_8B import APILLM



# Atomic proposition decomposition + symbolic logic generation
PROMPT_ATOMIC_FULL = '''You are a logical atomization assistant. Given a natural language sentence, decompose it into the smallest set of atomic propositions (no logical connectives inside) and represent the logical structure using these atoms.

Output a JSON object with:
- "atomic_facts": a dictionary mapping atomic IDs (A1, A2, ...) to their English statements, preserving original meaning.
- "symbolic_logic": a logical formula using these atomic_facts and connectives (∧, ∨, ¬, →, ↔, ⊕) and parentheses.

Rules:
1. Each atomic statement must be a declarative clause without "and", "or", "if...then", "not", or similar logical words inside.
2. Keep the original tense and phrasing as much as possible; do not add or omit key meaning.
3. If the sentence contains adverbs like "often", "usually", "always" — ignore them for the pure logical form (treat as a universal/general implication unless specified otherwise).
4. Exclusive OR (choosing one out of two, such as "either... or"), use ⊕.
5. Output only valid JSON, no extra text.

Example:
Input: "If Bonnie is either both a young child or teenager who wishes to further her academic career and chaperones high school dances or neither a young child nor teenager who wishes to further her academic career, then Bonnie is either a student who attends the school or is an inactive member."
Output:
{
  "atomic_facts": {
    "A1": "Bonnie is a young child.",
    "A2": "Bonnie is a teenager.",
    "A3": "Bonnie wishes to further her academic career.",
    "A4": "Bonnie chaperones high school dances.",
    "A5": "Bonnie is a student.",
    "A6": "Bonnie attends the school.",
    "A7": "Bonnie is an inactive member.",
  },
  "symbolic_logic": "(((A1 ∨ A2) ∧ A3 ∧ A4) ⊕ (¬(A1 ∨ A2) ∧ ¬A3)) → ((A5 ∧ A6) ⊕ A7)"
}

Now process the following input:
Input: "$user_fact"

(Note: Only the result in correct JSON format is required!)'''

def atomic_str(data: dict) -> str:
    """Convert the atomic facts dictionary into the formatted string."""
    result = ""
    for key, val in data.items():
        result += f'- {key}: "{val}",\n'
    result = result.rstrip(",\n")
    return result

def extract_json_from_text(text: str) -> str:
    """
    Generic JSON extractor: clean extra text, code blocks, and comments, then capture the outermost {} content.
    Handles: ```json code blocks, surrounding explanatory text, redundant newlines/spaces.
    """
    # 1. Remove code block markers ```json / ```
    text = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE)
    # 2. Strip comments line by line (simple // comments)
    lines = [line.split("//")[0] for line in text.splitlines()]
    clean_text = "\n".join(lines)
    # 3. Capture from the first '{' to the last '}' to form a standard JSON region
    left_idx = clean_text.find("{")
    right_idx = clean_text.rfind("}")
    if left_idx == -1 or right_idx == -1 or left_idx > right_idx:
        return ""
    json_str = clean_text[left_idx: right_idx + 1]
    # 4. Trim leading/trailing whitespace
    return json_str.strip()

def safe_json_load(text: str) -> dict:
    """
    Multi-level fault-tolerant JSON parsing:
    1. First try full JSON parsing (including extracting the JSON fragment and parsing it)
    2. On failure, fall back to regex extraction:
       - Double-quoted content after A1/A2/... as atomic_facts
       - Double-quoted content containing logic symbols (∧∨¬→↔) as symbolic_logic
    3. Duplicate keys keep the last one (override behavior)
    """
    # Preliminary cleanup: remove markdown code block markers
    raw_text = text
    clean_text = re.sub(r"```(json)?", "", text, flags=re.IGNORECASE)

    result = {
        "atomic_facts": {},
        "symbolic_logic": ""
    }

    # ===== Stage 1: try standard JSON parsing =====
    candidates = [raw_text.strip(), clean_text.strip()]

    json_candidate = extract_json_from_text(clean_text)
    if json_candidate:
        candidates.append(json_candidate)

    for candidate in candidates:
        try:
            data = json.loads(candidate)
            if isinstance(data, dict) and ("atomic_facts" in data or "symbolic_logic" in data):
                # Extract atomic_facts: keep only A\d+ keys and normalize to uppercase
                facts = data.get("atomic_facts", {})
                if isinstance(facts, dict):
                    for k, v in facts.items():
                        if re.match(r'^A\d+$', str(k), re.IGNORECASE):
                            result["atomic_facts"][str(k).upper()] = str(v)
                # Extract symbolic_logic
                if "symbolic_logic" in data:
                    result["symbolic_logic"] = str(data["symbolic_logic"])
                return result
        except (json.JSONDecodeError, Exception):
            continue

    # ===== Stage 2: regex extraction (only when JSON parsing fully fails) =====

    # 2.1 Extract atomic_facts
    # Matches: A1: "value" / "A1": "value" / A1 = "value" / "A1" : "value", etc.; handles escaped quotes \"
    atomic_pattern = re.compile(
        r'["\']?\s*(A\d+)\s*["\']?\s*[:=]\s*["\']((?:[^"\\]|\\.)*)["\']',
        re.IGNORECASE
    )

    for match in atomic_pattern.finditer(clean_text):
        key = match.group(1).upper()   # A1, A2...
        val = match.group(2)
        # Handle common escape characters
        val = val.replace('\\"', '"').replace('\\\\', '\\').replace("\\'", "'")
        result["atomic_facts"][key] = val   # Overwrite on duplicates; keep the last occurrence

    # 2.2 Extract symbolic_logic
    logic_symbols = r'[∧∨¬→↔]'

    # Preferred strategy: find the value of the symbolic_logic key
    sym_pattern = re.compile(
        r'["\']?\s*symbolic_logic\s*["\']?\s*[:=]\s*["\']((?:[^"\\]|\\.)*)["\']',
        re.IGNORECASE
    )

    sym_matches = list(sym_pattern.finditer(clean_text))
    if sym_matches:
        # Keep the last match
        val = sym_matches[-1].group(1)
        val = val.replace('\\"', '"').replace('\\\\', '\\').replace("\\'", "'")
        result["symbolic_logic"] = val
    else:
        # Fallback strategy: find any double/single-quoted string containing logic symbols
        # First record the span of atomic fact values, to avoid mistaking them for symbolic_logic
        atomic_ranges = []
        for m in atomic_pattern.finditer(clean_text):
            atomic_ranges.append((m.start(2), m.end(2)))

        # Find all quoted strings
        string_pattern = re.compile(r'["\']((?:[^"\\]|\\.)*)["\']')

        for match in string_pattern.finditer(clean_text):
            start, end = match.start(1), match.end(1)
            # Skip if this string belongs to an atomic fact value
            in_atomic = any(s <= start < e or s < end <= e for s, e in atomic_ranges)
            if in_atomic:
                continue

            content = match.group(1).replace('\\"', '"').replace('\\\\', '\\').replace("\\'", "'")
            if re.search(logic_symbols, content):
                # Overwrite in each iteration; keep the last match containing logic symbols
                result["symbolic_logic"] = content

    return result

def generate_atomic(llm: APILLM, user_fact: str) -> dict:
    """
    When atomic_facts are unavailable: call the LLM to generate atomic propositions + logical expression.
    :param llm: LLM instance
    :param user_fact: original natural language sentence
    :return: dictionary containing atomic_facts and symbolic_logic
    """
    prompt = Template(PROMPT_ATOMIC_FULL).substitute(user_fact=user_fact)
    try:
        resp = llm.generate(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            do_sample=True,
            temperature=0.01
        ).strip()
    except:
        # time.sleep(30)
        resp = llm.generate(
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            do_sample=True,
            temperature=0.01
        ).strip()
    # Use the robust parsing function instead of a plain try-except
    parsed_data = safe_json_load(resp)
    return parsed_data


def extract_brace_content(resp):
    left = resp.find("{")
    right = resp.rfind("}")
    if left == -1 or right == -1 or left >= right:
        return "None"
    content = resp[left+1:right].strip()
    # Iteratively strip outer nested braces
    while content.startswith("{") and content.endswith("}"):
        content = content[1:-1].strip()
    return content


def process_json_file(input_path: str, output_path: str, llm: APILLM):
    """
    Process items one by one, save incrementally, skip failed items, and support resuming from checkpoints.
    :param input_path: input JSON file path
    :param output_path: output JSON file path
    :param llm: LLM instance
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} not found")

    # Load the raw input data
    with open(input_path, "r", encoding="utf-8") as f:
        origin_data = json.load(f)  # Testing: [100:]

    # Initialize the output list: if the output file exists, load existing results first (resume)
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            print(f"Existing output file detected; loaded {len(output_data)} historical results. Resuming from checkpoint...")
        except Exception:
            print("Failed to parse the output file; starting a new output list")
            output_data = []
    else:
        output_data = []

    total_num = len(origin_data)
    # Resume from the number of processed items to avoid re-processing
    start_idx = len(output_data)

    for idx in range(start_idx, total_num):
        # if idx <= 296:
        #     continue
        item = origin_data[idx]
        print(f"\n===== Processing item {idx+1}/{total_num} =====")
        user_fact = item.get("claim", "")

        if not user_fact:
            print(f"[Warning] Claim at index {idx} is empty; skipped")
            output_data.append(item)
            # Save after each item
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            continue

        # Full generation
        full_res = generate_atomic(llm, user_fact)
        item["atomic_facts"] = full_res.get("atomic_facts", {})
        item["symbolic_logic"] = full_res.get("symbolic_logic", "")
        

        # Append to the result list and write to file immediately (incremental save)
        output_data.append({
            "id": item["id"],
            "claim": item["claim"],
            "atomic_facts": item["atomic_facts"],
            "symbolic_logic": item["symbolic_logic"],
            "claim_entities": item["claim_entities"]
        })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Item {idx+1} processed and saved")

        llm.save_token_stats()
        llm.reset_token_stats()
        llm.log_count = idx+1

    print(f"\nAll done! Processed {total_num} items in total. Results saved to: {output_path}")


if __name__ == "__main__":
    # Initialize the LLM
    model_name = "Qwen3-8B"
    llm = APILLM(log_path=f"./log/{model_name}/tokens_log")

    # Configuration: file paths (modify as needed)
    INPUT_JSON = "./FactLens_Entity.json"
    OUTPUT_JSON = f"./data/FactLens_{model_name}.json"

    # Run
    process_json_file(INPUT_JSON, OUTPUT_JSON, llm)