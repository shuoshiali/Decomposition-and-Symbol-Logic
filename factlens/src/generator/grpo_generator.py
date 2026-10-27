import json
import os
import re
from string import Template
import time

import sys
from pathlib import Path

from utils_Qwen_8B_SFT_GRPO import APILLM

from logic_verifier import (
    LogicValidator, SymbolicLogicParser, are_z3_equivalent,
    normalize_atomic_facts, build_teacher_index, load_json_robust,
)
from reward_grpo import LogicReward


# ==================== Prompt (consistent with DiReRL SFT/GRPO training format) ====================
PROMPT_ATOMIC_FULL = '''You are a logical reasoning assistant with deep linguistic expertise. Your task is to perform logical atomic decomposition of natural language claims.

Extract minimal sufficient atomic facts and generate the symbolic logic expression.
Output ONLY these two XML tags, no explanations, no analysis, no other stages:

<atomic>
A1: [first atomic fact]
A2: [second atomic fact]
...
</atomic>
<symbol>[symbolic logic using A1, A2, ... and connectives ∧∨¬→↔]</symbol>

CRITICAL RULES for symbolic_logic:
1. If the claim simply states multiple facts (uses "and", "while", commas to list facts), use ONLY the ∧ connective: "A1 ∧ A2 ∧ A3". Do NOT use →, ↔, ∨, or ¬.
2. Use → ONLY if the original claim contains explicit conditional words: "if", "when", "unless", "provided that", "implies".
3. Use ¬ ONLY if the original claim contains explicit negation: "not", "never", "no", "without".
4. Use ∨ ONLY if the original claim contains explicit alternatives: "or", "either...or".
5. NEVER construct tautologies like "(A ∧ B) → (A ∧ B)" or "((A)∨(¬A)) → B".
6. NEVER add unnecessary parentheses or nested structures for simple claims.
7. Each atomic statement must be a declarative clause without "and", "or", "if...then", "not", or similar logical words inside.
8. Keep the original tense and phrasing as much as possible; do not add or omit key meaning.
9. Output only the two XML tags, no extra text.

Now process the following input:
Claim: "$user_fact"
'''


# ==================== XML parsing (key fix: extract symbol first, then atomic) ====================
def parse_xml_completion(completion: str) -> dict:
    """Extract atomic_facts and symbolic_logic from the XML format.

    Fix strategy:
    1. Extract the <symbol> block first and remove it, so that A1 references inside the symbolic
       expression are not mistakenly matched as atomic facts
    2. Extract A1:xxx pairs from the <atomic> block
    3. If the number of atoms in <atomic> is less than the number of references in the symbolic
       expression, backfill from the full text
    """
    atomic_facts = {}
    symbolic_logic = ""

    # ---------- Step 1: extract the <symbol> block first ----------
    symbol_match = re.search(
        r'<symbol[^>]*>(.*?)(?:</symbol>|$)',
        completion,
        re.DOTALL | re.IGNORECASE
    )
    if symbol_match:
        symbolic_logic = symbol_match.group(1).strip()
        symbolic_logic = re.sub(r'\s+', ' ', symbolic_logic)
        symbolic_logic = symbolic_logic.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

    # ---------- Step 2: remove the <symbol> block to avoid contamination ----------
    body = completion
    if symbol_match:
        body = body.replace(symbol_match.group(0), '')

    # ---------- Step 3: try extracting from the <atomic> block ----------
    atomic_match = re.search(
        r'<atomic[^>]*>(.*?)(?:</atomic>|$)',
        body,
        re.DOTALL | re.IGNORECASE
    )
    atomic_block = atomic_match.group(1) if atomic_match else body

    def _extract_atomic_from_text(text: str, existing: dict) -> dict:
        """Extract A1:xxx patterns from text, skipping keys that already exist."""
        result = dict(existing)
        for line in text.split('\n'):
            line = line.strip()
            if not line:
                continue
            # Strip residual XML tags
            line = re.sub(r'<[^>]+>', '', line).strip()
            # Match A1: xxx (anchored at line start)
            m = re.match(r'^(A\d+)\s*[:=.\s]\s*(.+)$', line)
            if not m:
                continue
            key, val = m.groups()
            if key in result:
                continue  # already exists; skip
            val = val.strip()
            # Filter symbolic fragments (starting with a logical connective or parenthesis)
            if val.startswith('∧') or val.startswith('∨') or val.startswith('¬') or val.startswith('('):
                continue
            # Filter values containing →/↔ (these belong to the symbolic expression)
            if '→' in val or '↔' in val:
                continue
            # Strip trailing XML tags
            val = re.sub(r'</\w+>\s*$', '', val)
            if val:
                result[key] = val
        return result

    # Extract from the <atomic> block
    atomic_facts = _extract_atomic_from_text(atomic_block, {})

    # ---------- Step 4: backfill from the full text if atomic is incomplete ----------
    # Determine which atoms are referenced by the symbolic expression
    refs = set(re.findall(r'\bA\d+\b', symbolic_logic))
    if len(atomic_facts) < len(refs):
        # Backfill from the full body (with the symbol block already removed)
        atomic_facts = _extract_atomic_from_text(body, atomic_facts)

    return {
        "atomic_facts": atomic_facts,
        "symbolic_logic": symbolic_logic
    }


# ============ Change 2: simplified post-processor (basic validation only, no canonicality fix) ============
class BasePostProcessor:
    """Basic GRPO post-processing: format validation only, no canonicality fix."""

    def __init__(self, teacher_index=None):
        self.validator = LogicValidator(alpha=0.90)
        self.teachers = teacher_index or {}

    def validate(self, claim: str, atomic: dict, symbolic: str) -> dict:
        """Basic validation: hard gates + z3 equivalence (0/1), no fixes."""
        if not atomic or not symbolic:
            return {"atomic_facts": atomic, "symbolic_logic": symbolic, "valid": False}

        # 1. Parseable
        try:
            SymbolicLogicParser(atomic).parse(symbolic)
        except Exception:
            return {"atomic_facts": atomic, "symbolic_logic": symbolic, "valid": False}

        # 2. Consistent atom references
        refs = set(re.findall(r"\bA\d+\b", symbolic))
        if not refs or not refs.issubset(set(atomic.keys())):
            return {"atomic_facts": atomic, "symbolic_logic": symbolic, "valid": False}

        # 3. z3 equivalence (if a teacher is available)
        teacher = self.teachers.get(claim)
        z3_equiv = False
        if teacher:
            try:
                mapping = self.validator._align_atomic_facts(teacher.atomic_facts, atomic)
                if mapping:
                    mapped = self.validator._map_symbolic_variables(symbolic, mapping)
                    expr_t = SymbolicLogicParser(teacher.atomic_facts).parse(teacher.symbolic_logic)
                    expr_s = SymbolicLogicParser(teacher.atomic_facts).parse(mapped)
                    z3_equiv = are_z3_equivalent(expr_t, expr_s)
            except Exception:
                pass

        return {
            "atomic_facts": atomic,
            "symbolic_logic": symbolic,
            "valid": True,
            "z3_equiv": z3_equiv
        }


# ============ Change 3: use BasePostProcessor in generate_atomic ============
def generate_atomic(llm, user_fact: str, post_processor: BasePostProcessor = None) -> dict:
    prompt = Template(PROMPT_ATOMIC_FULL).substitute(user_fact=user_fact)
    resp = llm.generate(
        messages=[{"role": "user", "content": prompt}],
        do_sample=True,
        temperature=0.7
    ).strip()

    parsed = parse_xml_completion(resp)

    if post_processor:
        result = post_processor.validate(user_fact, parsed["atomic_facts"], parsed["symbolic_logic"])
        return result

    return {
        "atomic_facts": parsed["atomic_facts"],
        "symbolic_logic": parsed["symbolic_logic"],
        "valid": bool(parsed["atomic_facts"] and parsed["symbolic_logic"])
    }


# ==================== Batch processing ====================
# ============ Change 4: initialize BasePostProcessor in process_json_file ============
def process_json_file(input_path: str, output_path: str, llm, teacher_file: str = None):
    post_processor = None
    if teacher_file and os.path.exists(teacher_file):
        teacher_index = build_teacher_index(teacher_file)
        post_processor = BasePostProcessor(teacher_index)
        print(f"[BaseGRPO] Loaded teacher index: {len(teacher_index)} claims")
    else:
        post_processor = BasePostProcessor()

    # Load the raw input data
    with open(input_path, "r", encoding="utf-8") as f:
        origin_data = json.load(f)

    # Resume from checkpoint
    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            print(f"Existing output file detected; loaded {len(output_data)} historical results. Resuming from checkpoint...")
        except Exception:
            output_data = []
    else:
        output_data = []

    total_num = len(origin_data)
    start_idx = len(output_data)

    for idx in range(start_idx, total_num):
        item = origin_data[idx]
        print(f"\n===== Processing item {idx+1}/{total_num} =====")
        user_fact = item.get("claim", "")

        if not user_fact:
            print(f"[Warning] Claim at index {idx} is empty; skipped")
            output_data.append(item)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            continue

        # Generation + post-processing
        full_res = generate_atomic(llm, user_fact, post_processor)
        
        item["atomic_facts"] = full_res["atomic_facts"]
        item["symbolic_logic"] = full_res["symbolic_logic"]
        item["valid"] = full_res.get("valid", False)
        item["fixed"] = full_res.get("fixed", False)
        if full_res.get("fix_reason"):
            item["fix_reason"] = full_res["fix_reason"]

        # Diagnostic output
        print(f"  Atomic keys: {list(full_res['atomic_facts'].keys())}")
        print(f"  Symbolic: {full_res['symbolic_logic']}")
        print(f"  Valid: {full_res.get('valid', False)}", end="")
        if full_res.get("fixed"):
            print(f" | Fixed: {full_res['fix_reason']}", end="")
        print()

        # Save
        output_data.append({
            "id": item.get("id", idx),
            "claim": item["claim"],
            "atomic_facts": item["atomic_facts"],
            "symbolic_logic": item["symbolic_logic"],
            "valid": item["valid"],
            "fixed": item.get("fixed", False),
            "fix_reason": item.get("fix_reason", ""),
            "claim_entities": item.get("claim_entities", {})
        })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Item {idx+1} processed and saved")

        if hasattr(llm, 'save_token_stats'):
            llm.save_token_stats()
            llm.reset_token_stats()
            llm.log_count = idx + 1

    print(f"\nAll done! Processed {total_num} items in total. Results saved to: {output_path}")


if __name__ == "__main__":
    # Initialize the LLM (pointing to the DiReRL checkpoint)
    model_name = "Qwen3-8B-SFT_GRPO"
    llm = APILLM(log_path=f"./log/{model_name}/tokens_log")

    # Configuration: file paths
    INPUT_JSON = "./FactLens_Entity.json"
    OUTPUT_JSON = f"./data/FactLens_{model_name}.json"
    TEACHER_FILE = "./FOLIO_ALL_Sampling_Qwen.json"

    # Run
    process_json_file(INPUT_JSON, OUTPUT_JSON, llm, teacher_file=TEACHER_FILE)