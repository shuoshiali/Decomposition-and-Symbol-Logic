import json
import os
import random
import re
from string import Template
import sys
from pathlib import Path

from utils.utils_DVLR_8B import APILLM

from tools.logic_verifier import (
    LogicValidator, SymbolicLogicParser, are_z3_equivalent,
    normalize_atomic_facts, build_teacher_index, load_json_robust,
)

from tools.query_dependency import get_dependency_parse_multi
HAS_DEPENDENCY_PARSER = True

from tools.workflow import (
    extract_linguistic_features,
    build_forward_system,
    build_forward_input
)

SYSTEM_MESSAGE = (
    "You are a logical reasoning assistant with deep linguistic expertise. "
    "Your task is to perform logical atomic decomposition of natural language claims. "
    "You MUST follow the three-stage format strictly: "
    "(1) Parse: analyze syntactic structure to identify scope boundaries, "
    "logical connectives (XOR ⊕, NOR, AND ∧, OR ∨), conditional markers (If/then →), "
    "argument roles (ARG1, ARG2), and restrictor clauses; "
    "(2) Decompose: extract minimal sufficient atomic facts; "
    "(3) Symbolize: generate the formal symbolic logic expression. "
    "Use the provided syntactic clues as structural prior knowledge to resolve ambiguities."
)

SYSTEM_MESSAGE_TREE = (
    "You are a logical reasoning assistant with deep linguistic expertise. "
    "Your task is to perform logical atomic decomposition of natural language claims. "
    "You MUST follow the three-stage format strictly: "
    "(1) Parse: analyze syntactic structure to identify scope boundaries, "
    "logical connectives (XOR ⊕, NOR, AND ∧, OR ∨), conditional markers (If/then →), "
    "argument roles (ARG1, ARG2), and restrictor clauses; "
    "(2) Decompose: extract minimal sufficient atomic facts; "
    "(3) Symbolize: generate the formal symbolic logic expression. "
    "Use the provided syntactic clues as structural prior knowledge to resolve ambiguities.\n\n"
    "You MUST output strictly in the following XML format and nothing else:\n"
    "<stage>Parse</stage>\n"
    "<analysis>\n...your parse reasoning...\n</analysis>\n\n"
    "<stage>Decompose</stage>\n"
    "<atomic>\n...one atomic fact per line...\n</atomic>\n\n"
    "<stage>Symbolize</stage>\n"
    "<symbol>\n...symbolic logic expression...\n</symbol>\n\n"
    "Do NOT include any additional stages (e.g., Verify). "
    "Do NOT add any text, markdown, or explanations outside these XML tags."
)

# ==================== Constants identical to workflow ====================
RANDOM_SEED = 12
random.seed(RANDOM_SEED)

FORMAT_TYPES = ["bullet", "compact", "json_like"]
MASK_NONE_RATIO = 0.20
MASK_LITE_RATIO = 0.30


# ==================== Dual-mode XML / free-text parser ====================
def parse_xml_completion(completion: str) -> dict:
    """Extract atomic_facts and symbolic_logic from the model output.
    
    Supports two formats:
    1. Standard XML (<atomic> + <symbol>) - consistent with SFT training outputs
    2. Free-text fallback ((1) Parse ... (2) Decomposition ... (3) Symbolization ...)
       - fault-tolerant parsing when the system prompt lacks the XML constraints
    """
    atomic_facts = {}
    symbolic_logic = ""

    if not completion or not isinstance(completion, str):
        return {"atomic_facts": {}, "symbolic_logic": ""}

    # ========== Strategy 1: standard XML format ==========
    symbol_match = re.search(
        r'<symbol[^>]*>\s*(.*?)\s*</symbol>',
        completion,
        re.DOTALL | re.IGNORECASE
    )
    atomic_match = re.search(
        r'<atomic[^>]*>\s*(.*?)\s*</atomic>',
        completion,
        re.DOTALL | re.IGNORECASE
    )

    if symbol_match and atomic_match:
        symbolic_logic = symbol_match.group(1).strip()
        symbolic_logic = re.sub(r'\s+', ' ', symbolic_logic)
        symbolic_logic = symbolic_logic.replace('&lt;', '<').replace('&gt;', '>').replace('&amp;', '&')

        atomic_block = atomic_match.group(1)
        for line in atomic_block.split('\n'):
            line = line.strip()
            if not line:
                continue
            line = re.sub(r'<[^>]+>', '', line).strip()
            m = re.match(r'^(?:-\s*)?(A\d+)\s*[:=.\s]\s*(.+)$', line)
            if not m:
                continue
            key, val = m.groups()
            val = val.strip()
            if val.startswith('∧') or val.startswith('∨') or val.startswith('¬') or val.startswith('('):
                continue
            if '→' in val or '↔' in val:
                continue
            val = re.sub(r'</\w+>\s*$', '', val)
            if val and key not in atomic_facts:
                atomic_facts[key] = val

        return {
            "atomic_facts": atomic_facts,
            "symbolic_logic": symbolic_logic
        }

    # ========== Strategy 2: free-text fallback ==========
    # Typical pattern: (1) Parse: ...  (2) Decomposition: ...  (3) Symbolization: A1 AND A2
    lines = completion.split('\n')

    # 2.1 Extract the symbolic logic (search from the end; it is usually last)
    for line in reversed(lines):
        line_stripped = line.strip()
        if not line_stripped:
            continue
        # Strip common prefixes: (3) Symbolization: / Symbolic Logic: / (3) Symbolize:
        clean = re.sub(r'^\(\d+\)\s*\w+[:：]\s*', '', line_stripped)
        clean = re.sub(r'^[A-Za-z\s]+[:：]\s*', '', clean)
        clean = clean.strip()
        # Check whether it is a logic expression: contains A\d+ and at least one connective
        if re.search(r'\bA\d+\b', clean) and any(c in clean for c in '∧∨¬→↔'):
            symbolic_logic = clean
            break

    # 2.2 Extract the atomic facts
    in_atomic_section = False
    for line in lines:
        line_stripped = line.strip()
        if not line_stripped:
            continue
        
        # Detect the atomic-section boundary
        if re.search(r'^\(\d+\)\s*decompos', line_stripped, re.IGNORECASE) or \
           re.search(r'atomic\s*fact', line_stripped, re.IGNORECASE):
            in_atomic_section = True
            continue
        
        # Once the symbolization section starts, stop extracting atomic facts
        if re.search(r'symboliz|symbolic', line_stripped, re.IGNORECASE) and \
           any(c in line_stripped for c in '∧∨¬→↔'):
            break
        
        # Match an 'A1: xxx' line (the '- A1: xxx' variant is also supported)
        m = re.match(r'^(?:-\s*)?(A\d+)\s*[:=.\s]\s*(.+)$', line_stripped)
        if m:
            key, val = m.groups()
            val = val.strip()
            # Skip logic-expression lines
            if val.startswith(('∧', '∨', '¬', '(')):
                continue
            if '→' in val or '↔' in val:
                continue
            val = re.sub(r'</\w+>\s*$', '', val)
            if val and key not in atomic_facts:
                atomic_facts[key] = val

    return {
        "atomic_facts": atomic_facts,
        "symbolic_logic": re.sub(r'\s+', ' ', symbolic_logic) if symbolic_logic else ""
    }


# ==================== Validator post-processing (unchanged) ====================
class DiReRLPostProcessor:
    def __init__(self, teacher_index=None):
        self.validator = LogicValidator(alpha=0.90)
        self.teachers = teacher_index or {}

    def validate_and_fix(self, claim: str, atomic: dict, symbolic: str) -> dict:
        if not atomic or not symbolic:
            return {"atomic_facts": atomic, "symbolic_logic": symbolic, "valid": False}

        try:
            SymbolicLogicParser(atomic).parse(symbolic)
        except Exception:
            return {"atomic_facts": atomic, "symbolic_logic": symbolic, "valid": False}

        refs = set(re.findall(r"\bA\d+\b", symbolic))
        if not refs or not refs.issubset(set(atomic.keys())):
            return {"atomic_facts": atomic, "symbolic_logic": symbolic, "valid": False}

        simplicity, reason = self.validator._detect_meaningless_complexity(claim, symbolic, atomic)
        if simplicity < 0.1:
            fixed = " ∧ ".join(sorted(atomic.keys()))
            return {
                "atomic_facts": atomic,
                "symbolic_logic": fixed,
                "valid": True,
                "fixed": True,
                "fix_reason": reason
            }

        if '→' in symbolic:
            parts = symbolic.split('→', 1)
            if len(parts) == 2:
                lhs, rhs = parts[0].strip(), parts[1].strip()
                try:
                    expr_lhs = SymbolicLogicParser(atomic).parse(lhs)
                    expr_rhs = SymbolicLogicParser(atomic).parse(rhs)
                    if are_z3_equivalent(expr_lhs, expr_rhs):
                        fixed = " ∧ ".join(sorted(atomic.keys()))
                        return {
                            "atomic_facts": atomic,
                            "symbolic_logic": fixed,
                            "valid": True,
                            "fixed": True,
                            "fix_reason": "self_implication"
                        }
                except Exception:
                    pass

        logicality = self.validator._evaluate_logicality(claim, atomic, symbolic)

        return {
            "atomic_facts": atomic,
            "symbolic_logic": symbolic,
            "valid": True,
            "fixed": False,
            "simplicity": simplicity,
            "logicality": logicality
        }


# ==================== Main generation function (fixed system prompt) ====================
def generate_atomic(
    llm: APILLM,
    user_fact: str,
    post_processor: DiReRLPostProcessor = None,
    dependency_tree: str = "",
    use_postprocess: bool = True,
    dataset: str = ""
) -> dict:
    """Strictly mirror the prompt-building logic of the fixed workflow.
    
    With tree: system = build_forward_system(...) (full XML constraints + clues)
    Without tree: system = SYSTEM_MESSAGE
    """
    if dependency_tree and dependency_tree.strip():
        # ========== With tree ==========
        features = extract_linguistic_features(dependency_tree)
        
        # Fixed config at inference time (no randomness)
        mask_type = "full"   # use full clues
        fmt = "bullet"       # use the structured-list format, the clearest

        system_prompt = build_forward_system(features, mask_type, fmt)
    else:
        # ========== Without tree ==========
        system_prompt = SYSTEM_MESSAGE

    if "ProntoQA" in dataset:
        # system_prompt += f"\n[Example: For \"Jompuses are dumpuses.\" atomic decompose into \"It is a Jompus.\" and \"It is a Jompus.\".]"
        system_prompt += f"\n[Example: For \"Each/Every Jompus is dumpus.\" atomic decompose into \"Jompus is dumpus.\".]"
    elif "ProofWriter" in dataset:
        # if "This claim expresses a SINGLE atomic proposition." in system_prompt:
        #     system_prompt = system_prompt.replace("This claim expresses a SINGLE atomic proposition.", "")
        system_prompt += f"\n[Example: For \"Green things are popular.\" atomic decompose into \"Things are green.\" and \"Things are popular.\".]"
    elif "FOLIO" in dataset:
        # system_prompt += f"\n[Example: For \"Mike eats apples and bananas.\" atomic decompose into \"Mike eats apples.\" and \"Mike eats bananas.\" (Only two entities (the subject and the object) in each atom).]"
        system_prompt += f"\n[Note: Exclusive OR (choosing one out of two, such as \"either... or\"), use ⊕.]"  # ; For \"All (A1) are (A2).\", use → 
        # Mike is either eating apples and bananas, or sleeping.
        # system_prompt += f"\n[Example: \"All people who are inactive members chaperone dances.\"), \"A1\": \"People are inactive members\", \"A2\": \"People chaperone dances\", \"A1 → A2\".]"
        system_prompt = system_prompt
    elif "Multi-LogiEval" in dataset:
        system_prompt += f"\n[Example: For \"Logically, if Liam were ill, he should have rested.\" atomic decompose into \"Liam were ill.\" and \"He should have rested.\".]"
    elif "FLD" in dataset:
        system_prompt += f"\n[Example: For \"The fact that proceedings but not cannibalism occurs does not stand.\" atomic decompose into \"Proceedings occurs.\" and \"Cannibalism occurs.\".]"
        
    # User content identical to training
    user_content = build_forward_input(user_fact)

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_content},
    ]

    # Diagnostics: uncomment on first run to confirm the system-prompt tail contains the XML constraints
    # print(f"\n[DEBUG] System prompt tail:\n...{system_prompt[-250:]}")
    # print(f"[DEBUG] User:\n{user_content}")

    resp = llm.generate(
        messages=messages,
        do_sample=True,
        temperature=0.01
    ).strip()

    parsed = parse_xml_completion(resp)

    if post_processor and use_postprocess:
        result = post_processor.validate_and_fix(
            user_fact, parsed["atomic_facts"], parsed["symbolic_logic"]
        )
        return result

    return {
        "atomic_facts": parsed["atomic_facts"],
        "symbolic_logic": parsed["symbolic_logic"],
        "valid": bool(parsed["atomic_facts"] and parsed["symbolic_logic"]),
        "raw_completion": resp
    }


# ==================== Batch processing (unchanged) ====================
def process_json_file(
    input_path: str,
    output_path: str,
    llm: APILLM,
    teacher_file: str = None,
    use_dependency_tree: bool = False,
    nlp = None
):
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File not found: {input_path}")

    post_processor = None
    if teacher_file and os.path.exists(teacher_file):
        teacher_index = build_teacher_index(teacher_file)
        post_processor = DiReRLPostProcessor(teacher_index)
        print(f"[DiReRL] Loaded teacher index: {len(teacher_index)} claims")
    else:
        post_processor = DiReRLPostProcessor()

    with open(input_path, "r", encoding="utf-8") as f:
        origin_data = json.load(f)

    if os.path.exists(output_path):
        try:
            with open(output_path, "r", encoding="utf-8") as f:
                output_data = json.load(f)
            print(f"Existing output file detected; loaded {len(output_data)} historical results, continuing...")
        except Exception:
            output_data = []
    else:
        output_data = []

    total_num = len(origin_data)  # total_num = 1  
    start_idx = len(output_data)

    for idx in range(start_idx, total_num):
        item = origin_data[idx]
        print(f"\n===== Processing item {idx+1}/{total_num} =====")

        user_fact = []
        if "ProntoQA" in input_path:
            user_fact = item["context"].split(". ")
            user_fact = [s + "." if i < len(user_fact) - 1 else s for i, s in enumerate(user_fact)]
            user_fact.append(item["question"])
        elif "ProofWriter" in input_path:
            user_fact = item["context"].split(". ")
            user_fact = [s + "." if i < len(user_fact) - 1 else s for i, s in enumerate(user_fact)]
            user_fact.append(item["question"].replace("Based on the above information, is the following statement true, false, or unknown? ",""))
        elif "FOLIO" in input_path:
            user_fact = item["context"].split(". ")
            user_fact = [s + "." if i < len(user_fact) - 1 else s for i, s in enumerate(user_fact)]
            user_fact.append(item["question"])
        elif "Multi-LogiEval" in input_path:
            user_fact = item["context"].split(". ")
            user_fact = [s + "." if i < len(user_fact) - 1 else s for i, s in enumerate(user_fact)]
            user_fact.append(item["question"].replace("?","."))
        elif "FLD" in input_path:
            user_fact = item["context"].split(". ")
            user_fact = [s + "." if i < len(user_fact) - 1 else s for i, s in enumerate(user_fact)]
            user_fact.append(item["question"])
        else:
            user_fact.append(item.get("claim", ""))

        if not user_fact:
            print(f"[Warning] Item {idx} has an empty claim; skipped")
            output_data.append(item)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            continue

        item["decomposition"] = []
        count = 0
        for fact in user_fact:
            print(f"\n===== Processing item {idx+1}, part {count+1}/{len(user_fact)} =====")
            count +=1
            dependency_tree = ""
            if use_dependency_tree:
                dependency_tree = item.get("dependency_tree", "") or item.get("tree", "")
                if not dependency_tree and nlp is not None and HAS_DEPENDENCY_PARSER:
                    try:
                        dependency_tree = get_dependency_parse_multi(fact, nlp, mode='frame')
                        print(f"[DiReRL] Dependency tree generated on the fly, length: {len(dependency_tree)} chars")
                    except Exception as e:
                        print(f"[Warning] Dependency parsing failed for item {idx}: {e}; falling back to no-tree mode")
                        dependency_tree = ""
                elif not dependency_tree and nlp is None:
                    print(f"[Warning] Item {idx} has no dependency_tree field and no nlp provided; falling back to no-tree mode")

            full_res = generate_atomic(llm, fact, post_processor, dependency_tree, use_postprocess=True, dataset=input_path)
            
            item["decomposition"].append({
                "claim": fact,
                "atomic_facts": full_res["atomic_facts"],
                "symbolic_logic": full_res["symbolic_logic"],
                "valid": full_res.get("valid", False),
                "fixed": full_res.get("fixed", False),
                "fix_reason": full_res.get("fix_reason", ""),
                "dependency_tree": dependency_tree
            })

            # print(f"  Atomic keys: {list(full_res['atomic_facts'].keys())}")
            # print(f"  Symbolic: {full_res['symbolic_logic']}")
            # print(f"  Valid: {full_res.get('valid', False)}", end="")
            # if full_res.get("fixed"):
            #     print(f" | Fixed: {full_res['fix_reason']}", end="")
            # if dependency_tree:
            #     print(f" | Tree: {'used' if dependency_tree else 'none'}", end="")
            # print()

        output_data.append({
            "id": item.get("id", idx),
            "claims": user_fact,
            "decomposition": item["decomposition"]
        })
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"Item {idx+1} processed and saved")

        if hasattr(llm, 'save_token_stats'):
            llm.save_token_stats()
            llm.reset_token_stats()
            llm.log_count = idx+1

    print(f"\nAll done! Processed {total_num} items; results saved to: {output_path}")


if __name__ == "__main__":
    DATASET_NAME = "ProofWriter"

    llm = APILLM(log_path=f"./log/{DATASET_NAME}_DVLR_8B/tokens_log")

    INPUT_JSON = f"./datasets/{DATASET_NAME}.json"
    OUTPUT_JSON = f"./atomic_results/{DATASET_NAME}_Atomic_8B.json"
    TEACHER_FILE = "./FOLIO_ALL_DepSampling.json"

    USE_TREE = True
    
    nlp = None
    if USE_TREE:
        from stanfordcorenlp import StanfordCoreNLP
        nlp = StanfordCoreNLP(
            r'./stanfordnlp/stanfordcorenlp/stanford-corenlp-full-2018-01-31',
            lang='en'
        )
        print("[DiReRL] Stanford CoreNLP initialized (on-the-fly generation mode)")

    process_json_file(
        INPUT_JSON,
        OUTPUT_JSON,
        llm,
        teacher_file=TEACHER_FILE,
        use_dependency_tree=USE_TREE,
        nlp=nlp
    )

    if nlp is not None:
        nlp.close()
        print("[DiReRL] Stanford CoreNLP closed")