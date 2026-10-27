import json
import random
import re
from typing import List, Dict, Any
from pathlib import Path


# ==================== Configuration Constants ====================
RANDOM_SEED = 12
REVERSE_TASK_RATIO = 0.10   # Plan 7: mix in 10% reverse auxiliary tasks
MASK_NONE_RATIO = 0.20      # Plan 3: 20% no clues (pure SFT)
MASK_LITE_RATIO = 0.30      # Plan 3: 30% simplified clues
MASK_FULL_RATIO = 0.50      # Plan 3: 50% full clues
FORMAT_TYPES = ["bullet", "compact", "json_like"]  # Plan 5: three clue formats


def set_seed(seed: int = RANDOM_SEED):
    random.seed(seed)


# ==================== Plan 1: Dependency Tree Feature Extraction ====================

def extract_linguistic_features(dep_tree: str) -> Dict[str, Any]:
    """
    Extract structured linguistic features from the raw dependency tree text.
    All visual noise such as indentation and brackets is discarded;
    only logically relevant syntactic markers are retained.
    """
    features = {
        "logical_connectives": [],
        "conditionals": [],
        "arguments": [],
        "restrictions": [],
        "variables": []
    }

    for line in dep_tree.split('\n'):
        line = line.strip()
        if not line or '→' not in line:
            continue

        # Conditional markers
        if '→ COND:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ COND:\s*(.+)', line)
            if m:
                features["conditionals"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "logic": m.group(4).strip()
                })
        # XOR
        elif '→ XOR:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ XOR:\s*(.+)', line)
            if m:
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "XOR", "logic": m.group(4).strip()
                })
        # Disjunction
        elif '→ DISJ:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ DISJ:\s*(.+)', line)
            if m:
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "DISJ", "logic": m.group(4).strip()
                })
        # Conjunction
        elif '→ CONJ:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ CONJ:\s*(.+)', line)
            if m:
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "CONJ", "logic": m.group(4).strip()
                })
        # NOR
        elif '→ NOR' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "NOR", "logic": "¬∨"
                })
        # ARG1
        elif '→ ARG1' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["arguments"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "role": "ARG1"
                })
        # ARG2
        elif '→ ARG2' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["arguments"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "role": "ARG2"
                })
        # Restrictor clauses
        elif '→ RESTRICT' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["restrictions"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3)
                })
        # Variables
        elif '→ VAR:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ VAR:\s*(.+)', line)
            if m:
                features["variables"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "var": m.group(4).strip()
                })

    return features


# ==================== Plan 5: Render Clues in Three Formats ====================

def _format_bullet(features: Dict, lite: bool = False) -> str:
    """Format A: structured list (human-readable, hierarchical)"""
    sections = []

    if features["logical_connectives"]:
        lines = ["Logical Connectives (scope markers):"]
        for c in features["logical_connectives"]:
            lines.append(f"  • {c['token']}[{c['idx']}] ({c['relation']}): {c['type']} → {c['logic']}")
        sections.append("\n".join(lines))

    if features["conditionals"]:
        lines = ["Conditional Markers (P→Q boundaries):"]
        for c in features["conditionals"]:
            lines.append(f"  • {c['token']}[{c['idx']}] ({c['relation']}): {c['logic']}")
        sections.append("\n".join(lines))

    if not lite:
        if features["arguments"]:
            lines = ["Core Arguments (entity tracking):"]
            for a in features["arguments"]:
                lines.append(f"  • {a['token']}[{a['idx']}] ({a['relation']}): {a['role']}")
            sections.append("\n".join(lines))

        if features["restrictions"]:
            lines = ["Restrictor Clauses (variable binding):"]
            for r in features["restrictions"]:
                lines.append(f"  • {r['token']}[{r['idx']}] ({r['relation']}): domain restrictor")
            sections.append("\n".join(lines))

        if features["variables"]:
            lines = ["Bound Variables (anaphora):"]
            for v in features["variables"]:
                lines.append(f"  • {v['token']}[{v['idx']}] ({v['relation']}): bound to {v['var']}")
            sections.append("\n".join(lines))

    return "\n\n".join(sections) if sections else "(No linguistic features)"


def _format_compact(features: Dict, lite: bool = False) -> str:
    """Format B: compact lines (key-value style, token-efficient)"""
    clues = []

    for c in features["logical_connectives"]:
        clues.append(f"{c['token']}[{c['idx']}]:{c['type']}({c['logic']})")
    for c in features["conditionals"]:
        clues.append(f"{c['token']}[{c['idx']}]:COND({c['logic']})")

    if not lite:
        for a in features["arguments"]:
            clues.append(f"{a['token']}[{a['idx']}]:{a['role']}")
        for r in features["restrictions"]:
            clues.append(f"{r['token']}[{r['idx']}]:RESTRICT")
        for v in features["variables"]:
            clues.append(f"{v['token']}[{v['idx']}]:VAR({v['var']})")

    return "CLUES: " + "; ".join(clues) if clues else "CLUES: none"


def _format_json_like(features: Dict, lite: bool = False) -> str:
    """Format C: pseudo-JSON (structured but without indentation, to prevent the model from memorizing layout)"""
    obj = {}

    if features["logical_connectives"]:
        obj["conn"] = [
            {"t": c["token"], "i": int(c["idx"]), "r": c["relation"], "type": c["type"], "logic": c["logic"]}
            for c in features["logical_connectives"]
        ]
    if features["conditionals"]:
        obj["cond"] = [
            {"t": c["token"], "i": int(c["idx"]), "r": c["relation"], "logic": c["logic"]}
            for c in features["conditionals"]
        ]

    if not lite:
        if features["arguments"]:
            obj["args"] = [
                {"t": a["token"], "i": int(a["idx"]), "r": a["relation"], "role": a["role"]}
                for a in features["arguments"]
            ]
        if features["restrictions"]:
            obj["restr"] = [
                {"t": r["token"], "i": int(r["idx"]), "r": r["relation"]}
                for r in features["restrictions"]
            ]
        if features["variables"]:
            obj["vars"] = [
                {"t": v["token"], "i": int(v["idx"]), "r": v["relation"], "bind": v["var"]}
                for v in features["variables"]
            ]

    return json.dumps(obj, ensure_ascii=False) if obj else "{}"


def format_linguistic_features(features: Dict, fmt: str, lite: bool = False) -> str:
    """Unified entry point: render linguistic clues in the specified format"""
    if fmt == "bullet":
        return _format_bullet(features, lite)
    elif fmt == "compact":
        return _format_compact(features, lite)
    elif fmt == "json_like":
        return _format_json_like(features, lite)
    else:
        return _format_bullet(features, lite)


# ==================== System Prompt Construction (Plan 2) ====================

FORWARD_SYSTEM_BASE = (
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

REVERSE_SYSTEM = (
    "You are a linguistic-logic alignment analyst. "
    "Your task is to identify the syntactic sources of every logical operator in a given "
    "symbolic expression, and explain the scope resolution rationale based on dependency structure.\n\n"
    "You MUST output strictly in the following XML format and nothing else:\n"
    "<stage>Operator Mapping</stage>\n"
    "<mapping>\n...operator-to-syntax mapping...\n</mapping>\n\n"
    "<stage>Reference Atoms</stage>\n"
    "<atomic>\n...atomic facts...\n</atomic>\n\n"
    "Do NOT add any text, markdown, or explanations outside these XML tags."
)


def build_forward_system(features: Dict, mask_type: str, fmt: str) -> str:
    """
    Plan 2: embed linguistic clues into the System Prompt.
    mask_type: none | lite | full
    """
    if mask_type == "none":
        return FORWARD_SYSTEM_BASE

    clues = format_linguistic_features(features, fmt, lite=(mask_type == "lite"))

    return (
        f"{FORWARD_SYSTEM_BASE}\n\n"
        f"[Syntactic Clues for This Claim]\n{clues}\n\n"
        f"Use these clues to determine operator scope and argument binding accurately."
    )


def build_forward_input(claim: str) -> str:
    """
    Plan 2: the input keeps only the Claim; the task description is no
    longer repeated here because it already lives in the system instruction.
    """
    return f"Claim: {claim}"


def build_forward_output(atomic_facts: Dict[str, str], symbolic_logic: str,
                         features: Dict, claim: str) -> str:
    """Forward task output: Parse → Decompose → Symbolize (strictly three stages, no Verify)"""
    atomic_lines = [f"{k}: {v}" for k, v in sorted(atomic_facts.items())]
    atomic_block = "\n".join(atomic_lines)

    # Parse reasoning (derived from the extracted features, not the raw tree)
    reasoning = []

    if features["conditionals"]:
        toks = [f"{c['token']}[{c['idx']}]" for c in features["conditionals"]]
        reasoning.append(
            f"1. Conditional Frame: Introduced by {'/'.join(toks)}. "
            f"Structure: P → Q, antecedent precedes consequent."
        )

    xor_items = [c for c in features["logical_connectives"] if c["type"] == "XOR"]
    nor_items = [c for c in features["logical_connectives"] if c["type"] == "NOR"]
    disj_items = [c for c in features["logical_connectives"] if c["type"] == "DISJ"]
    conj_items = [c for c in features["logical_connectives"] if c["type"] == "CONJ"]

    if xor_items:
        toks = [f"{x['token']}[{x['idx']}]" for x in xor_items]
        reasoning.append(
            f"2. Exclusive Disjunction: Correlative markers {'/'.join(toks)} indicate XOR (⊕)."
        )
    if nor_items:
        toks = [f"{n['token']}[{n['idx']}]" for n in nor_items]
        reasoning.append(
            f"2. Negative Disjunction: {'/'.join(toks)} indicates NOR (¬∨)."
        )
    if disj_items:
        toks = [f"{d['token']}[{d['idx']}]" for d in disj_items]
        reasoning.append(
            f"3. Inclusive Disjunction: {'/'.join(toks)} maps to (∨)."
        )
    if conj_items:
        toks = [f"{c['token']}[{c['idx']}]" for c in conj_items]
        reasoning.append(
            f"3. Conjunction: {'/'.join(toks)} maps to (∧)."
        )

    if features["restrictions"] and features["variables"]:
        reasoning.append(
            "4. Relative Clause Binding: Restrictors (acl:relcl) with bound variables "
            "resolve anaphora to the subject domain."
        )

    if features["arguments"]:
        arg1s = [a for a in features["arguments"] if a["role"] == "ARG1"]
        if arg1s:
            reasoning.append(
                f"5. Subject Tracking: ARG1 anchored to {arg1s[0]['token']}[{arg1s[0]['idx']}]."
            )

    parse_text = "\n".join(reasoning) if reasoning else "Complex nested structure with multiple operators."

    return (
        "<stage>Parse</stage>\n"
        "<analysis>\n" + parse_text + "\n</analysis>\n\n"
        "<stage>Decompose</stage>\n"
        "<atomic>\n" + atomic_block + "\n</atomic>\n\n"
        "<stage>Symbolize</stage>\n"
        f"<symbol>{symbolic_logic}</symbol>"
    )


# ==================== Plan 7: Reverse Auxiliary Task (10% mixed in) ====================

def build_reverse_record(item: Dict, features: Dict, fmt: str) -> Dict:
    """
    Reverse task: given the Claim + Symbolic Logic + tree clues, identify the
    syntactic source of every logical operator. This forces the model to build
    a bidirectional mapping and strengthens its understanding of the syntax-logic relation.
    """
    claim = item["claim"]
    symbolic_logic = item["symbolic_logic"]
    atomic_facts = item["atomic_facts"]

    # Reverse tasks always use full clues (the task itself is to analyze the mapping)
    clues = format_linguistic_features(features, fmt, lite=False)

    input_text = (
        f"Claim: {claim}\n\n"
        f"Symbolic Logic: {symbolic_logic}\n\n"
        f"[Syntactic Clues]\n{clues}\n\n"
        f"Task: For every logical operator in the symbolic expression, identify its "
        f"syntactic source(s) in the dependency structure and explain the scope resolution rationale."
    )

    # Build the output: map each logical operator back to its syntactic source
    analysis_lines = []

    if features["conditionals"]:
        toks = [f"{c['token']}[{c['idx']}]" for c in features["conditionals"]]
        analysis_lines.append(
            f"• → (Implication): Source = conditional marker(s) {'/'.join(toks)}. "
            f"Antecedent governed by conditional node; consequent follows 'then'."
        )

    for c in features["logical_connectives"]:
        if c["type"] == "XOR":
            analysis_lines.append(
                f"• ⊕ (Exclusive OR): Source = {c['token']}[{c['idx']}] ({c['relation']}). "
                f"Correlative disjunction (cc:preconj + cc) forces exactly one alternative."
            )
        elif c["type"] == "NOR":
            analysis_lines.append(
                f"• NOR (Negated Disjunction): Source = {c['token']}[{c['idx']}] ({c['relation']}). "
                f"Equivalent to ¬A ∧ ¬B by De Morgan's laws."
            )
        elif c["type"] == "CONJ":
            analysis_lines.append(
                f"• ∧ (Conjunction): Source = {c['token']}[{c['idx']}] ({c['relation']}). "
                f"Coordinates parallel conjuncts under shared governor."
            )
        elif c["type"] == "DISJ":
            analysis_lines.append(
                f"• ∨ (Inclusive Disjunction): Source = {c['token']}[{c['idx']}] ({c['relation']}). "
                f"One or more disjuncts may hold."
            )

    if features["arguments"]:
        arg_lines = [f"{a['token']}[{a['idx']}]:{a['role']}" for a in features["arguments"]]
        analysis_lines.append(
            f"• Argument Binding: {'; '.join(arg_lines)}. "
            f"Predicate-argument structure resolved via dependency nsubj/dobj relations."
        )

    if features["variables"]:
        var_lines = [f"{v['token']}[{v['idx']}]:VAR({v['var']})" for v in features["variables"]]
        analysis_lines.append(
            f"• Variable Resolution: {'; '.join(var_lines)}. "
            f"Relative pronouns bound to subject domain via acl:relcl restrictors."
        )

    output_text = (
        "<stage>Operator Mapping</stage>\n"
        "<mapping>\n" + "\n".join(analysis_lines) + "\n</mapping>\n\n"
        "<stage>Reference Atoms</stage>\n"
        "<atomic>\n" + "\n".join(f"{k}: {v}" for k, v in sorted(atomic_facts.items())) + "\n</atomic>"
    )

    return {
        "instruction": "Analyze the logic-syntax alignment for the following claim.",
        "input": input_text,
        "output": output_text,
        "system": REVERSE_SYSTEM,
        "label": "reverse_aux"
    }


# ==================== Main Conversion Function (integrates all plans) ====================

def convert_to_desft(raw_data: List[Dict]) -> List[Dict]:
    """
    DeSFT data generator.
    """
    alpaca_records = []

    for item in raw_data:
        claim = item["claim"]
        dep_tree = item.get("dependency_tree", "")

        # Extract features once and reuse
        features = extract_linguistic_features(dep_tree)

        # Plan 7: generate a reverse auxiliary task with 10% probability
        if random.random() < REVERSE_TASK_RATIO:
            fmt = random.choice(FORMAT_TYPES)
            record = build_reverse_record(item, features, fmt)
            alpaca_records.append(record)
            continue

        # Forward task data
        atomic_facts = item["atomic_facts"]
        symbolic_logic = item["symbolic_logic"]

        # Plan 3: random masking (20% none / 30% lite / 50% full)
        r = random.random()
        if r < MASK_NONE_RATIO:
            mask_type = "none"
        elif r < MASK_NONE_RATIO + MASK_LITE_RATIO:
            mask_type = "lite"
        else:
            mask_type = "full"

        # Plan 5: random format
        fmt = random.choice(FORMAT_TYPES)

        # Plan 2: clues go into the system prompt; input keeps only the claim
        system_prompt = build_forward_system(features, mask_type, fmt)
        input_text = build_forward_input(claim)

        # Output (strictly three stages, no Verify)
        output_text = build_forward_output(atomic_facts, symbolic_logic, features, claim)

        record = {
            "instruction": "Decompose the claim into minimal sufficient atomic facts, then generate the symbolic logic expression.",
            "input": input_text,
            "output": output_text,
            "system": system_prompt,
            "label": f"desft_{mask_type}_{fmt}"
        }

        alpaca_records.append(record)

    return alpaca_records


# ==================== Usage Example ====================

if __name__ == "__main__":
    set_seed(42)

    DEP_FILE = "./data/Claim2Atom_Depend.json"
    DATA_FILE = "./data/Claim2Atom_Select.json"
    OUTPUT_FILE = "./desft_data/chaim_desft_data.json"

    with open(DEP_FILE, "r", encoding="utf-8") as f:
        dep_data = json.load(f)
    with open(DATA_FILE, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    sample_data = []
    for dep, sample in zip(dep_data, raw_data):
        sample["dependency_tree"] = dep["dependency_tree"]
        sample_data.append(sample)

    # Convert
    enhanced_data = convert_to_desft(sample_data)

    # Label distribution statistics
    labels = {}
    for r in enhanced_data:
        labels[r.get("label", "unknown")] = labels.get(r.get("label", "unknown"), 0) + 1
    print("Label distribution:", json.dumps(labels, indent=2))

    # Save
    out_path = Path(OUTPUT_FILE)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(enhanced_data, f, indent=2, ensure_ascii=False)

    print(f"\n✅ Saved {len(enhanced_data)} records to {out_path}")

    # Print three representative samples
    for target_label in ["desft_full_bullet", "desft_none_compact", "reverse_aux"]:
        print("\n" + "=" * 70)
        print(f"SAMPLE: {target_label}")
        print("=" * 70)
        for r in enhanced_data:
            if r.get("label") == target_label or target_label in r.get("label", ""):
                print(json.dumps(r, indent=2, ensure_ascii=False))
                break