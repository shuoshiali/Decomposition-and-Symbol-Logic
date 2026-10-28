import json
import random
import re
from typing import Any, Dict

# ==================== Constants (fixed at inference time) ====================
RANDOM_SEED = 12
random.seed(RANDOM_SEED)

FORMAT_TYPES = ["bullet", "compact", "json_like"]
MASK_NONE_RATIO = 0.20
MASK_LITE_RATIO = 0.30

SYSTEM_MESSAGE_TREE = (
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
    "<analysis>\n...\n</analysis>\n\n"
    "<stage>Decompose</stage>\n"
    "<atomic>\n...\n</atomic>\n\n"
    "<stage>Symbolize</stage>\n"
    "<symbol>\n...\n</symbol>\n\n"
    "Do NOT add any text, markdown, or explanations outside these XML tags."
)


# ==================== Feature extraction (core dependency) ====================

def extract_linguistic_features(dep_tree: str) -> Dict[str, Any]:
    """Extract structured linguistic clues from the raw dependency-tree text (with semantic filtering and an atomicity guard)."""
    features = {
        "logical_connectives": [],
        "conditionals": [],
        "arguments": [],
        "restrictions": [],
        "variables": [],
        "atomicity_guard": {"suppress": False, "reason": None}
    }

    lines = [l.strip() for l in dep_tree.split('\n') if l.strip() and '→' in l]

    # ==================== Build tree-level index ====================
    token_info = {}  # idx -> {"word": str, "relations": set()}
    for line in lines:
        m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
        if m:
            rel, word, idx = m.group(1), m.group(2), m.group(3)
            token_info.setdefault(idx, {"word": word, "relations": set()})
            token_info[idx]["relations"].add(rel)

    sorted_items = sorted(token_info.items(), key=lambda x: int(x[0]))
    reconstructed = " ".join(info["word"] for _, info in sorted_items).lower()

    def _nearby_words(idx: str, window: int = 3) -> set:
        idx_int = int(idx)
        return {info["word"].lower() for other_idx, info in token_info.items()
                if abs(int(other_idx) - idx_int) <= window}

    # Check whether the tree has a verbal predicate (distinguish NP structures from full sentences)
    verbal_rels = {"root", "cop", "xcomp", "ccomp", "advcl", "acl"}
    has_verb = any(
        any(r.startswith(vr) or vr in r for vr in verbal_rels for r in info["relations"])
        for info in token_info.values()
    )

    # ==================== Atomicity guard (global pattern detection) ====================
    math_patterns = [
        "sum of", "product of", "difference between", "ratio of",
        "total of", "total amount of", "average of", "mean of", "between",
        "percent change in", "percentage change in", "net change in",
        "adjusted balance", "beginning of period", "end of period",
        "case reserves", "total reserves", "cumulative",
    ]
    is_math_expr = any(pat in reconstructed for pat in math_patterns)

    exclusivity = {"only", "unique", "sole", "single"}
    has_exclusivity = any(w in reconstructed for w in exclusivity)
    has_relclause = any(
        any("acl" in r or "relcl" in r for r in info["relations"])
        for info in token_info.values()
    )

    if is_math_expr and not has_verb:
        features["atomicity_guard"] = {"suppress": True, "reason": "MATHEMATICAL_EXPRESSION"}
    elif has_exclusivity and has_relclause:
        features["atomicity_guard"] = {"suppress": True, "reason": "EXCLUSIVE_RESTRICTIVE_CLAUSE"}
    elif not has_verb and len(token_info) > 0:
        features["atomicity_guard"] = {"suppress": True, "reason": "NON_VERBAL_NP"}

    # ==================== Feature extraction after semantic filtering ====================
    for line in lines:
        line = line.strip()
        if not line or '→' not in line:
            continue

        # ---------- CONJ ----------
        if '→ CONJ:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ CONJ:\s*(.+)', line)
            if m:
                rel, token, idx, logic = m.group(1), m.group(2), m.group(3), m.group(4).strip()
                tok_lower = token.lower()
                nearby = _nearby_words(idx, window=2)

                # Filter 1: and/or inside a mathematical expression -> non-propositional coordination
                if tok_lower in {"and", "or"} and is_math_expr:
                    continue

                # Filter 2: coordination inside a prepositional phrase (sum of A and B / between A and B)
                if tok_lower in {"and", "or"} and (nearby & {"of", "with", "between", "among"}):
                    if not has_verb:
                        continue

                # Filter 3: and/or inside an exclusivity marker + relative clause
                if tok_lower in {"and", "or"} and has_exclusivity and has_relclause:
                    continue

                # Filter 4: purely nominal coordination
                if rel in {"cc", "conj"} and tok_lower in {"and", "or"}:
                    conj_items = [
                        info for oidx, info in token_info.items()
                        if any("conj" in r for r in info["relations"]) and oidx != idx
                    ]
                    if conj_items:
                        nominal_rels = {"nsubj", "dobj", "iobj", "nmod", "compound", "det", "amod"}
                        all_nominal = all(
                            any(nr in r for r in info["relations"] for nr in nominal_rels)
                            or len(info["relations"]) == 0
                            for info in conj_items
                        )
                        any_verbal_conj = any(
                            any(vr in r for r in info["relations"] for vr in verbal_rels)
                            for info in conj_items
                        )
                        if all_nominal and not any_verbal_conj:
                            continue

                features["logical_connectives"].append({
                    "relation": rel, "token": token, "idx": idx,
                    "type": "CONJ", "logic": logic
                })

        # ---------- XOR ----------
        elif '→ XOR:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ XOR:\s*(.+)', line)
            if m:
                if features["atomicity_guard"]["suppress"]:
                    continue
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "XOR", "logic": m.group(4).strip()
                })

        # ---------- DISJ ----------
        elif '→ DISJ:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ DISJ:\s*(.+)', line)
            if m:
                if features["atomicity_guard"]["suppress"]:
                    continue
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "DISJ", "logic": m.group(4).strip()
                })

        # ---------- NOR ----------
        elif '→ NOR' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                if features["atomicity_guard"]["suppress"]:
                    continue
                features["logical_connectives"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3),
                    "type": "NOR", "logic": "¬∨"
                })

        # ---------- COND / ARG / RESTRICT / VAR ----------
        elif '→ COND:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ COND:\s*(.+)', line)
            if m:
                features["conditionals"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "logic": m.group(4).strip()
                })
        elif '→ ARG1' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["arguments"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "role": "ARG1"
                })
        elif '→ ARG2' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["arguments"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "role": "ARG2"
                })
        elif '→ RESTRICT' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\]', line)
            if m:
                features["restrictions"].append({
                    "relation": m.group(1), "token": m.group(2), "idx": m.group(3)
                })
        elif '→ VAR:' in line:
            m = re.search(r'\(([\w:]+)\)\s+(\w+)\[(\d+)\].*→ VAR:\s*(.+)', line)
            if m:
                features["variables"].append({
                    "relation": m.group(1), "token": m.group(2),
                    "idx": m.group(3), "var": m.group(4).strip()
                })

    return features


# ==================== Formatting helpers ====================

def _format_bullet(features: Dict, lite: bool = False) -> str:
    """Format A: structured list."""
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
    """Format B: compact line."""
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
    """Format C: pseudo-JSON."""
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
    """Unified entry point: render linguistic clues in the given format."""
    if fmt == "bullet":
        return _format_bullet(features, lite)
    elif fmt == "compact":
        return _format_compact(features, lite)
    elif fmt == "json_like":
        return _format_json_like(features, lite)
    else:
        return _format_bullet(features, lite)


# ==================== Prompt building (core dependency of dependency_generator.py) ====================

def build_forward_system(features: Dict, mask_type: str, fmt: str) -> str:
    """
    Build the system prompt exactly consistent with SFT.
    mask_type: none | lite | full
    """
    # Atomicity guard override
    guard = features.get("atomicity_guard", {})

    if guard.get("suppress"):
        reason = guard.get("reason", "")
        guard_msg = (
            f"\n\n[ATOMICITY GUARD: {reason}] "
            f"This claim expresses a SINGLE atomic proposition."
        )
        # guard_msg = (
        #     f"\n\n[Note: This proposition needs to be more detailed (use \"It is\"/\"There are\").]"
        # )
        base = SYSTEM_MESSAGE_TREE + guard_msg
        return f"{base}"

    #     if mask_type == "none":
    #         return base

    #     clues = format_linguistic_features(features, fmt, lite=(mask_type == "lite"))
    #     return (
    #         f"{base}\n\n"
    #         f"[Syntactic Clues for This Claim]\n{clues}\n\n"
    #         f"Use these clues only for scope validation, but respect the ATOMICITY GUARD above."
    #     )

    if mask_type == "none":
        return SYSTEM_MESSAGE_TREE

    clues = format_linguistic_features(features, fmt, lite=(mask_type == "lite"))

    return (
        f"{SYSTEM_MESSAGE_TREE}\n\n"
        f"[Syntactic Clues for This Claim]\n{clues}\n"
    )


def build_forward_input(claim: str) -> str:
    """Build the user input exactly consistent with SFT (the tree has moved into the system message; the input keeps only the claim)."""
    return f"Claim: {claim}"