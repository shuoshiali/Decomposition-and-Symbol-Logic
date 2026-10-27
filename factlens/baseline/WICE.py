import json
import os
import re
from string import Template
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')

from utils_Qwen_8B import APILLM


# ========== Stage 1: WICE claim decomposition prompt ==========
PROMPT_WICE_SYSTEM = "Seperate the decomposed subclaims with a hyphen."

PROMPT_WICE = '''Segment the following sentence into individual facts:

Sentence: Other title changes included Lord Steven Regal and The Nasty Boys winning the World Television Championship and the World Tag Team Championship respectively.
Facts:
- Lord Steven Regal wan the World Television Championship.
- The Nasty Boys wan and the World Tag Team Championship.

Sentence: The parkway was opened in 2001 after just under a year of construction and almost two decades of community requests.
Facts:
- The parkway was opened in 2001.
- The parkway was opened after just under a year of construction.
- The parkway was opened after two decades of community requests.

Sentence: Touring began in Europe in April-June with guitarist Paul Gilbert as the opening act, followed by Australia and New Zealand in July, Mexico and South America in late July-August, and concluding in North America in October-November.
Facts:
- Touring began in Europe in April-June.
- The opening act was guitarist Paul Gilbert.
- There was a tour in Australia in July.
- There was a tour in New Zealand in July.
- There was a tour in Mexico in late July-August.
- There was a tour in South America in late July-August
- The tour was concluded in North America in October-November.

Sentence: In March 2018, the company partnered With Amazon Web Services (AWS) to offer Al-enabled conversational solutions to customers in India.
Facts:
- The company partnered with Amazon Web Services (AWS) in March 2018.
- The two companies partnered to offer Al-enabled conversational solutions to customers in India.

Sentence: The most significant of these is in Germany, which now has a Yazidi community of more than 200,000 living primarily in Hannover, Bielefeld, Celle, Bremen, Bad Oeynhausen, Pforzheim and Oldenburg.
Facts:
- The most significant of these is in Germany.
- Germany now has a Yazidi community of more than 200,000.
- Yazidi community in Germany lives primarily in Hannover.
- Yazidi community in Germany lives primarily in Bielefeld.
- Yazidi community in Germany lives primarily in Celle.
- Yazidi community in Germany lives primarily in Bremen.
- Yazidi community in Germany lives primarily in Bad Oeynhausen.
- Yazidi community in Germany lives primarily in Pforzheim.
- Yazidi community in Germany lives primarily in Oldenburg.

Sentence: A previous six-time winner of the Nations' Cup, Sebastian Vettel became Champion of Champions for the first time, defeating Tom Kristensen, who made the final for the fourth time, 2-0.
Facts:
- Sebastian Vettel is a previous six-time winner of the Nations' Cup.
- Sebastian Vettel became Champion of Champions for the first time.
- Sebastian Vettel defeated Tom Kristensen.
- Tom Kristensen made the final for the fourth time.
- The score was 2-0.

Sentence: {claim}
Facts:
'''


# ========== Stage 2: Generate symbolic logic from subclaims ==========
PROMPT_SYMBOL_FROM_SUBCLAIMS = """You are a logical atomization assistant. Given a natural language sentence and its decomposed subclaims (atomic propositions), represent the sentence's logical structure using these subclaims.

Output is enclosed in "{{}}":
- a logical formula using these subclaims (referenced as A1, A2, ...) and connectives (∧, ∨, ¬, →, ↔) and parentheses.

Rules:
1. Each subclaim is referenced as A1, A2, ... in the order provided.
2. Use logical connectives (∧, ∨, ¬, →, ↔) and parentheses to accurately represent the logical structure.
3. Output only the formula enclosed in {{}}.

---
Example:
Input: 
Fact: "If Bonnie is either both a young child or teenager who wishes to further her academic career and chaperones high school dances or neither a young child nor teenager who wishes to further her academic career, then Bonnie is either a student who attends the school or is an inactive member."
Subclaims:
- A1: "Bonnie is a young child."
- A2: "Bonnie is a teenager."
- A3: "Bonnie wishes to further her academic career and educational opportunities."
- A4: "Bonnie chaperones high school dances."
- A5: "Bonnie is a student who attends the school."
- A6: "Bonnie is an inactive and disinterested member of the community."

Output:
{{(((A1 ∨ A2) ∧ A3 ∧ A4) ∨ (¬A1 ∧ ¬A2 ∧ ¬A3)) → (A5 ∨ A6)}}
---
Now process the following input:
Input: 
Fact: "$user_fact"
Subclaims:
$user_subclaims

Output:"""


def subclaims_str(subclaims: list) -> str:
    """Convert a list of subclaims into an A1, A2, ... formatted string."""
    result = ""
    for i, sc in enumerate(subclaims):
        result += f'- A{i+1}: "{sc}",\n'
    result = result.rstrip(",\n")
    return result


def extract_brace_content(resp: str) -> str:
    """Extract the content wrapped in {{}} or {}."""
    # Try matching {{...}} first
    double_left = resp.find("{{")
    double_right = resp.find("}}")
    if double_left != -1 and double_right != -1 and double_left < double_right:
        return resp[double_left + 2:double_right].strip()
    
    # Fallback to a single pair of braces
    left = resp.find("{")
    right = resp.rfind("}")
    if left == -1 or right == -1 or left >= right:
        return "None"
    content = resp[left + 1:right].strip()
    while content.startswith("{") and content.endswith("}"):
        content = content[1:-1].strip()
    return content


def parse_wice_output(resp: str) -> list:
    """Parse the WICE decomposition output: split by '- ' and strip prefixes and filler text."""
    subclaims = []
    for line in resp.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip common prefixes and format markers
        if line.startswith("Sentence:") or line.startswith("Facts:") or line.startswith("Please") or line.startswith("Here"):
            continue
        # Remove the '- ' prefix
        if line.startswith("- "):
            line = line[2:].strip()
        elif line.startswith("-"):
            line = line[1:].strip()
        # Filter out filler lines and empty strings
        nonsense = [
            "No additional subclaim is needed",
            "No additional subclaim can be generated",
            "No further decomposition is possible",
            "The claim is already concise",
            "The original claim is too short",
            "This claim cannot be further decomposed"
        ]
        if line and not any(n.lower() in line.lower() for n in nonsense):
            subclaims.append(line)
    return subclaims


def generate_wice_decompose(llm: APILLM, user_fact: str) -> list:
    """
    Stage 1: WICE claim decomposition.
    :return: list of subclaims
    """
    prompt = PROMPT_WICE.format(claim=user_fact)
    messages = [
        {"role": "system", "content": PROMPT_WICE_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    
    try:
        resp = llm.generate(
            messages=messages,
            do_sample=True,
            temperature=0.0
        ).strip()
    except Exception:
        time.sleep(30)
        resp = llm.generate(
            messages=messages,
            do_sample=True,
            temperature=0.0
        ).strip()
    
    return parse_wice_output(resp)


def generate_symbol_from_subclaims(llm: APILLM, user_fact: str, subclaims: list) -> str:
    """
    Stage 2: Generate symbolic logic from subclaims.
    """
    if not subclaims:
        return ""
    
    subclaims_formatted = subclaims_str(subclaims)
    prompt = Template(PROMPT_SYMBOL_FROM_SUBCLAIMS).substitute(
        user_fact=user_fact,
        user_subclaims=subclaims_formatted
    )
    
    try:
        resp = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            do_sample=True,
            temperature=0.01
        ).strip()
    except Exception:
        time.sleep(30)
        resp = llm.generate(
            messages=[{"role": "user", "content": prompt}],
            do_sample=True,
            temperature=0.01
        ).strip()
    
    return extract_brace_content(resp)


def process_json_file(input_path: str, output_path: str, llm: APILLM):
    """
    Two-stage pipeline: WICE decomposition -> symbolic logic generation.
    Process items one by one, save incrementally, skip failed items, and support resuming from checkpoints.
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"File {input_path} not found")

    with open(input_path, "r", encoding="utf-8") as f:
        origin_data = json.load(f)

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
    start_idx = len(output_data)

    for idx in range(start_idx, total_num):
        item = origin_data[idx]
        print(f"\n===== Processing item {idx+1}/{total_num} =====")
        user_fact = item.get("claim", "")

        if not user_fact:
            print(f"[Warning] Claim at index {idx} is empty; skipped")
            output_data.append({
                "id": item.get("id"),
                "claim": user_fact,
                "subclaims": [],
                "symbolic_logic": ""
            })
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, ensure_ascii=False, indent=2)
            continue

        # ========== Stage 1: WICE decomposition ==========
        print(f"  [Stage 1] WICE decomposition...")
        subclaims = generate_wice_decompose(llm, user_fact)
        print(f"  Decomposed into {len(subclaims)} subclaims")
        for i, sc in enumerate(subclaims[:5]):
            print(f"    - {sc}")
        if len(subclaims) > 5:
            print(f"    ... and {len(subclaims) - 5} more")

        # ========== Stage 2: Symbolic logic generation ==========
        print(f"  [Stage 2] Generating symbolic logic...")
        symbolic_logic = generate_symbol_from_subclaims(llm, user_fact, subclaims)
        print(f"  Symbolic logic: {symbolic_logic}")

        # Save the result
        output_data.append({
            "id": item.get("id"),
            "claim": user_fact,
            "subclaims": subclaims,
            "symbolic_logic": symbolic_logic,
            "claim_entities": item.get("claim_entities", [])
        })
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(output_data, f, ensure_ascii=False, indent=2)
        print(f"  Item {idx+1} processed and saved")

        llm.save_token_stats()
        llm.reset_token_stats()
        llm.log_count = idx + 1

    print(f"\nAll done! Processed {total_num} items in total. Results saved to: {output_path}")


if __name__ == "__main__":
    model_name = "Qwen3-8B_WICE"
    llm = APILLM(log_path=f"./src/log/{model_name}/tokens_log")

    INPUT_JSON = "./src/FactLens_Entity.json"
    OUTPUT_JSON = f"./src/data/FactLens_{model_name}.json"

    process_json_file(INPUT_JSON, OUTPUT_JSON, llm)