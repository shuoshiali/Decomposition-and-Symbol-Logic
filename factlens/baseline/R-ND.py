import json
import os
import re
from string import Template
import time

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + '/src')

from utils_Qwen_8B import APILLM


# ========== Stage 1: R-ND claim decomposition prompt ==========
PROMPT_RND_SYSTEM = "Seperate the decomposed subclaims with a hyphen."

PROMPT_RND = '''Please breakdown the following sentence into independent facts: He made his acting debut in the film The Moon is the Sun's Dream (1992), and continued to appear in small and supporting roles throughout the 1990s.
- He has an acting debut.
- He acted in a film.
- His acting debut was in a film.
- His acting debut was in The Moon is the Sun's Dream.
- He acted in The Moon is the Sun's Dream.
- The Moon is the Sun's Dream is a film.
- The Moon is the Sun's Dream was released in 1992.
- His acting debut occurred in 1992.
- He appeared in small roles.
- He appeared in supporting roles.
- His small roles occurred throughout the 1990s.
- His supporting roles occurred throughout the 1990s.
- His appearance in small roles occurred after his acting debut.
- His appearance in supporting roles occurred after his acting debut.

Please breakdown the following sentence into independent facts: He is also a successful producer and engineer, having worked with a wide variety of artists, including Willie Nelson, Tim McGraw, and Taylor Swift.
- He is a producer.
- He is successful at being a producer.
- He is an engineer.
- He is successful at being an engineer.
- He has worked with a wide variety of artists.
- Willie Nelson is an artist.
- He has worked with Willie Nelson.
- Tim McGraw is an artist.
- He has worked with Tim McGraw.
- Taylor Swift is an artist.
- He has worked with Taylor Swift.

Please breakdown the following sentence into independent facts: In 1963, Collins became one of the third group of astronauts selected by NASA and he served as the back-up Command Module Pilot for the Gemini 7 mission.
- NASA selected a third group of astronauts.
- Collins belonged to the third group of astronauts.
- Collins was selected by NASA.
- Collins's selection by NASA occurred in 1963.
- The Gemini 7 mission has a back-up Command Module Pilot.
- Collins's role in the Gemini 7 mission was as the back-up Command Module Pilot.
- Collins participated in the Gemini 7 mission.

Please breakdown the following sentence into independent facts: In addition to his acting roles, Bateman has written and directed two short films and is currently in development on his feature debut.
- Bateman has acting roles.
- Bateman has written short films.
- The number of short films Bateman has written is two.
- Bateman has directed short films.
- The number of short films Bateman has directed is two.
- Bateman is currently in development on his feature debut.
- The two short films were made before his feature debut.
- His acting roles came before his feature debut.

Please breakdown the following sentence into independent facts: Michael Collins (born October 31, 1930) is a retired American astronaut and test pilot who was the Command Module Pilot for the Apollo 11 mission in 1969.
- Michael Collins was born in October.
- Michael Collins was born on the 31st day of a month.
- Michael Collins was born in 1930.
- Michael Collins is retired.
- Michael Collins is American.
- Michael Collins was an astronaut.
- Michael Collins was a test pilot.
- Michael Collins participated in the Apollo 11 mission.
- Michael Collins's participation in the Apollo 11 mission occurred in 1969.
- The Apollo 11 mission was active in 1969.
- The day of Michael Collins's birth occurred before his year of participation in the Apollo 11 mission.
- The Apollo 11 mission had a Command Module Pilot.
- Michael Collins's role in the Apollo 11 mission was as the Command Module Pilot.

Please breakdown the following sentence into independent facts: He was an American composer, conductor, and musical director.
- He was American.
- He was a composer.
- He was a conductor.
- He was a musical director.

Please breakdown the following sentence into independent facts: She currently stars in the romantic comedy series, Love and Destiny, which premiered in 2019.
- She stars in Love and Destiny.
- Love and Destiny is a series.
- Love and Destiny is a romantic comedy.
- Love and Destiny premiered in 2019.

Please breakdown the following sentence into independent facts: During his professional career, McCoy played for the Broncos, the San Diego Chargers, the Minnesota Vikings, and the Jacksonville Jaguars.
- McCoy had a professional career.
- McCoy played for the Broncos.
- McCoy played for the San Diego Chargers.
- The Chargers are from San Diego.
- McCoy played for the Minnesota Vikings.
- The Vikings are from Minnesota.
- McCoy played for the Jacksonville Jaguars.
- The Jaguars are from Jacksonville.

Please breakdown the following sentence into independent facts: {claim}
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


def parse_rnd_output(resp: str) -> list:
    """Parse the R-ND decomposition output: split by '- ' and strip prefixes and filler text."""
    subclaims = []
    for line in resp.splitlines():
        line = line.strip()
        if not line:
            continue
        # Skip common filler prefixes
        if line.startswith("Please breakdown") or line.startswith("Here") or line.startswith("The following"):
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
            "No additional subclaim needed",
            "No second subclaim is needed",
            "No further decomposition is possible",
            "No further decomposition possible",
            "No further decomposition needed",
            "no further decomposition is needed",
            "The claim is already concise",
            "The original claim is too short",
            "The original claim is already concise",
            "This claim cannot be further decomposed"
        ]
        if line and not any(n.lower() in line.lower() for n in nonsense):
            subclaims.append(line)
    return subclaims


def generate_rnd_decompose(llm: APILLM, user_fact: str) -> list:
    """
    Stage 1: R-ND claim decomposition.
    :return: list of subclaims
    """
    prompt = PROMPT_RND.format(claim=user_fact)
    messages = [
        {"role": "system", "content": PROMPT_RND_SYSTEM},
        {"role": "user", "content": prompt}
    ]
    
    try:
        resp = llm.generate(
            messages=messages,
            do_sample=True,
            temperature=0.01
        ).strip()
    except Exception:
        time.sleep(30)
        resp = llm.generate(
            messages=messages,
            do_sample=True,
            temperature=0.01
        ).strip()
    
    return parse_rnd_output(resp)


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
    Two-stage pipeline: R-ND decomposition -> symbolic logic generation.
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

        # ========== Stage 1: R-ND decomposition ==========
        print(f"  [Stage 1] R-ND decomposition...")
        subclaims = generate_rnd_decompose(llm, user_fact)
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
    model_name = "Qwen3-8B_R-ND"
    llm = APILLM(log_path=f"./src/log/{model_name}/tokens_log")

    INPUT_JSON = "./src/FactLens_Entity.json"
    OUTPUT_JSON = f"./src/data/FactLens_{model_name}.json"

    process_json_file(INPUT_JSON, OUTPUT_JSON, llm)