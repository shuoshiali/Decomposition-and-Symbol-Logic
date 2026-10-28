# -*- coding: utf-8 -*-

import json
from stanfordcorenlp import StanfordCoreNLP
import re
import nltk
from tqdm import tqdm

try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt', quiet=True)

try:
    from query.Tree_path_compression import process_depend_paths
except ImportError:
    def process_depend_paths(s):
        return s


# ==================== Logic pruning configuration ====================

# Base filtering: these dependency arcs are usually absorbed or resolved in FOL
_PRUNE_BASE_RELS = {'case', 'punct', 'aux', 'auxpass', 'expl', 'ref', 'dep', 'cop'}

# Quantifier determiners (a det arc is kept when its dependent is one of these)
_QUANTIFIERS = {'all', 'every', 'each', 'any', 'some', 'no', 'none', 'neither', 'both', 'either'}


def _extract_word(word_idx_str):
    """Extract the bare word from 'word[idx]'."""
    match = re.match(r'^(.+?)\[\d+\]$', word_idx_str)
    return match.group(1) if match else word_idx_str


def _should_keep_arc(rel, dep_word_idx):
    """
    Decide whether a dependency arc is kept in the logical frame.
    """
    base_rel = rel.split(':')[0]
    dep_word = _extract_word(dep_word_idx).lower()

    if base_rel in _PRUNE_BASE_RELS:
        return False

    # det: keep only quantifier determiners
    if base_rel == 'det':
        return dep_word in _QUANTIFIERS

    # mark: keep only if/unless/that/whether
    if base_rel == 'mark':
        return dep_word in ('if', 'unless', 'whether', 'that')

    return True


def _get_node_annotation(rel, dep_word_idx):
    """Generate an FOL logic annotation for the node as a direct hint for the 8B model."""
    dep_word = _extract_word(dep_word_idx).lower()
    base_rel = rel.split(':')[0]

    # Quantifier
    if base_rel == 'det':
        if dep_word in ('all', 'every', 'each', 'any'):
            return "  → QUANT: ∀x"
        if dep_word in ('some', 'a', 'an'):
            return "  → QUANT: ∃x"
        if dep_word in ('no', 'none', 'neither'):
            return "  → QUANT: ¬∃"
        if dep_word == 'both':
            return "  → QUANT: ∀ (binary)"
        if dep_word == 'either':
            return "  → XOR: ⊕"

    # Negation
    if base_rel == 'neg':
        return "  → NEG: ¬"

    # Condition
    if base_rel == 'mark':
        if dep_word == 'if':
            return "  → COND: →"
        if dep_word == 'unless':
            return "  → COND: ¬→"

    # Relative clause
    if 'acl' in rel and 'relcl' in rel:
        return "  → RESTRICT"

    # Coordination / disjunction / conjunction
    if rel == 'cc:preconj' and dep_word == 'either':
        return "  → XOR: ⊕"
    if rel == 'cc:preconj' and dep_word == 'neither':
        return "  → NOR"
    if base_rel == 'cc':
        if dep_word == 'or':
            return "  → DISJ: ∨"
        if dep_word == 'and':
            return "  → CONJ: ∧"
        if dep_word == 'nor':
            return "  → NOR"

    # Variable / argument markers
    if base_rel in ('nsubj', 'nsubj:pass'):
        if dep_word in ('who', 'which', 'what'):
            return "  → VAR: x"
        return "  → ARG1"
    if base_rel in ('dobj', 'obj', 'iobj'):
        return "  → ARG2"
    if base_rel == 'nmod':
        return "  → ARG/ADJUNCT"

    return ""


def _sort_children_for_logic(children):
    """Sort child nodes: quantifiers/operators first, then arguments, then clauses/modifiers."""
    def sort_key(item):
        child_word_idx, rel = item
        word = _extract_word(child_word_idx).lower()
        base_rel = rel.split(':')[0]

        if base_rel == 'det' and word in _QUANTIFIERS:
            return (0, 0, word)
        if base_rel == 'neg':
            return (0, 1, word)
        if base_rel == 'mark' and word in ('if', 'unless'):
            return (0, 2, word)
        if rel == 'cc:preconj':
            return (0, 3, word)
        if base_rel in ('nsubj', 'nsubj:pass'):
            return (1, 0, word)
        if base_rel in ('dobj', 'obj', 'iobj'):
            return (1, 1, word)
        if 'acl' in rel:
            return (2, 0, word)
        if 'advcl' in rel:
            return (2, 1, word)
        if base_rel == 'nmod':
            return (2, 2, word)
        return (3, 0, word)

    return sorted(children, key=sort_key)


# ==================== Predicate-centered frame formatting ====================

def _format_predicate_frame(dep_dict):
    """
    Convert the dependency dict into a predicate-centered frame (for the 8B FOL prompt).
    """
    lines = []

    # Extract the actual root predicate pointed to by ROOT
    root_entries = dep_dict.get('Root', [])
    root_word = None
    for child, rel in root_entries:
        if rel in ('ROOT', 'root', 'HED'):
            root_word = child
            break

    if not root_word:
        return "[No root found]"

    lines.append(f"Predicate frame (root = {root_word}):")
    _format_subtree(dep_dict, root_word, prefix="  ", lines=lines)
    return '\n'.join(lines)


def _format_subtree(dep_dict, node, prefix, lines):
    """Recursively format a subtree, prune logic-irrelevant nodes, add annotations."""
    children = dep_dict.get(node, [])
    filtered = [(c, r) for c, r in children if _should_keep_arc(r, c)]

    if not filtered:
        return

    filtered = _sort_children_for_logic(filtered)

    for i, (child, rel) in enumerate(filtered):
        is_last = (i == len(filtered) - 1)
        connector = '└── ' if is_last else '├── '
        annotation = _get_node_annotation(rel, child)

        lines.append(f"{prefix}{connector}({rel}) {child}{annotation}")
        extension = '    ' if is_last else '│   '
        _format_subtree(dep_dict, child, prefix + extension, lines)


# ==================== Raw path mode (backward compatible) ====================

def preorder_traversal(tree, node, path, result):
    if node not in tree or not tree[node]:
        result.append("->".join(path))
        return
    for child, weight in tree[node]:
        if len(path) == 0:
            new_path = path + [f"{node}-({weight})->{child}"]
        else:
            new_path = path + [f"({weight})->{child}"]
        preorder_traversal(tree, child, new_path, result)


# ==================== Core interface (backward compatible) ====================

def dependency(sentence, nlp):
    words = list(nlp.word_tokenize(sentence))
    depend = list(nlp.dependency_parse(sentence))
    return depend, words


def Dependency_tree_builder(dependency_tree, words):
    """
    Dependency-dict builder kept fully backward compatible with the original code.
    """
    dependency_tree.sort(key=lambda x: x[2])
    words = [w + "[" + str(idx) + "]" for idx, w in enumerate(words)]
    rely_id = [arc[1] for arc in dependency_tree]
    relation = [arc[0] for arc in dependency_tree]
    heads = ['Root' if id == 0 else words[id - 1] for id in rely_id]

    dependency_dict = {}
    for i in range(len(words)):
        if relation[i] not in ['HED']:
            if heads[i] not in dependency_dict:
                dependency_dict[heads[i]] = []
            dependency_dict[heads[i]].append((words[i], relation[i]))
        else:
            if heads[i] == 'Root':
                if 'Root' not in dependency_dict:
                    dependency_dict['Root'] = []
                dependency_dict['Root'].append((words[i], relation[i]))
            else:
                if 'Root' not in dependency_dict:
                    dependency_dict['Root'] = []
                dependency_dict['Root'].append((heads[i], relation[i]))

    return dependency_dict


def get_dependency_parse(sentence, nlp, mode='frame'):
    """
    Generate the dependency parse.

    Args:
        sentence: str, input sentence
        nlp: StanfordCoreNLP instance
        mode: 'frame' (default) outputs the predicate-centered frame, for the 8B FOL prompt;
              'path' outputs raw leaf paths (backward compatible).

    Returns:
        str: formatted dependency parse
    """
    dep_line, word = dependency(sentence, nlp)

    offset = 0
    new_dep = []
    for i, (rel, idx1, idx2) in enumerate(dep_line):
        if rel == 'ROOT':
            offset = i
            new_dep.append((rel, idx1, idx2 + offset))
        else:
            new_dep.append((rel, idx1 + offset, idx2 + offset))

    dep_tree = Dependency_tree_builder(new_dep, word)

    if mode == 'frame':
        return _format_predicate_frame(dep_tree)
    else:
        result = []
        preorder_traversal(dep_tree, 'Root', [], result)
        result_str = " ".join(result)
        return process_depend_paths(result_str)


# ==================== Batch interface (multi-sentence) ====================

def get_dependency_parse_multi(text, nlp, mode='frame'):
    """
    Process multi-sentence text: split into sentences and generate the frame/path for each.
    Fits the [[TREE]] placeholder of the prompt template directly.
    """
    if not text or not text.strip():
        return "None."

    sentences = nltk.sent_tokenize(text.strip())
    outputs = []

    for sent in sentences:
        sent = sent.strip()
        if not sent:
            continue
        if sent[-1] not in '.?!':
            sent += '.'

        try:
            parsed = get_dependency_parse(sent, nlp, mode=mode)
            outputs.append(parsed)  # note: the 'Sentence: ...' prefix was removed
        except Exception as e:
            outputs.append(f"[Parse Error: {str(e)}]")  # same prefix removal

    return '\n\n'.join(outputs) if outputs else "None."


# ==================== Test ====================

if __name__ == "__main__":
    nlp = StanfordCoreNLP(
        r'/T20200133/Models/stanfordnlp/stanfordcorenlp/stanford-corenlp-full-2018-01-31',
        lang='en'
    )

    # DATA_FILE = "../data/FOLIO_ALL_Select.json"
    # OUTPUT_FILE = "../data/FOLIO_ALL_Depend.json"

    # with open(DATA_FILE, "r", encoding="utf-8") as f:
    #     sample_data = json.load(f)

    # save_data = []
    # for item in tqdm(sample_data):
    #     dependency_tree = get_dependency_parse_multi(item["claim"], nlp, mode='frame')
    #     item["dependency_tree"] = dependency_tree

    #     save_data.append(item)
    #     with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
    #         json.dump(save_data, f, indent=2, ensure_ascii=False)
    
    test_data = "If someone is white and blue, then they are nice."
    dependency_tree = get_dependency_parse_multi(test_data, nlp, mode='raw')
    print(dependency_tree)

    nlp.close()