import spacy


def extract_triples(nlp, text: str) -> list[tuple[str, str, str]]:
    """
    Extract triples (subject, relation, object) from text.
    Strategy: dependency-based; iterate over each verb/copula and extract its nsubj and dobj/attr.
    Fictional words (tagged X by spaCy) are treated as nouns in syntactic roles.
    """
    doc = nlp(text)
    triples = []
    
    for sent in doc.sents:
        for token in sent:
            # Identify the predicate head: a verb, copula, or fictional verb
            if not (token.pos_ in {"VERB", "AUX"} or _is_fictional_verb(token)):
                continue
            
            # Extract the subject list
            subjects = []
            for child in token.children:
                if child.dep_ in {"nsubj", "nsubjpass", "csubj"} or _is_fictional_noun(child):
                    subj = _get_phrase_text(child)
                    if subj:
                        subjects.append(subj)
            
            # Extract the object list (direct object, attribute, prepositional object)
            objects = []
            for child in token.children:
                if child.dep_ in {"dobj", "attr", "pobj", "iobj", "acomp", "oprd"}:
                    obj = _get_phrase_text(child)
                    if obj:
                        objects.append(obj)
                elif child.dep_ == "prep":
                    # Prepositional-phrase object, e.g. "dances at school"
                    for pchild in child.children:
                        if pchild.dep_ in {"pobj", "pcomp"}:
                            obj = _get_phrase_text(pchild)
                            if obj:
                                objects.append(obj)
            
            # Combine into triples
            if subjects:
                relation = token.lemma_ if token.pos_ == "VERB" else token.text
                if objects:
                    for subj in subjects:
                        for obj in objects:
                            triples.append((subj, relation, obj))
                else:
                    # No object: try to find an attribute/complement (copular construction such as "is a student")
                    for child in token.children:
                        if child.dep_ in {"attr", "acomp"} or _looks_like_noun(child):
                            obj = _get_phrase_text(child)
                            if obj:
                                triples.append((subjects[0], relation, obj))
    
    # Fallback: if dependency parsing yields no triples, use the position heuristic
    if not triples:
        triples = _fallback_position_heuristic(doc)
    
    return triples


def _is_fictional_noun(token) -> bool:
    """Check whether the token is a fictional noun (X tag serving a noun role)."""
    if token.pos_ == "X":
        if token.dep_ in {"nsubj", "nsubjpass", "dobj", "attr", "pobj", "appos", "compound", "nmod"}:
            return True
        # Preceded by a determiner/adjective modifier, very likely a noun
        if any(c.dep_ in {"det", "amod"} and c.i < token.i for c in token.children):
            return True
    return False


def _is_fictional_verb(token) -> bool:
    """Check whether the token is a fictional verb (X tag serving as the predicate)."""
    return token.pos_ == "X" and token.dep_ in {"ROOT", "ccomp", "xcomp", "relcl", "advcl", "conj"}


def _looks_like_noun(token) -> bool:
    """Check whether the token looks like a noun (including fictional words)."""
    return token.pos_ in {"NOUN", "PROPN", "PRON", "X"} or _is_fictional_noun(token)


def _get_phrase_text(token) -> str:
    """
    Extract the phrase text spanned by the token and its subtree.
    Skip leading determiners (det/poss); keep the head noun and trailing modifiers.
    """
    phrase_tokens = []
    for t in sorted(token.subtree, key=lambda x: x.i):
        if t.is_punct:
            continue
        # Skip leading determiners such as "The", "her", "a"
        if t.dep_ in {"det", "poss"} and t.i < token.i:
            continue
        phrase_tokens.append(t)
    
    if not phrase_tokens:
        return token.text
    
    return " ".join(t.text for t in phrase_tokens).strip()


def _fallback_position_heuristic(doc) -> list[tuple[str, str, str]]:
    """
    Fallback strategy: when dependency parsing completely fails, extract via a position heuristic.
    Find the first verb; the first noun before it is the subject, the first noun after it is the object.
    """
    triples = []
    for sent in doc.sents:
        tokens = [t for t in sent if not t.is_punct]
        
        # Locate the first verb
        verb_idx = None
        for i, t in enumerate(tokens):
            if t.pos_ in {"VERB", "AUX"}:
                verb_idx = i
                break
        
        if verb_idx is None:
            continue
        
        # Search backward for the subject (fictional words count as nouns)
        subj = None
        for i in range(verb_idx - 1, -1, -1):
            if _looks_like_noun(tokens[i]):
                subj = tokens[i].text
                break
        
        # Search forward for the object
        obj = None
        for i in range(verb_idx + 1, len(tokens)):
            if _looks_like_noun(tokens[i]):
                obj = tokens[i].text
                break
        
        if subj and obj:
            triples.append((subj, tokens[verb_idx].lemma_, obj))
    
    return triples


if __name__ == "__main__":
    nlp = spacy.load("en_core_web_lg")
    text = """If Bonnie is either both a young child or teenager who wishes to further her academic career and educational opportunities and chaperones high school dances or neither is a young child nor teenager who wishes to further her academic career and educational opportunities, then Bonnie is either a student who attends the school or is an inactive and disinterested member of the community."""
    
    triples = extract_triples(nlp, text)
    print(f"Extracted {len(triples)} triples:")
    for triple in triples:
        print(f"  {triple}")