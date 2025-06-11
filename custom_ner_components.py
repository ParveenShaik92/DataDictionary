import spacy
from spacy.matcher import PhraseMatcher
from spacy.tokens import Span, Doc
from spacy.language import Language # Required for @Language.component
import os # For checking if model needs download
import sys # For sys.exit if model download fails

# Define terms to match as GENDER
gender_terms = ["Male", "Female", "M", "F"]

# This global matcher will be initialized when setup_gender_ner_component is called
# It needs access to nlp.vocab, so we'll set it up inside the function.
_gender_matcher = None

# Register the custom component with the @Language.component decorator
# This function will be called by nlp.add_pipe
@Language.component("gender_entity_component")
def gender_entity_component(doc: Doc) -> Doc:
    """
    spaCy custom pipeline component to detect GENDER entities based on a phrase matcher.
    """
    global _gender_matcher
    if _gender_matcher is None:
        # This should ideally not happen if setup_gender_ner_component is called first,
        # but as a fallback, initialize with the current nlp's vocab.
        _gender_matcher = PhraseMatcher(doc.vocab, attr="LOWER")
        patterns = [doc.vocab.make_doc(term) for term in gender_terms]
        _gender_matcher.add("GENDER", patterns)


    matches = _gender_matcher(doc)
    new_ents = list(doc.ents)  # Keep existing entities

    for match_id, start, end in matches:
        span = Span(doc, start, end, label=doc.vocab.strings["GENDER"])

        # Check if the entity is already part of the entities in doc.ents
        # and avoid adding overlapping entities
        is_overlap = False
        for ent in doc.ents:
            if (span.start >= ent.start and span.start < ent.end) or \
               (span.end > ent.start and span.end <= ent.end) or \
               (ent.start >= span.start and ent.start < span.end) or \
               (ent.end > span.start and ent.end <= span.end):
                is_overlap = True
                break
        if not is_overlap:
            new_ents.append(span)

    # Ensure we only add valid, non-overlapping entities
    doc.ents = spacy.util.filter_spans(new_ents) # filters overlapping or invalid spans
    return doc

def setup_gender_ner_component(nlp: Language) -> Language:
    """
    Sets up the spaCy pipeline with the custom GENDER entity detection component.

    Args:
        nlp (Language): The spaCy Language model object.

    Returns:
        Language: The spaCy Language model with the custom component added.
    """
    global _gender_matcher

    # Initialize the PhraseMatcher with the provided nlp's vocab
    _gender_matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    patterns = [nlp.make_doc(term) for term in gender_terms]
    _gender_matcher.add("GENDER", patterns)

    # Add custom component to the pipeline using its string name
    if "gender_entity_component" not in nlp.pipe_names:
        nlp.add_pipe("gender_entity_component", after="ner")
        print("'gender_entity_component' added to spaCy pipeline.")
    else:
        print("'gender_entity_component' already in pipeline.")

    return nlp