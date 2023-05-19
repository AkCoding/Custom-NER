import spacy

# Load the spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the custom NER patterns
ner_patterns = [
    ("MRS CORO - Unity Road", {"label": "ORG"}),
    ("Apple", {"label": "ORG"}),
    ("Python", {"label": "LANGUAGE"}),
    ("Java", {"label": "LANGUAGE"}),
    ("JavaScript", {"label": "LANGUAGE"})
]

# Add the custom NER patterns to the pipeline
ner = nlp.get_pipe("ner")
for pattern in ner_patterns:
    ner.add_label(pattern[1]['label'])

# Example text
text = "google, and Apple are major tech companies."

# Process the text
doc = nlp(text)

# Extract the custom named entities
custom_entities = []
for ent in doc.ents:
    if ent.label_ in ["ORG", "LANGUAGE"]:
        custom_entities.append((ent.text, ent.label_))

# Print the custom named entities
for entity, label in custom_entities:
    print("Entity:", entity)
    print("Label:", label)
    print()

