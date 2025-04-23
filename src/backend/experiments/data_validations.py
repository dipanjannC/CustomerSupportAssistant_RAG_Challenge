# import pandas as pd
# import spacy
# import contextualSpellCheck
# import re
# from pathlib import Path
# import os

# # Load spaCy model and add contextual spell checker
# nlp = spacy.load("en_core_web_sm")
# contextualSpellCheck.add_to_pipe(nlp)

# project_root = Path(__file__).resolve().parents[3]
# raw_dataset = os.path.join(project_root,"data" ,"raw" , "raw_customer_support_dataset.json")

# # Load your dataset
# # Replace 'raw_customer_support_dataset.json' with your actual file path
# df = pd.read_json(raw_dataset, lines=True)
# df = df.sample(1000)  # Sample 1000 rows for analysis

# # Define a set of informal terms (slang)
# informal_terms = {
#     'lol', 'omg', 'btw', 'idk', 'smh', 'brb', 'lmao', 'rofl', 'wtf', 'bff', 'tbh',
#     'gr8', 'thx', 'u', 'ur', 'pls', 'plz', 'k', 'ya', 'cya', 'luv', 'omfg', 'nvm'
# }

# # Define a set of ambiguous words
# ambiguous_words = {
#     'bank', 'bat', 'lead', 'tear', 'bark', 'light', 'match', 'right', 'rock', 'spring',
#     'current', 'date', 'kind', 'mean', 'well', 'fine', 'fair', 'case', 'left', 'object'
# }

# # Initialize lists to store analysis results
# misspellings = []
# informal_counts = []
# ambiguity_counts = []

# for text in df['output']:
#     doc = nlp(text)

#     misspelled = len(doc._.suggestions_spellCheck)
#     words = set(re.findall(r'\b\w+\b', text.lower()))
#     informal_count = len(words & informal_terms)

#     # Contextual ambiguity detection
#     ambiguity_count = len(words & ambiguous_words)

#     # Append results
#     misspellings.append(misspelled)
#     informal_counts.append(informal_count)
#     ambiguity_counts.append(ambiguity_count)

# df['misspellings'] = misspellings
# df['informal_terms'] = informal_counts
# df['ambiguous_words'] = ambiguity_counts


# print(df.head())
# # Save the updated DataFrame to a new JSON file
# output_path = os.path.join(project_root, "data", "output", "customer_support_analysis.json")
# df.to_json(output_path, orient='records', lines=True)
