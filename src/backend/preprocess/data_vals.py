import pandas as pd
import spacy
import contextualSpellCheck
import re
from pathlib import Path
import os
import multiprocessing
from functools import partial
from tqdm import tqdm

def analyze_text(text, nlp, informal_terms, ambiguous_words):
    """Analyze a single text entry."""
    if not isinstance(text, str) or not text:
        return 0, 0, 0
    
    # Process with spaCy for misspellings
    doc = nlp(text[:10000])  # Limit text length to avoid memory issues
    misspelled = len(doc._.suggestions_spellCheck) if hasattr(doc._, 'suggestions_spellCheck') else 0
    
    # Simple regex for word matching - faster than spaCy tokenization for this purpose
    words = set(re.findall(r'\b\w+\b', text.lower()))
    
    # Set intersections are very fast operations
    informal_count = len(words & informal_terms)
    ambiguity_count = len(words & ambiguous_words)
    
    return misspelled, informal_count, ambiguity_count

def process_batch(batch, nlp, informal_terms, ambiguous_words):
    """Process a batch of texts."""
    results = []
    for text in batch:
        results.append(analyze_text(text, nlp, informal_terms, ambiguous_words))
    return results

def main():
    # Project paths
    project_root = Path(__file__).resolve().parents[3]
    raw_dataset = os.path.join(project_root, "data", "raw", "raw_customer_support_dataset.json")
    output_path = os.path.join(project_root, "data", "output", "customer_support_analysis.json")
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Load spaCy model with optimized settings
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])  # Disable components we don't need
    contextualSpellCheck.add_to_pipe(nlp)
    
    # Pre-compile sets for faster lookups
    informal_terms = {
        'lol', 'omg', 'btw', 'idk', 'smh', 'brb', 'lmao', 'rofl', 'wtf', 'bff', 'tbh',
        'gr8', 'thx', 'u', 'ur', 'pls', 'plz', 'k', 'ya', 'cya', 'luv', 'omfg', 'nvm'
    }
    
    ambiguous_words = {
        'bank', 'bat', 'lead', 'tear', 'bark', 'light', 'match', 'right', 'rock', 'spring',
        'current', 'date', 'kind', 'mean', 'well', 'fine', 'fair', 'case', 'left', 'object'
    }
    
    print("Loading dataset...")
    # Read only necessary columns and use chunksize for large datasets
    df = pd.read_json(raw_dataset, lines=True)
    df = df.sample(1000)  # Sample 1000 rows for analysis
    
    texts = df['output'].tolist()
    
    # Determine optimal batch size and number of processes
    num_cpu = multiprocessing.cpu_count()
    batch_size = max(1, len(texts) // (num_cpu * 2))  # Adjust for your specific machine
    
    # Create batches for parallel processing
    batches = [texts[i:i + batch_size] for i in range(0, len(texts), batch_size)]
    
    print(f"Processing {len(texts)} texts using {num_cpu} processes with batch size {batch_size}...")
    
    # Process in parallel
    with multiprocessing.Pool(processes=num_cpu) as pool:
        process_func = partial(process_batch, nlp=nlp, informal_terms=informal_terms, ambiguous_words=ambiguous_words)
        all_results = list(tqdm(pool.imap(process_func, batches), total=len(batches)))
    
    # Flatten results
    results = [item for sublist in all_results for item in sublist]
    
    # Add results to DataFrame
    misspellings, informal_counts, ambiguity_counts = zip(*results)
    df['misspellings'] = misspellings
    df['informal_terms'] = informal_counts
    df['ambiguous_words'] = ambiguity_counts
    
    # Display summary
    print("\nAnalysis Complete:")
    print(f"Average misspellings per text: {df['misspellings'].mean():.2f}")
    print(f"Average informal terms per text: {df['informal_terms'].mean():.2f}")
    print(f"Average ambiguous words per text: {df['ambiguous_words'].mean():.2f}")
    
    # Display the first few rows of the updated DataFrame
    print("\nFirst few rows:")
    print(df.head())
    
    # Save the updated DataFrame to a new JSON file
    print(f"\nSaving results to {output_path}")
    df.to_json(output_path, orient='records', lines=True)
    print("Done!")

if __name__ == "__main__":
    main()