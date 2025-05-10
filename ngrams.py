import os
from collections import Counter

def generate_ngrams(corpus_file, n):
    ngram_counter = Counter()
    
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            # Tokenization: split on spaces
            words = line.strip().split()
            
            # Add n-1 start markers and one end marker
            words = ["<s>"] + words + ["</s>"]
            
            # Extract n-grams
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                ngram_counter[ngram] += 1
    
    return ngram_counter



def process_language_files(directory, n):
    # Charger les mots les plus fréquents pour chaque langue
    language_frequent_words = {}
    for filename in os.listdir(language_files_directory):
        if filename.endswith("_cleaned_ngrams.txt"):
            language = filename.split("_")[0]
            file_path = os.path.join(language_files_directory, filename)
            language_frequent_words[language] = get_most_frequent_words(file_path)

            print(f"Processing file: {filename}")
            
            # Generate n-grams for the current file
            ngram_counter = generate_ngrams(file_path, n)
            
            # Create an output file for each input file
            output_filename = f"{os.path.splitext(filename)[0]}_ngrams.txt"
            output_path = os.path.join(directory, output_filename)
            
            with open(output_path, 'w', encoding='utf-8') as output_file:
                # Write n-grams and their counts to the output file
                for ngram, count in ngram_counter.items():
                    output_file.write(" ".join(ngram) + " " + str(count) + "\n")
            
            print(f"Output written to: {output_filename}")

# Usage
language_files_directory = "language_files2"  # Replace with the correct path
n = 3  # Example n-gram size, change as needed
process_language_files(language_files_directory, n)
