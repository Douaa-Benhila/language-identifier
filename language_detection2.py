import os
import math
import re
import unicodedata
from collections import Counter
import time

def preprocess_text(line):
    """Nettoyage de texte : minuscule, suppression des accents, des nombres isolés et de la ponctuation."""
    line = line.lower()
    line = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode("utf-8")
    line = re.sub(r"\b\d+\b", " ", line)
    line = re.sub(r"[^\w\s']", " ", line)
    line = re.sub(r"\s+", " ", line).strip()
    return line

def get_most_frequent_words(corpus_file, threshold=0.02):
    """Identifie les mots les plus fréquents dans un corpus."""
    word_counter = Counter()
    total_words = 0

    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = preprocess_text(line)
            words = line.split()
            word_counter.update(words)
            total_words += len(words)

    frequent_words = {word for word, count in word_counter.items() if count / total_words > threshold}
    return frequent_words

def generate_ngrams(corpus_file, n, frequent_words=None):
    """Génère des n-grammes à partir d'un fichier corpus."""
    ngram_counter = Counter()

    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = preprocess_text(line)
            words = [word for word in line.split() if frequent_words is None or word not in frequent_words]
            words = ["<s>"] * (n - 1) + words + ["</s>"]
            for i in range(len(words) - n + 1):
                ngram = tuple(words[i:i + n])
                ngram_counter[ngram] += 1

    # Pré-calculer la magnitude pour la similarité cosinus
    ngram_counter._magnitude = math.sqrt(sum(count ** 2 for count in ngram_counter.values()))
    return ngram_counter

def cosine_similarity(ngram_counter_1, ngram_counter_2):
    """Calcule la similarité cosinus entre deux dictionnaires de n-grammes."""
    if not hasattr(ngram_counter_2, '_magnitude'):
        ngram_counter_2._magnitude = math.sqrt(sum(count ** 2 for count in ngram_counter_2.values()))

    magnitude_1 = math.sqrt(sum(count ** 2 for count in ngram_counter_1.values()))
    magnitude_2 = ngram_counter_2._magnitude

    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0.0

    dot_product = sum(ngram_counter_1[ngram] * ngram_counter_2[ngram]
                      for ngram in ngram_counter_1 if ngram in ngram_counter_2)
    return dot_product / (magnitude_1 * magnitude_2)

def load_language_ngrams(directory, n):
    """Charge les n-grammes pour chaque langue à partir des fichiers dans un répertoire."""
    language_ngrams = {}

    for filename in os.listdir(directory):
        if filename.endswith("_cleaned_ngrams.txt"):
            language = filename.split("_")[0]
            file_path = os.path.join(directory, filename)
            frequent_words = get_most_frequent_words(file_path)
            language_ngrams[language] = generate_ngrams(file_path, n, frequent_words)

    return language_ngrams

def generate_ngrams_from_sentence(phrase, n, frequent_words=None):
    """Génère les n-grammes d'une phrase donnée."""
    words = preprocess_text(phrase).split()
    words = [word for word in words if frequent_words is None or word not in frequent_words]
    words = ["<s>"] * (n - 1) + words + ["</s>"]
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))

def detect_language(phrase, language_ngrams, n):
    """Détecte la langue d'une phrase avec une approche en deux phases."""
    phrase_ngrams = generate_ngrams_from_sentence(phrase, n)

    if len(phrase_ngrams) <= 5:
        similarities = [(language, cosine_similarity(phrase_ngrams, ngram_counter))
                        for language, ngram_counter in language_ngrams.items()]
        return max(similarities, key=lambda x: x[1])[0]
    else:
        top_langs = {}
        for language, ngram_counter in language_ngrams.items():
            common_ngrams = sum(1 for ngram in phrase_ngrams if ngram in ngram_counter)
            top_langs[language] = common_ngrams

        candidates = sorted(top_langs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        best_similarity = -1
        detected_language = None
        for language, _ in candidates:
            similarity = cosine_similarity(phrase_ngrams, language_ngrams[language])
            if similarity > best_similarity:
                best_similarity = similarity
                detected_language = language

        return detected_language

def process_file(input_file, language_ngrams, n, batch_size=50):
    """Traite un fichier d'entrée par lots."""
    start_time = time.time()
    modify = False
    all_lines = []

    with open(input_file, 'r', encoding='utf-8') as file:
        all_lines = file.readlines()

    results = []
    for i in range(0, len(all_lines), batch_size):
        batch = all_lines[i:i + batch_size]
        batch_results = []

        for line in batch:
            parts = line.split("\t")
            if len(parts) != 2:
                print(f"Skipping malformed line: {line.strip()}")
                batch_results.append(line)
                continue

            phrase, true_language = parts[0], parts[1].strip()
            detected_language = detect_language(phrase, language_ngrams, n)

            if true_language == "??":
                batch_results.append(f"{phrase}\t{detected_language}\n")
                modify = True
            else:
                batch_results.append(line)

        results.extend(batch_results)
        
        progress = min(100, int((i + len(batch)) / len(all_lines) * 100))
        elapsed = time.time() - start_time
        estimated_total = elapsed / progress * 100 if progress > 0 else 0
        remaining = estimated_total - elapsed
        print(f"Progression: {progress}% - Temps écoulé: {elapsed:.2f}s - Temps restant estimé: {remaining:.2f}s")

    if modify:
        with open(input_file, 'w', encoding='utf-8') as file:
            file.writelines(results)
    else :
        with open("dev_results2.txt", 'w', encoding='utf-8') as file:
            file.writelines(results)

    total_time = time.time() - start_time
    print(f"Traitement du fichier terminé en {total_time:.2f} secondes")
    return total_time

def main():
    # Enregistrer le temps de début global
    program_start_time = time.time()
    
    language_files_directory = "language_files2"
    input_file = "dev.txt"
    n = 3

    print("Chargement des n-grammes...")
    load_start_time = time.time()
    language_ngrams = load_language_ngrams(language_files_directory, n)
    load_time = time.time() - load_start_time
    print(f"Chargement des n-grammes terminé en {load_time:.2f} secondes")

    print("Traitement du fichier...")
    process_time = process_file(input_file, language_ngrams, n)
    
    # Calculer et afficher le temps total
    total_time = time.time() - program_start_time
    print("\nRésumé des temps d'exécution :")
    print(f"- Chargement des n-grammes : {load_time:.2f} secondes")
    print(f"- Traitement du fichier : {process_time:.2f} secondes")
    print(f"Temps total d'exécution : {total_time:.2f} secondes")

if __name__ == "__main__":
    main()