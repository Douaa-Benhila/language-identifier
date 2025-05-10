import os
import math
import re
import unicodedata
from collections import Counter
import pickle
import time
from functools import lru_cache

# Cache pour le prétraitement de texte
@lru_cache(maxsize=10000)
def preprocess_text_cached(line):
    """Version mise en cache du prétraitement de texte"""
    line = line.lower()
    line = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode("utf-8")
    line = re.sub(r"\b\d+\b", " ", line)  
    line = re.sub(r"[^\w\s']", " ", line)  
    line = re.sub(r"\s+", " ", line).strip()
    return line

def preprocess_text(line):
    """Wrapper pour la fonction mise en cache"""
    return preprocess_text_cached(line)

def get_most_frequent_words(corpus_file, threshold=0.02, cache_dir="cache"):
    """
    Identifie les mots les plus fréquents avec mise en cache.
    """
    # Créer le répertoire de cache s'il n'existe pas
    os.makedirs(cache_dir, exist_ok=True)
    
    # Nom du fichier de cache basé sur le nom du fichier corpus et le seuil
    cache_filename = os.path.join(cache_dir, f"{os.path.basename(corpus_file)}_{threshold}_frequent.pkl")
    
    # Vérifier si le cache existe
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                return pickle.load(f)
        except:
            pass  # En cas d'erreur, continuer avec le calcul normal
    
    word_counter = Counter()
    total_words = 0

    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = preprocess_text(line)
            words = line.split()
            word_counter.update(words)
            total_words += len(words)

    # Sélectionner les mots les plus fréquents dépassant le seuil
    frequent_words = {word for word, count in word_counter.items() if count / total_words > threshold}
    
    # Sauvegarder dans le cache
    with open(cache_filename, 'wb') as f:
        pickle.dump(frequent_words, f)
    
    return frequent_words

def generate_ngrams(corpus_file, n, frequent_words=None, cache_dir="cache"):
    """
    Génère des n-grammes avec mise en cache.
    """
    # Créer le répertoire de cache s'il n'existe pas
    os.makedirs(cache_dir, exist_ok=True)
    
    # Nom du fichier de cache
    cache_filename = os.path.join(cache_dir, f"{os.path.basename(corpus_file)}_{n}_ngrams.pkl")
    
    # Vérifier si le cache existe
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                return pickle.load(f)
        except:
            pass  # En cas d'erreur, continuer avec le calcul normal
    
    ngram_counter = Counter()
    
    # Prétraiter l'intégralité du fichier en une seule fois
    all_words = []
    with open(corpus_file, 'r', encoding='utf-8') as file:
        for line in file:
            line = preprocess_text(line)
            words = [word for word in line.split() if frequent_words is None or word not in frequent_words]
            all_words.extend(words)
    
    # Ajouter les marqueurs de début et fin
    all_words = ["<s>"] * (n - 1) + all_words + ["</s>"]
    
    # Générer tous les n-grammes en une seule passe
    for i in range(len(all_words) - n + 1):
        ngram = tuple(all_words[i:i + n])
        ngram_counter[ngram] += 1
    
    # Sauvegarder dans le cache
    with open(cache_filename, 'wb') as f:
        pickle.dump(ngram_counter, f)
    
    return ngram_counter

def cosine_similarity(ngram_counter_1, ngram_counter_2):
    """
    Calcule la similarité cosinus entre deux dictionnaires de n-grammes.
    Version optimisée qui évite les calculs redondants.
    """
    # Pré-calculer les magnitudes en dehors de la fonction si possible
    if not hasattr(ngram_counter_2, '_magnitude'):
        ngram_counter_2._magnitude = math.sqrt(sum(count ** 2 for count in ngram_counter_2.values()))
    
    magnitude_1 = math.sqrt(sum(count ** 2 for count in ngram_counter_1.values()))
    magnitude_2 = ngram_counter_2._magnitude
    
    if magnitude_1 == 0 or magnitude_2 == 0:
        return 0.0
    
    # Utiliser l'intersection des clés pour le produit scalaire
    dot_product = sum(ngram_counter_1[ngram] * ngram_counter_2[ngram] 
                      for ngram in ngram_counter_1 if ngram in ngram_counter_2)
    
    return dot_product / (magnitude_1 * magnitude_2)

def load_language_ngrams(directory, n, cache_dir="cache"):
    """
    Charge les n-grammes pour chaque langue avec mise en cache.
    """
    cache_filename = os.path.join(cache_dir, f"all_languages_{n}_ngrams.pkl")
    os.makedirs(cache_dir, exist_ok=True)
    
    # Vérifier si le cache pour toutes les langues existe
    if os.path.exists(cache_filename):
        try:
            with open(cache_filename, 'rb') as f:
                language_ngrams = pickle.load(f)
                # Pré-calculer les magnitudes pour la similarité cosinus
                for lang, ngrams in language_ngrams.items():
                    ngrams._magnitude = math.sqrt(sum(count ** 2 for count in ngrams.values()))
                return language_ngrams
        except:
            pass  # En cas d'erreur, continuer avec le chargement normal
    
    language_ngrams = {}
    
    for filename in os.listdir(directory):
        if filename.endswith("_cleaned_ngrams.txt"):
            language = filename.split("_")[0]
            file_path = os.path.join(directory, filename)
            
            # Identifier les mots fréquents pour filtrer les biais
            frequent_words = get_most_frequent_words(file_path, cache_dir=cache_dir)
            
            # Générer les n-grammes en excluant ces mots
            language_ngrams[language] = generate_ngrams(file_path, n, frequent_words, cache_dir=cache_dir)
            
            # Pré-calculer la magnitude pour la similarité cosinus
            language_ngrams[language]._magnitude = math.sqrt(sum(count ** 2 for count in language_ngrams[language].values()))
    
    # Sauvegarder dans le cache
    with open(cache_filename, 'wb') as f:
        pickle.dump(language_ngrams, f)
    
    return language_ngrams

def generate_ngrams_from_sentence(phrase, n, frequent_words=None):
    """
    Génère les n-grammes d'une phrase.
    """
    words = preprocess_text(phrase).split()
    words = [word for word in words if frequent_words is None or word not in frequent_words]
    words = ["<s>"] * (n - 1) + words + ["</s>"]
    return Counter(tuple(words[i:i + n]) for i in range(len(words) - n + 1))

def detect_language(phrase, language_ngrams, n):
    """
    Détecte la langue d'une phrase.
    Version optimisée qui utilise un tri partiel pour trouver la meilleure correspondance.
    """
    phrase_ngrams = generate_ngrams_from_sentence(phrase, n)
    
    # Utiliser une approche en deux phases: d'abord un filtrage rapide, puis un calcul précis
    # si la phrase est très courte
    if len(phrase_ngrams) <= 5:
        # Pour les phrases très courtes, faire un calcul complet
        similarities = [(language, cosine_similarity(phrase_ngrams, ngram_counter)) 
                       for language, ngram_counter in language_ngrams.items()]
        return max(similarities, key=lambda x: x[1])[0]
    else:
        # Pour les phrases plus longues, trouver le top 3 des langues d'abord
        top_langs = {}
        for language, ngram_counter in language_ngrams.items():
            # Estimation rapide: compter les n-grammes communs
            common_ngrams = sum(1 for ngram in phrase_ngrams if ngram in ngram_counter)
            top_langs[language] = common_ngrams
        
        # Sélectionner les 3 meilleures langues candidates
        candidates = sorted(top_langs.items(), key=lambda x: x[1], reverse=True)[:3]
        
        # Calculer la similarité cosinus précise uniquement pour ces candidates
        best_similarity = -1
        detected_language = None
        
        for language, _ in candidates:
            similarity = cosine_similarity(phrase_ngrams, language_ngrams[language])
            if similarity > best_similarity:
                best_similarity = similarity
                detected_language = language
        
        return detected_language

def process_file(input_file, language_ngrams, n, batch_size=50):
    """
    Traite un fichier d'entrée par lots pour une meilleure efficacité.
    """
    start_time = time.time()
    modify = False
    all_lines = []
    results = []
    
    # Charger toutes les lignes en mémoire
    with open(input_file, 'r', encoding='utf-8') as file:
        all_lines = file.readlines()
    
    # Traiter par lots
    for i in range(0, len(all_lines), batch_size):
        batch = all_lines[i:i+batch_size]
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
                #print(f"Phrase: {phrase}")
                #print(f"True Language: {true_language}, Detected Language: {detected_language}")
                #print("=" * 50)
        
        results.extend(batch_results)
        
        # Afficher l'avancement
        progress = min(100, int((i + len(batch)) / len(all_lines) * 100))
        elapsed = time.time() - start_time
        estimated_total = elapsed / progress * 100 if progress > 0 else 0
        remaining = estimated_total - elapsed
        print(f"Progression: {progress}% - Temps écoulé: {elapsed:.2f}s - Temps restant estimé: {remaining:.2f}s")
    
    if modify:
        with open(input_file, 'w', encoding='utf-8') as file:
            file.writelines(results)
    else :
        with open("test_results.txt", 'w', encoding='utf-8') as file:
            file.writelines(results)
    
    total_time = time.time() - start_time
    print(f"Traitement terminé en {total_time:.2f} secondes")

# Point d'entrée principal
def main():
    start_time = time.time()
    
    # Utilisation du programme
    language_files_directory = "language_files2"
    input_file = "test2.txt"
    n = 3
    
    print("Chargement des n-grammes...")
    language_ngrams = load_language_ngrams(language_files_directory, n)
    print(f"Chargement terminé en {time.time() - start_time:.2f} secondes")
    
    # Traiter le fichier d'entrée
    process_file(input_file, language_ngrams, n)
    
    total_time = time.time() - start_time
    print(f"Temps total d'exécution: {total_time:.2f} secondes")

if __name__ == "__main__":
    main()