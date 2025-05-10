import os
import re
import unicodedata

def preprocess_text(line):
    # Convertir en minuscules
    line = line.lower()
    
    # Normalisation Unicode (décompose les accents)
    line = unicodedata.normalize("NFKD", line).encode("ascii", "ignore").decode("utf-8")

    # Suppression des nombres uniquement s'ils sont isolés (évite de supprimer les dates importantes)
    line = re.sub(r"\b\d+\b", " ", line)  

    # Remplacement de la ponctuation par des espaces (sauf l'apostrophe)
    line = re.sub(r"[^\w\s']", " ", line)  

    # Suppression des espaces multiples
    line = re.sub(r"\s+", " ", line).strip()

    return line

# Fonction pour prétraiter tous les fichiers .txt dans un dossier
def preprocess_files(directory):
    for filename in os.listdir(directory):
        if filename.endswith(".txt") and not filename.endswith("_cleaned.txt"):  
            file_path = os.path.join(directory, filename)
            cleaned_file_path = os.path.join(directory, filename.replace(".txt", "_cleaned.txt"))
            
            with open(file_path, "r", encoding="utf-8") as infile, open(cleaned_file_path, "w", encoding="utf-8") as outfile:
                for line in infile:
                    cleaned_line = preprocess_text(line)
                    if cleaned_line:  # Ne pas écrire de lignes vides
                        outfile.write(cleaned_line + "\n")
            
            print(f"Fichier pré-traité : {cleaned_file_path}")

# Exécution du programme
language_files_directory = "language_files2"
preprocess_files(language_files_directory)
