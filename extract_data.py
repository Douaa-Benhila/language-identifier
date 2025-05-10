import os

def split_train_file(input_file, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    with open(input_file, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.rsplit('\t', 1)  # Séparer la phrase de la langue
            if len(parts) != 2:
                continue  # Ignorer les lignes mal formatées
            phrase, lang = parts
            lang = lang.strip()
            
            output_file = os.path.join(output_folder, f"{lang}.txt")
            with open(output_file, 'a', encoding='utf-8') as out_file:
                out_file.write(phrase.strip() + "\n")
                
if __name__ == "__main__":
    split_train_file("train.txt", "language_files2")
