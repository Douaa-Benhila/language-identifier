# Projet de Classification de Textes – Identification de la Langue

## Description

Ce projet vise à développer un modèle capable de détecter automatiquement la langue d'un texte donné en entrée. Il repose sur l'analyse des n-grammes de mots et utilise des techniques d'apprentissage supervisé pour classer les textes dans différentes langues européennes.

### Objectifs :
- Créer un prototype de classification de texte basé sur l'identification de la langue.
- Utiliser des n-grammes de caractères et de mots pour entraîner un modèle de classification performant.
- Offrir une solution efficace pour le traitement automatique des langues (TAL), la traduction automatique, et la classification de documents multilingues.

## Fonctionnalités

- **Détection automatique de la langue** : À partir d'un texte en entrée, le modèle identifie la langue européenne (par exemple, français, anglais, espagnol).
- **Prétraitement des données** : Nettoyage des textes pour en faciliter l'analyse.
- **Extraction des n-grammes** : Utilisation de trigrammes de mots pour capturer des motifs linguistiques typiques.
- **Entraînement du modèle** : Un modèle est formé sur un corpus multilingue pour prédire la langue avec une grande précision.

## Prérequis

Avant de commencer, assurez-vous d'avoir installé Python 3.x et les bibliothèques suivantes :
- `nltk`
- `numpy`
- `pandas`

Vous pouvez installer ces dépendances avec `pip` :

```bash
pip install langid nltk numpy pandas
