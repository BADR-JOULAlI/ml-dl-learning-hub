# Module 09 — Traitement automatique des langues

**Niveau :** intermédiaire à avancé · **Durée :** 6–8 h · **Prérequis :** modules 03–07, groupe `nlp` pour le matériel NLTK hérité.

## Objectifs

Définir une tâche NLP avant le nettoyage ; construire un baseline Bag of Words ou TF-IDF ; expliquer pourquoi tokenisation et vocabulaire se fit uniquement sur train ; comparer un modèle linéaire à des embeddings ; suivre formes et masques d’une séquence ; décrire l’idée de self-attention.

TF-IDF donne à un terme fréquent dans un document mais rare dans le corpus un poids plus grand. Les embeddings sont des vecteurs appris pour des tokens. RNN/LSTM traitent les séquences de manière récurrente ; Transformer utilise attention et position pour mixer les tokens en parallèle.

## Activité

Démarrer avec [analyse de sentiment](../../projects/sentiment-analysis/README.md), comparer mots et n-grams avec un pipeline `TfidfVectorizer + LogisticRegression`, puis inspecter faux positifs/négatifs. Une introduction Transformer/Hugging Face sans clé API reste un objectif d’extension.

## Ressources historiques

Les notebooks `TP1_Preprocessing_NLP_solution.ipynb` et `TP2_Vectorisation_Word_Embedding_solution.ipynb`, ainsi que les scripts `solution_tp*_*.py`, sont des supports préexistants conservés à la racine. Les sorties notebook ont été effacées avant publication ; exécuter les cellules peut télécharger des corpus NLTK. En particulier, Reuters-21578 est assorti de conditions de recherche seulement selon la [notice des licences NLTK](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md) : les données ne sont ni incluses ni réhébergées par le dépôt. Vérifier leurs conditions et citation avant usage. Ces supports ne sont pas nécessaires pour le parcours débutant.
