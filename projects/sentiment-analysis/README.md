# Projet — Sentiment sur de courts avis

Le notebook [sentiment avec TF-IDF](../../modules/09-nlp/01_sentiment_tfidf.ipynb) utilise un minuscule corpus écrit pour enseigner le protocole. Il n’est pas représentatif d’avis clients réels et ses métriques sont instables.

## Étapes

1. Formuler la tâche binaire et identifier les erreurs importantes.
2. Inspecter textes, classes, répétitions et longueur des exemples.
3. Séparer stratifié avant d’ajuster le vocabulaire.
4. Comparer `TfidfVectorizer` uni-grammes au couple uni/bigrammes en CV train.
5. Évaluer sur test une fois, examiner les erreurs et les mots dominants.
6. Documenter ambiguïté, sarcasme, domaine, langue et faible taille.

Les fichiers NLP préexistants dans la racine enseignent aussi prétraitement, vectorisation et word embeddings sur Reuters via NLTK. Leur téléchargement de corpus dépend de la disponibilité NLTK et ils ne sont pas requis pour exécuter l’exemple TF-IDF.
