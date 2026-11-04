# Bien démarrer

## 1. Récupérer le dépôt

```bash
git clone <URL_DU_DEPOT>
cd ml-dl-learning-hub
```

Si vous travaillez déjà dans une copie locale, `cd` à la racine du projet. Gardez les données téléchargées hors du versionnement.

## 2. Créer l’environnement

Le parcours de base utilise Python 3.11 ou 3.12. Avec `venv` :

```bash
python -m venv .venv
# PowerShell : .venv\Scripts\Activate.ps1
# bash/zsh : source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e ".[dev]"
```

Pour Conda, créer un environnement Python 3.11 puis lancer les deux commandes pip ci-dessus. Pour `uv`, `uv venv` puis `uv pip install -e ".[dev]"`.

Groupes optionnels : `.[dl]` pour PyTorch, `.[mlops]` pour MLflow et FastAPI, `.[nlp]` pour les notebooks NLTK, `.[parquet]` pour Arrow/Parquet. Ils peuvent être combinés : `python -m pip install -e ".[dev,dl,mlops,nlp,parquet]"`. Installez PyTorch depuis le [sélecteur officiel](https://pytorch.org/get-started/locally/) si vous avez besoin d’une version CUDA adaptée à votre GPU.

## 3. Ouvrir un notebook

```bash
jupyter lab
```

Ouvrir un fichier `.ipynb`, choisir le noyau de l’environnement `.venv`, puis exécuter les cellules dans l’ordre. Dans Colab, téléverser le notebook ; si le notebook importe `learning_hub`, installer le dépôt ou ajouter son dossier `src` au chemin Python.

Dans VS Code, installer les extensions Python et Jupyter, ouvrir la racine du dépôt et sélectionner l’interpréteur `.venv`. Colab convient si l’installation locale n’est pas souhaitée ; le parcours de base fonctionne sans GPU.

## 4. Git sans risque

```bash
git status
git switch -c votre-sujet
git add chemin/de/vos/fichiers
git commit -m "Ajoute une leçon sur les métriques"
```

Avant de contribuer, examiner `git diff` et ne pas ajouter de données personnelles, de gros fichiers, de poids de modèle, de clé API ou de sortie de notebook volumineuse.

## Environnements limités

- Sans GPU, suivre les mêmes exemples sur CPU ; les exercices du dépôt sont dimensionnés pour des jeux de données légers.
- Si l’installation de PyTorch est trop lourde, les modules 00 à 05 et les projets tabulaires restent disponibles.
- En cas de problème de mémoire, redémarrer le noyau, réduire la taille de lot et supprimer les sorties avant de committer.
- La connexion est nécessaire pour installer les paquets la première fois. Les parcours principaux n’appellent aucune API payante.
