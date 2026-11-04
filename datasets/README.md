# Datasets

No large dataset is committed. The core course uses small datasets bundled with scikit-learn or data generated deterministically in `src/learning_hub/preprocessing.py`. The sentiment lesson includes a tiny, hand-written example corpus inside its notebook.

The historical root NLP notebooks load Reuters-21578 through NLTK at runtime. The corpus is not included, its notebook outputs have been cleared, and NLTK marks it as research-use-only; review the [NLTK dataset license notice](https://github.com/nltk/nltk_data/blob/gh-pages/DATASET-LICENSES.md) before downloading or using it.

When adding data, prefer a documented loader or a link to the authoritative source. Record license, source, retrieval date if relevant, expected size and checksum. Exclude personal data, secrets, trained weights and large files; update `.gitignore` where appropriate.
