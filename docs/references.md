# Références pédagogiques et réutilisation

Ces ressources ont été consultées pour concevoir un ordre d’apprentissage original. Nous empruntons des idées pédagogiques générales (progression, alternance pratique/théorie, exercices fréquents, projets, boucles d’entraînement explicites), pas leurs textes, notebooks, figures ou code. Le contenu de ce dépôt est rédigé indépendamment.

| Ressource | Apport étudié | Licence/conditions observées dans la source |
|---|---|---|
| [Microsoft — ML for Beginners](https://github.com/microsoft/ML-For-Beginners) | Leçons regroupées par thème, thèmes fil rouge, quiz avant/après, projets graduels. | Le dépôt présente une licence MIT pour le logiciel. Toute réutilisation d’un élément précis doit conserver attribution et avis de licence. |
| [Aurélien Géron — Hands-On ML 3](https://github.com/ageron/handson-ml3) | Notebooks exécutables, progression d’un projet ML, environnements documentés, exercices et checklist de projet. | Licence Apache-2.0 publiée dans le dépôt ; conserver les avis et notices si du contenu couvert était repris. Aucun contenu n’est repris ici. |
| [fastai — fastbook](https://github.com/fastai/fastbook) | Départ par l’application, apprentissage expérimental, allers-retours entre modèle et fondements. | Attention : le dépôt indique que le code est GPL-3.0, mais que le texte (dont les cellules Markdown des notebooks) n’est pas librement redistribuable. Aucun texte, notebook ou figure n’est copié. |
| [Daniel Bourke — PyTorch Deep Learning](https://github.com/mrdbourke/pytorch-deep-learning) | Décomposition claire des étapes d’un workflow PyTorch et supports liés à des projets. | Licence MIT annoncée dans le dépôt. Aucun code ou contenu n’est copié. |
| [DataTalksClub — MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) | Progression tracking → pipeline → déploiement → monitoring → pratiques d’ingénierie → projet. | Aucune licence globale explicite n’a été retenue pour cette référence lors de l’inspection. Seuls l’ordre général et les thèmes sont cités ; aucun fichier ni texte n’est repris. Vérifier les notices des fichiers en amont d’une réutilisation. |

Le statut de licence peut évoluer et les dépôts peuvent contenir des fichiers de tiers avec d’autres conditions. Cette page décrit l’inspection faite pour ce travail ; elle ne remplace pas les notices officielles.

## Application concrète dans ce cours

- Les [leçons Microsoft](https://github.com/microsoft/ML-For-Beginners) associent pratique, quiz et projet. Ici, chaque chapitre de base mène à une expérience que l’on peut modifier, puis à un exercice et une interprétation.
- Les [notebooks de Géron](https://github.com/ageron/handson-ml3) montrent l’intérêt d’un parcours reproductible. Nos exemples fixent une graine, affichent formes et unités, séparent train/validation/test et gardent le preprocessing dans les pipelines.
- [fastbook](https://github.com/fastai/fastbook) place tôt des expériences concrètes avant de revenir aux mécanismes. Notre CNN sur chiffres, notre exemple XOR et les figures de convolution suivent ce va-et-vient, avec des explications et des données originales.
- Le [parcours PyTorch de Daniel Bourke](https://github.com/mrdbourke/pytorch-deep-learning) sépare tensors, workflow, classification, vision et code modulaire. Notre boucle PyTorch expose explicitement forward, loss, gradients, optimizer et évaluation, puis affiche les courbes train/validation.
- Le [MLOps Zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) enchaîne tracking, pipeline, déploiement et monitoring. Notre notebook local montre une table d’expériences, une pipeline sérialisée et la comparaison visuelle entre distribution de référence et lot récent.

Les visualisations sont générées par le code du cours ; leurs titres, axes et paragraphes d’interprétation indiquent à quelle question elles répondent et ce qu’elles ne démontrent pas. Le [guide visuel](pedagogie-visuelle.md) donne la grille de lecture commune.
