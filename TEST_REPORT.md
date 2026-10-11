# Rapport de tests — DevMind AI

Date d’exécution : **11 octobre 2026**.

## Résultats automatisés

Commande :

```powershell
.\.venv\Scripts\python.exe -m pytest -q
```

Résultat final actuel : **55 tests réussis en 45,77 s**, sans échec ni test ignoré.

La suite couvre l’extraction PDF/TXT/Markdown, les documents vides et corrompus,
le chunking, ChromaDB, la déduplication et la suppression, les filtres par
document/repository, le reranking GitHub, la correspondance citations/sources,
les erreurs Groq, l’historique, les principaux parcours Streamlit, l'isolation
de deux sessions ChromaDB, le nettoyage du stockage temporaire et les garde-fous
de publication Cloud.

## Campagne Groq réelle

Le script `scripts/run_real_validation.py` a exécuté **9 appels réels** avec
`openai/gpt-oss-120b` :

- connectivité de base : PASS ;
- deux questions factuelles sur un document contrôlé : PASS ;
- refus d’une information absente : PASS ;
- isolation d’un second document : PASS ;
- régression « Le but principal de ce repo » sur le dépôt Packt : PASS ;
- explication, débogage et amélioration de code : PASS.

La réponse GitHub correcte cite uniquement `README.md`. Les références retournées
ont des chemins et liens GitHub réels. Trois appels navigateur supplémentaires
(deux analyses d’architecture pendant l’itération CSS et une explication de code)
portent la validation complète à **12 appels réels**, dans la limite demandée.

## Mesures observées

| Mesure | Résultat |
|---|---:|
| Chargement à froid du modèle local | 60,405 s |
| Indexation d’un petit document | 0,430 s |
| Recherche sémantique | 0,0644 s |
| Import + indexation du dépôt Packt | 14,628 s |
| Appels Groq de la campagne | 0,320 à 3,112 s |
| Action navigateur « Expliquer l’architecture » | 24 à 56 s selon cold start/retry |

## Validation navigateur

- action GitHub « Expliquer l’architecture » : PASS ;
- réponse Markdown (titres, tableau, listes, code inline) : PASS ;
- panneau de sources avec fichiers réels et liens valides : PASS ;
- assistant code avec réponse Groq réelle : PASS ;
- scroll du chat long et accès au composeur : PASS ;
- viewports 1920×1080, 1440×900, 1366×768 et 480×800 : PASS ;
- débordement horizontal global : absent sur les quatre tailles ;
- logs serveur après redémarrage : aucune exception ;
- console navigateur : seulement les erreurs de reconnexion attendues pendant le
  redémarrage contrôlé du serveur, aucune erreur applicative après reconnexion.

Les détails complets, limites et décisions de release figurent dans
[`FINAL_VALIDATION_REPORT.md`](FINAL_VALIDATION_REPORT.md).
