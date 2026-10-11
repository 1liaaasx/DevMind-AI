# DevMind AI — Rapport final de stabilisation

> Ce rapport conserve les résultats de la phase de stabilisation locale. La
> préparation multi-utilisateur et les 55 tests actuels sont documentés dans
> [DEPLOYMENT_READINESS_REPORT.md](DEPLOYMENT_READINESS_REPORT.md).

Date : **11 octobre 2026**  
Modèle : **Groq `openai/gpt-oss-120b`**

## 1. Problèmes initiaux

| Problème | Statut initial |
|---|---|
| Réponses longues impossibles à lire entièrement | FAIL |
| Bouton GitHub « Expliquer l’architecture » perçu comme bloqué | FAIL |
| But du dépôt Packt confondu avec un exercice d’EDA bancaire | FAIL |
| Markdown affiché littéralement (`**texte**`) | FAIL |
| Sources récupérées mais non nécessairement citées | PARTIAL |
| Premier chargement du modèle sans anticipation | PARTIAL |

## 2. Causes racines

1. **Scroll** — le CSS imposait `overflow:hidden` à `html`, au conteneur
   Streamlit principal et au panneau central, puis enfermait les messages dans
   une zone de hauteur fixe. Une longue réponse dépendait donc d’un petit scroll
   imbriqué difficile à atteindre.
2. **Architecture GitHub** — le premier clic pouvait cumuler le chargement local
   de Sentence Transformers (jusqu’à environ une minute sur CPU), la recherche
   et Groq dans une exécution synchrone. L’interface affichait un statut, mais le
   cold start donnait encore une impression de gel.
3. **Mauvais but du dépôt** — pour la requête exacte « le but principale de ce
   repo », Chroma classait `LICENSE` puis un prompt d’EDA en tête. Le README
   arrivait 8ᵉ avec une distance de `0,7553`, juste au-delà du seuil `0,75`, et
   disparaissait du contexte.
4. **Markdown brut** — la réponse était échappée puis injectée dans une balise
   `<p>` avec des `<br>`. Elle ne passait jamais par le moteur Markdown de
   Streamlit.
5. **Sources trop larges** — tous les passages transmis à Groq étaient affichés,
   même lorsque la réponse n’en citait qu’un sous-ensemble.

## 3. Correctifs implémentés

- le workspace central est l’unique scroll du chat sur desktop ; la zone de
  messages grandit naturellement et le composeur reste atteignable ;
- les panneaux navigation et sources gardent leur hauteur et ne disparaissent
  plus au bas d’une réponse ;
- les réponses utilisent `st.markdown` et `st.code`, sans HTML non sûr ;
- le modèle d’embeddings est préchargé après le rendu initial, partagé au niveau
  processus et protégé par verrou ;
- les requêtes GitHub globales ajoutent des preuves structurelles strictement
  filtrées par le `document_hash` actif ;
- un reranking léger privilégie README, documentation, manifests, points
  d’entrée et code, et abaisse les licences/exemples hors sujet ;
- les chunks identiques sont dédupliqués ;
- les citations sont normalisées, validées, filtrées et renumérotées ;
- le panneau affiche le vrai nom du fichier, le repository, le chemin, l’extrait
  et un lien GitHub valide ;
- Groq utilise un timeout de 25 s et un seul retry SDK avec backoff exponentiel ;
- une troncature par limite de génération est explicitement signalée ;
- les erreurs inattendues sont journalisées côté serveur sans traceback dans
  l’interface ;
- les soumissions RAG simultanées sont bloquées par état de session.

## 4. Fichiers modifiés ou créés

### Application

- `streamlit_app.py`
- `styles.css`
- `devmind_rag/embeddings.py`
- `devmind_rag/vector_store.py`
- `devmind_rag/service.py`
- `devmind_rag/github.py`
- `devmind_rag/groq.py`

### Tests et validation

- `tests/test_rag_service.py`
- `tests/test_groq.py`
- `tests/test_streamlit_ui.py`
- `scripts/run_real_validation.py`
- `TEST_REPORT.md`
- `FINAL_VALIDATION_REPORT.md`

### Documentation et preuves visuelles

- `README.md`
- `ARCHITECTURE.md`
- `assets/devmind-final-github.jpg`
- `assets/devmind-final-mobile.jpg`
- `assets/devmind-final-code.jpg`

## 5. Qualité RAG

| Vérification | Statut | Preuve |
|---|---|---|
| Faits exacts documentaires | PASS | PostgreSQL et 250 événements/minute |
| Recherche sémantique | PASS | passage Atlas retrouvé |
| Isolation multi-document | PASS | Atlas/Orion non mélangés |
| Filtre repository | PASS | toutes les sources du dépôt actif |
| Déduplication | PASS | seconde indexation sans nouveau chunk |
| Suppression ciblée | PASS | Atlas supprimé, Orion conservé |
| Information absente | PASS | refus explicite, aucun revenu inventé |
| Attribution | PASS | citations valides et sources renumérotées |

Le prompt traite les fichiers récupérés comme des données non fiables et interdit
d’en suivre les instructions. Cette défense réduit les injections indirectes,
mais aucune technique de prompting ne les élimine totalement.

## 6. Explorateur GitHub

Repository réel :
`PacktPublishing/Prompt-Engineering-in-Python-with-GPT-and-the-OpenAI-API`.

| Contrôle | Statut |
|---|---|
| Réutilisation d’un dépôt déjà indexé | PASS |
| Pas de contamination entre repositories | PASS |
| Question exacte « Le but principal de ce repo » | PASS |
| Réponse fondée sur `README.md` | PASS |
| Bouton « Expliquer l’architecture » | PASS |
| Liens vers de vrais fichiers GitHub | PASS |
| Aucun code téléchargé exécuté | PASS |
| Timeouts et tailles bornées | PASS |

La réponse réelle décrit correctement le dépôt comme le support du cours/ouvrage
de prompt engineering et cite uniquement `README.md`. La conclusion erronée sur
le churn bancaire n’est plus reproduite.

## 7. Assistant code

| Mode | Exemple | Statut |
|---|---|---|
| Expliquer | factorielle récursive | PASS |
| Analyser une erreur | `IndexError` sur `numbers[5]` | PASS |
| Améliorer | boucle transformée en compréhension | PASS |

Ces appels passent directement par Groq et ne retournent aucune source ChromaDB.
Le test navigateur a confirmé le rendu des titres, tableaux et blocs de code.

## 8. UI/UX et scroll

| Viewport | Scroll long | Pas d’overflow horizontal | Statut |
|---|---|---|---|
| 1920×1080 | oui | oui | PASS |
| 1440×900 | oui | oui | PASS |
| 1366×768 | oui | oui | PASS |
| 480×800 | empilement naturel | oui | PASS |

Mesures DOM sur la réponse longue : panneau central `overflow-y:auto`, zone de
messages `overflow-y:visible`, hauteur de conversation supérieure au viewport,
composeur atteignable et panneaux latéraux visibles. Une conversation de dix
messages avec listes et blocs de code est aussi couverte par `AppTest`.

## 9. Tests Groq réels

La campagne contrôlée a exécuté 9 appels et les tests navigateur 3 appels, soit
**12 appels Groq réels au total**.

| Scénario | Statut |
|---|---|
| Connectivité de base | PASS |
| Document RAG | PASS |
| Information absente | PASS |
| Multi-document | PASS |
| But du dépôt Packt | PASS |
| Architecture GitHub dans le navigateur | PASS |
| Explication de code | PASS |
| Débogage | PASS |
| Amélioration | PASS |

Une assertion textuelle du runner attendait initialement « factorielle » alors
que la réponse correcte employait « factoriel ». Le matcher a été corrigé sans
modifier la réponse ni relancer inutilement la campagne.

## 10. Tests automatisés

```text
48 passed in 128.80s
```

Statut : **PASS**. Aucun test n’a été supprimé ou affaibli. Des régressions ont
été ajoutées pour le but d’un repository, le filtrage des citations, la longueur
Groq et dix messages Markdown/code.

## 11. Tests navigateur

- parcours accueil → GitHub → repository → architecture : PASS ;
- réponse longue lisible du début à la fin : PASS ;
- composeur accessible après la réponse : PASS ;
- sources contextualisées et cliquables : PASS ;
- formulaire Assistant code → réponse Groq : PASS ;
- responsive aux quatre tailles demandées : PASS ;
- erreur serveur actuelle : aucune ;
- erreurs console actuelles : aucune erreur applicative ; les seuls événements
  relevés correspondent à la reconnexion pendant le redémarrage contrôlé.

Statut : **PASS**.

## 12. Performances observées

| Mesure | Temps |
|---|---:|
| Cold start Sentence Transformers | 60,405 s |
| Petit document après chargement | 0,430 s |
| Recherche sémantique | 0,0644 s |
| Import/indexation GitHub | 14,628 s |
| Groq (campagne) | 0,320 à 3,112 s |
| Architecture dans le navigateur | environ 24 à 56 s |

Le cold start local reste le principal coût. Il est maintenant anticipé en
arrière-plan et partagé. Le bouton GitHub reste utilisable pendant l’attente et
retourne une réponse ou une erreur bornée au lieu de bloquer indéfiniment.

## 13. Sécurité et confidentialité

| Contrôle | Statut |
|---|---|
| `.env` ignoré | PASS |
| `.devmind/` ignoré | PASS |
| Clé absente des fichiers contrôlés hors `.env` | PASS (0 correspondance) |
| Contenu GitHub jamais exécuté | PASS |
| Requêtes réseau bornées | PASS |
| HTML du modèle non exécuté | PASS |
| Extraits envoyés à Groq signalés dans l’UI | PASS |
| Répertoire temporaire de campagne retiré | PASS (placé dans la Corbeille) |

## 14. Limites restantes

- les PDF scannés nécessitent un OCR externe ;
- le premier chargement CPU du modèle local peut durer environ une minute ;
- l’historique disparaît à la fin de la session Streamlit ;
- ChromaDB reste local et mono-utilisateur ;
- l’API GitHub publique peut limiter les imports sans authentification ;
- les réponses Groq ne sont pas streamées ;
- la défense contre les injections indirectes est robuste mais non absolue ;
- le dossier fourni n’est pas un dépôt Git : aucun commit/push ni validation de
  statut Git n’a pu être effectué.

## 15. État de préparation final

**PASS — prêt pour un portfolio GitHub et une démonstration locale contrôlée.**

Les bugs critiques de scroll, blocage perçu, contexte GitHub, Markdown et sources
sont corrigés et testés. Aucun bloqueur critique n’est connu. Une production
multi-utilisateur demanderait encore authentification, isolation des données,
persistance de l’historique, OCR et politique de rétention.
