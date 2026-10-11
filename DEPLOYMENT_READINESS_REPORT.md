# Rapport de préparation au déploiement

Date de validation : 11 octobre 2026  
Cible : Streamlit Community Cloud  
Point d'entrée : `streamlit_app.py`  
Python retenu : 3.12

## Verdict

**Prêt pour publication GitHub puis déploiement contrôlé**, sous réserve de deux
actions externes qui n'ont volontairement pas été effectuées : créer/publier le
repository GitHub et configurer `GROQ_API_KEY` dans Streamlit Secrets.

Le déploiement Cloud lui-même n'a pas été exécuté. La compatibilité a été validée
localement et par résolution des roues Linux CPython 3.12, mais les ressources
réelles d'une instance Community Cloud ne peuvent être confirmées qu'après le
premier déploiement.

## Fichiers créés

- `devmind_rag/runtime.py` : namespace temporaire de processus, nettoyage borné
  et noms de collection propres aux sessions ;
- `requirements-dev.txt` : dépendances de test séparées de la production ;
- `tests/test_deployment.py` : contrôles de stockage temporaire, nettoyage et
  absence de secret publiable ;
- `DEPLOYMENT.md` : procédure GitHub, Secrets, logs, limites et validation live ;
- `DEPLOYMENT_READINESS_REPORT.md` : présent rapport.

## Fichiers modifiés

- `streamlit_app.py` ;
- `devmind_rag/service.py` ;
- `devmind_rag/vector_store.py` ;
- `devmind_rag/groq.py` ;
- `tests/test_rag_service.py` ;
- `tests/test_streamlit_ui.py` ;
- `requirements.txt` ;
- `.streamlit/config.toml` ;
- `.gitignore` ;
- `.env.example` ;
- `README.md` ;
- `ARCHITECTURE.md`.

## Compatibilité Cloud

| Contrôle | Statut | Résultat |
|---|---|---|
| Point d'entrée racine | PASS | `streamlit_app.py` |
| Python | PASS | développement et cible en 3.12 |
| Streamlit | PASS | 1.65.0, dernière version PyPI vérifiée |
| Dépendances de production | PASS | versions épinglées, `pytest` déplacé en développement |
| Résolution Linux CPython 3.12 | PASS | dry-run pip avec roues manylinux terminé sans erreur |
| Dépendances locales | PASS | `pip check` : aucune dépendance cassée |
| Chemins Windows absolus | PASS | aucun chemin Windows dans les sources publiables |
| Configuration Streamlit | PASS | upload borné à 8 Mo, mode headless, watcher désactivé |
| Démarrage local | PASS | serveur sur le port 8501, endpoint santé HTTP 200 |
| Vérification navigateur | PASS | accueil, Documents et Paramètres rendus sans erreur console |
| Déploiement Community Cloud réel | NON EXÉCUTÉ | nécessite GitHub, compte Streamlit et secret utilisateur |

## Résultats de sécurité

### Problème initial critique

Le service RAG était placé dans `st.cache_resource` avec une collection ChromaDB
commune dans `.devmind/chroma`. Une application publique aurait donc pu afficher
ou rechercher les sources ajoutées par une autre session.

### Stratégie appliquée

- un `RAGService` est stocké uniquement dans `st.session_state` ;
- chaque session reçoit un nom de collection ChromaDB UUID distinct ;
- les collections partagent seulement les fichiers du client Chroma du processus,
  jamais leur espace logique ni leurs recherches ;
- conversations, filtre documentaire, repository sélectionné et résultats de code
  restent dans la session Streamlit ;
- seul le modèle Sentence Transformers, immuable, est partagé globalement ;
- le serveur redémarré crée un nouveau namespace temporaire et ne rouvre pas les
  anciennes collections ;
- les anciens namespaces expirés sont supprimés avec un ciblage limité au préfixe
  DevMind ;
- un contrôle utilisateur permet d'effacer immédiatement toutes les données de la
  session active sans supprimer celles des autres sessions.

Les tests automatisés créent deux collections dans le même répertoire Chroma,
indexent un document privé dans la première, puis vérifient que la seconde ne peut
ni le lister ni le retrouver. Un second test vérifie que l'effacement d'une
collection ne touche pas l'autre.

### Secrets et publication

- `GROQ_API_KEY` est lu depuis Streamlit Secrets ou l'environnement local ;
- aucune clé n'est codée en dur ;
- `.env`, `.env.*`, `.streamlit/secrets.toml`, `.devmind/`, environnements,
  caches Python et logs sont ignorés ;
- le scan des fichiers publiables n'a trouvé aucun motif de clé Groq `gsk_` ;
- les messages d'erreur utilisateur ne révèlent pas la clé.

## Ressources et performance

- le modèle d'embeddings n'est chargé que sur les pages RAG ;
- une seule instance du modèle est partagée entre les sessions ;
- les documents privés ne sont pas placés dans un cache global ;
- uploads : 8 Mo maximum par fichier et cinq fichiers par opération ;
- index : 1 800 passages maximum par session ;
- GitHub : limites existantes conservées pour l'archive, les fichiers et le texte ;
- Groq : huit demandes par minute par session et trente par minute pour le
  processus ;
- durée Groq : timeout de 25 secondes et un retry maximum ;
- dépendances de test retirées de l'installation de production.

Le principal risque résiduel est l'empreinte mémoire de Torch, Sentence
Transformers et ChromaDB. Le modèle multilingue validé n'a pas été remplacé afin
d'éviter un changement de qualité ou de dimension des embeddings.

## Tests exécutés

```text
pytest -q
55 passed in 45.77s
```

Contrôles complémentaires :

- compilation de `streamlit_app.py` et de tous les modules `devmind_rag` : PASS ;
- `pip check` : PASS ;
- résolution PyPI Linux/manylinux, CPython 3.12, binaires uniquement : PASS ;
- scan de motifs de secrets hors fichiers ignorés : 0 résultat ;
- santé locale `/_stcore/health` : HTTP 200 `ok` ;
- contrôle DOM réel de l'accueil, de **Mes documents** et de **Paramètres** : PASS ;
- erreurs console navigateur : 0.

Aucun nouvel appel réel à Groq n'a été effectué pendant cette préparation. Les
tests utilisent des générateurs déterministes ; l'intégration Groq réelle avait
déjà été documentée dans `FINAL_VALIDATION_REPORT.md` avant cette phase.

## Limites restantes

- aucune authentification : la séparation est par session navigateur, pas par compte ;
- aucune conservation durable ni restauration après redémarrage ;
- un onglet fermé ne déclenche pas de hook fiable de suppression immédiate côté
  Streamlit ; la collection devient inaccessible et le namespace temporaire est
  éliminé après expiration ;
- l'API GitHub publique non authentifiée peut limiter les imports ;
- un PDF scanné requiert toujours un OCR ;
- la latence et la mémoire réelles de Community Cloud restent à observer ;
- l'application peut être mise en veille par Community Cloud lorsqu'elle est inactive.

## Blocages avant mise en ligne

1. Le dossier n'est pas encore un repository Git.
2. Aucun repository GitHub distant n'a été créé ou configuré.
3. `GROQ_API_KEY` doit être ajouté dans **Advanced settings > Secrets**.
4. Le premier déploiement doit être surveillé dans les logs pour confirmer la
   mémoire disponible pendant le chargement du modèle.

## Instructions exactes

Suivre [DEPLOYMENT.md](DEPLOYMENT.md). En résumé :

```powershell
git init -b main
git add .
git status --short
git commit -m "Prepare DevMind AI for Streamlit Community Cloud"
git remote add origin https://github.com/VOTRE_COMPTE/VOTRE_REPOSITORY.git
git push -u origin main
```

Dans Streamlit Community Cloud : repository et branche `main`, point d'entrée
`streamlit_app.py`, Python 3.12, puis secret :

```toml
GROQ_API_KEY = "votre_cle_groq"
```

La création du repository distant, le push et le déploiement attendent
l'autorisation explicite de l'utilisateur.
