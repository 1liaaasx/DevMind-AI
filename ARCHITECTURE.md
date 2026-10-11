# Architecture de DevMind AI

## Vue d’ensemble

```text
PDF / TXT / Markdown       Repository GitHub public
        │                            │
        └──────────────┬─────────────┘
                       ▼
Extraction, filtrage et nettoyage
        │  nom du fichier, page, empreinte SHA-256
        ▼
Découpage avec chevauchement
        │
        ▼
Sentence Transformers (modèle local partagé, lecture seule)
        │  vecteurs normalisés
        ▼
Collection ChromaDB temporaire propre à la session
        │
Question ──► embedding de requête ──► recherche sémantique
                                            │
                                            ▼
                                  passages et métadonnées
                                            │
                                            ▼
                                           Groq
                              (`openai/gpt-oss-120b`)
                                            │
                                            ▼
                               réponse + sources consultées
```

L'assistant code suit volontairement un chemin séparé de la récupération : le
code collé et la consigne sont envoyés directement à Groq, sans requête ChromaDB
et sans afficher de sources documentaires.

## Rôle des composants

### `streamlit_app.py`

Orchestre les actions utilisateur : ajout de fichiers, affichage de la bibliothèque,
questions, historique de session et présentation contextuelle des sources. Chaque
session reçoit un nom de collection ChromaDB aléatoire et son propre `RAGService`.
Seul le modèle d'embeddings, immuable, est placé dans `st.cache_resource`. Le
panneau de droite n'est créé que lorsqu'une réponse possède des sources réelles.
La logique RAG n’y est pas implémentée directement.

### `devmind_rag/documents.py`

- valide les extensions autorisées ;
- extrait le texte des PDF avec PyPDF ;
- décode les fichiers texte et Markdown ;
- normalise Unicode et espaces ;
- découpe chaque page en passages d’environ 900 caractères avec 150 caractères
  de chevauchement ;
- calcule une empreinte SHA-256 du fichier original.

Le chevauchement évite de perdre une information placée à la frontière de deux
passages. Les pages PDF sont traitées séparément afin de conserver une référence
précise.

### `devmind_rag/embeddings.py`

Charge le modèle local multilingue
`paraphrase-multilingual-MiniLM-L12-v2`. Il transforme les passages et questions
en vecteurs normalisés. L’interface est rendue avant le démarrage d’un préchargement
en arrière-plan, uniquement sur une page utilisant le RAG. Un cache protégé par
verrou partage ensuite une seule instance du modèle entre les services du processus
et empêche les chargements concurrents. Ce cache ne contient aucune donnée privée.

### `devmind_rag/vector_store.py`

Stocke dans ChromaDB le texte, le vecteur, le nom du fichier, la page, l’index du
passage et l’empreinte du document. Une collection au nom aléatoire appartient à
une seule session Streamlit ; aucune recherche ne traverse cette collection.
Avant une indexation, l’empreinte est recherchée : un fichier strictement identique
n’est pas vectorisé une seconde fois. Les fichiers Chroma vivent dans le répertoire
temporaire du processus et ne sont pas rouverts après un redémarrage. Une recherche peut recevoir
l'empreinte d'un document comme filtre : le bouton « Poser une question » de la
bibliothèque interroge ainsi uniquement le fichier sélectionné.

Les documents et repositories utilisent la même collection. Le champ
`source_type` distingue leur provenance, tandis que `document_hash` permet de
restreindre une conversation à une seule source.

Pour une question globale sur un repository, le store peut aussi restituer les
chunks du `document_hash` sélectionné. Cette lecture reste strictement filtrée :
elle sert uniquement à compléter la recherche sémantique avec des éléments de
structure du même dépôt.

La capacité est bornée à 1 800 passages par session et les uploads à 8 Mo par
fichier. Le bouton des paramètres permet d'effacer immédiatement la collection de
la session sans toucher aux autres collections.

### `devmind_rag/runtime.py`

Crée un namespace temporaire propre au processus serveur et des noms de collection
imprévisibles pour les sessions. Les anciens namespaces ne sont jamais rouverts
par l'application et les répertoires expirés sont nettoyés avec une portée stricte
au préfixe DevMind. Cette stratégie accepte la nature éphémère du disque Streamlit
Community Cloud sans promettre de persistance.

### `devmind_rag/github.py`

Valide les URL `github.com`, récupère la branche principale et télécharge
l'archive ZIP du repository public via l'API GitHub. Le module applique des
limites de taille et exclut notamment les dépendances, binaires, fichiers
minifiés, lockfiles et répertoires générés. Chaque fichier retenu conserve son
chemin et un lien vers GitHub. L’archive est bornée à 15 Mo ; au plus 250 fichiers,
5 Mo de contenu brut et 400 000 caractères sont retenus. README, manifests et
sources principales passent avant les exemples, tests et licences lors d’un
import partiel.

### `devmind_rag/groq.py`

Encapsule le SDK `groq` et le modèle `openai/gpt-oss-120b`. Le client reçoit
exclusivement la question et les passages retenus. La température basse privilégie
une réponse stable et factuelle. Chaque tentative est limitée à 25 secondes et le
SDK peut effectuer un seul retry avec backoff exponentiel pour une erreur
transitoire. Une réponse interrompue par la limite de longueur est signalée. Des
limiteurs en fenêtre glissante bornent les appels par session et pour l'ensemble
du processus afin de protéger la clé de la démonstration publique.

### `devmind_rag/service.py`

Pour une question, le service crée l’embedding de requête, recherche les passages,
écarte les résultats insuffisamment pertinents, construit un contexte numéroté et
demande à Groq de répondre uniquement avec ce contexte. Il retourne la réponse
et les passages exacts utilisés par l’interface.

Pour un repository, la recherche commence toujours par ChromaDB avec le
`document_hash` de la conversation. Les questions globales comme « quel est le
but ? » ou « explique l’architecture » ajoutent ensuite un petit ensemble de
preuves structurelles du même dépôt. Un reranking léger privilégie README,
documentation, manifests et points d’entrée, et abaisse les licences et exemples
hors sujet. Les questions précises conservent le classement sémantique et une
diversification vers le code source. Les passages identiques sont dédupliqués.

Les citations typographiques sont normalisées vers `[n]`. Après la génération,
seuls les passages explicitement cités sont conservés et renumérotés pour le
panneau de droite. Si Groq omet toute citation, les passages réellement envoyés
restent visibles, sans fabriquer de référence.

Si aucun passage pertinent n’est trouvé, Groq n’est pas appelé. Si la clé API
manque ou si l’appel échoue, l’erreur devient un message compréhensible sans
révéler de secret.

Le même service orchestre l'indexation GitHub et la recherche filtrée par
repository. Sa méthode `generate_code_help` utilise le générateur Groq sans
récupération sémantique ; cette séparation empêche de présenter une analyse de
code générale comme une réponse fondée sur des sources.

### Rendu et défilement Streamlit

Le texte utilisateur et les réponses Groq sont rendus par `st.markdown` et
`st.code`, avec le HTML désactivé. Le workspace central est l’unique zone de
défilement du chat sur desktop ; le conteneur de messages grandit avec son
contenu. Les panneaux navigation et sources restent accessibles, et les trois
zones s’empilent sous 760 px.

## Choix pédagogiques

Le projet évite un framework RAG supplémentaire. Chaque étape est une petite classe
ou fonction Python testable séparément. Les dépendances sont injectables dans
`RAGService`, ce qui permet aux tests d’utiliser des embeddings et un générateur
déterministes sans réseau.

## Sécurité et confidentialité

- la clé Groq provient uniquement de `GROQ_API_KEY` dans Streamlit Secrets ou
  dans l’environnement local ;
- `.env`, `.streamlit/secrets.toml`, `.devmind/` et les journaux sont ignorés par Git ;
- documents, fichiers GitHub, conversations et index sont isolés par session ;
- aucune donnée utilisateur n'est placée dans un cache Streamlit global ;
- seules les sources récupérées sont envoyées à Groq pour une question RAG ;
- les passages récupérés sont déclarés non fiables et leurs éventuelles
  instructions doivent être ignorées par le modèle ;
- dans l'assistant code, le code collé et les consignes sont envoyés à Groq,
  comme l'indique explicitement l'interface.

Cette version est adaptée à une démonstration publique anonyme et temporaire. Elle
n'offre pas de stockage durable ni de compte utilisateur. Un produit conservant
les données entre visites devra ajouter authentification, stockage externe avec
contrôle d'accès et politique de rétention explicite.
