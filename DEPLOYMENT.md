# Déployer DevMind AI sur Streamlit Community Cloud

Ce guide prépare une démonstration publique, anonyme et temporaire. Il ne transforme
pas l'application en service de stockage durable.

## Configuration retenue

- point d'entrée : `streamlit_app.py` ;
- version Python : **3.12** ;
- dépendances de production : `requirements.txt` ;
- secret requis : `GROQ_API_KEY` ;
- modèle Groq : `openai/gpt-oss-120b` ;
- stockage : collections ChromaDB temporaires, isolées par session ;
- système cible : Linux géré par Streamlit Community Cloud.

Python 3.12 correspond à la version utilisée pour les tests locaux et à la version
par défaut actuelle de Community Cloud. La version se choisit dans les paramètres
avancés au moment du déploiement ; elle ne peut pas être modifiée ensuite sans
recréer l'application.

Documentation officielle :

- [Déployer une application](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Dépendances](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)

## 1. Préparer le repository Git local

Le dossier n'était pas encore un repository Git lors de l'audit. Depuis PowerShell,
à la racine de `DevMind-AI` :

```powershell
git init -b main
git add .
git status --short
```

Avant le commit, vérifiez que les éléments suivants n'apparaissent pas parmi les
fichiers ajoutés :

- `.env` ;
- `.streamlit/secrets.toml` ;
- `.devmind/` ;
- `.venv/` ;
- `__pycache__/` et `.pytest_cache/` ;
- tout fichier `*.log` ;
- documents utilisateur privés.

La commande suivante aide à vérifier les fichiers ignorés :

```powershell
git status --short --ignored
```

Puis créez le premier commit :

```powershell
git commit -m "Prepare DevMind AI for Streamlit Community Cloud"
```

## 2. Publier sur GitHub

Créez manuellement un repository GitHub vide, public ou privé selon votre besoin.
Ne cochez pas l'ajout automatique d'un README si le repository local possède déjà
le sien. Ensuite :

```powershell
git remote add origin https://github.com/VOTRE_COMPTE/VOTRE_REPOSITORY.git
git push -u origin main
```

N'exécutez ces commandes qu'après avoir inspecté `git status` et le premier commit.
Ce projet ne crée ni repository distant ni push automatiquement.

## 3. Configurer Streamlit Community Cloud

1. Ouvrez [share.streamlit.io](https://share.streamlit.io/).
2. Cliquez sur **Create app**, puis choisissez le repository et la branche `main`.
3. Indiquez `streamlit_app.py` comme point d'entrée.
4. Ouvrez **Advanced settings**.
5. Sélectionnez **Python 3.12**.
6. Dans **Secrets**, ajoutez exactement :

   ```toml
   GROQ_API_KEY = "votre_cle_groq"
   ```

7. Enregistrez, puis lancez le déploiement.

Ne copiez jamais le fichier `.env` dans GitHub. Streamlit injecte le secret à
l'exécution ; l'application accepte à la fois Streamlit Secrets et la variable
d'environnement locale.

## 4. Lire les logs Cloud

Depuis l'application déployée, ouvrez **Manage app**, puis les logs. Contrôlez en
priorité :

- l'installation complète de `requirements.txt` ;
- l'absence d'erreur d'import ChromaDB, Torch ou Sentence Transformers ;
- le téléchargement initial du modèle d'embeddings ;
- les erreurs d'authentification ou de quota Groq ;
- les messages de limite de ressources.

Une erreur de clé se corrige dans **App settings > Secrets**, puis en redémarrant
l'application. Une modification de `requirements.txt` déclenche une reconstruction
de l'environnement.

## 5. Checklist de validation en production

Ouvrez d'abord une session privée ou un navigateur distinct pour chaque scénario :

1. la page d'accueil s'affiche sans erreur et sans demander de clé au navigateur ;
2. un petit fichier TXT est accepté dans **Mes documents** ;
3. une question reçoit une réponse et des références exactes ;
4. un repository GitHub public de petite taille peut être importé et interrogé ;
5. l'assistant code répond sans afficher de sources documentaires ;
6. un second navigateur ne voit ni les documents ni l'historique du premier ;
7. **Paramètres > Effacer toutes mes données** vide uniquement la session active ;
8. aucun secret, prompt ou contenu privé n'apparaît dans les logs Cloud.

## Isolation et cycle de vie des données

Chaque session Streamlit crée un `RAGService` et un nom de collection ChromaDB
aléatoire. Le modèle Sentence Transformers est le seul objet partagé globalement ;
il ne contient aucune donnée utilisateur. Les conversations restent dans
`st.session_state`.

Les fichiers ChromaDB utilisent un namespace temporaire propre au processus. Un
redémarrage crée un nouveau namespace et l'application ne rouvre pas les anciennes
données. Les namespaces expirés sont nettoyés. Cette stratégie évite les fuites
entre utilisateurs sans ajouter de base externe.

## Limites de la démonstration

- aucune authentification ni restauration entre visites ;
- stockage local temporaire, non sauvegardé ;
- 8 Mo par fichier, cinq fichiers par opération et 1 800 passages par session ;
- import GitHub public soumis aux limites de l'API GitHub sans authentification ;
- appels Groq limités par session et pour le processus ;
- téléchargement et chargement à froid du modèle d'embeddings potentiellement longs ;
- empreinte mémoire importante de Torch, Sentence Transformers et ChromaDB ;
- PDF image uniquement non pris en charge sans OCR ;
- mise en veille possible de l'application après une période sans trafic.

Si la mémoire Cloud devient insuffisante, le premier diagnostic consiste à lire les
logs et les indicateurs de ressources. Le modèle d'embeddings n'a volontairement
pas été remplacé afin de préserver la compatibilité des vecteurs et la qualité
multilingue validée localement.

## Retour arrière

Community Cloud redéploie la branche suivie. Pour revenir à une version connue :

```powershell
git revert IDENTIFIANT_DU_COMMIT
git push origin main
```

Utilisez `git revert` plutôt qu'une réécriture destructive de l'historique partagé.
