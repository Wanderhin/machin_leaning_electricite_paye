# HANDOFF v3 : projet MLOps « Prédiction des impayés télécom (Nigeria) »

> **Pour l'agent IA qui reprend ce projet** : lis ce fichier EN ENTIER avant de répondre ou d'écrire du code.
> Il contient le contexte, l'état exact d'avancement, les décisions déjà prises, les règles non négociables
> et les étapes restantes. Ne redemande pas à l'utilisateur ce qui est écrit ici.
> Le sujet complet de l'examen est dans `cc.md` (à lire aussi, il fait foi en cas de doute).
> **Mets à jour la section 10 (Journal d'avancement) à la fin de chaque étape.**

## ⚠️ CE QUI A CHANGÉ DANS LA v3 (à lire en premier)

1. **Étape 7 (CI/CD/CT) avancée de 7(i) à 7(iii)**, entièrement sur **vraies données** (D12 respecté) :
   - **7(i) isolation des candidats ✅ validée chez l'utilisateur** : `03_entrainement.py` et `04_evaluation.py` acceptent `--modeles-dir` et `--reports-dir` ; `04` accepte aussi `--alias`. Preuve : hash MD5 de `models/model_final.joblib` identique avant/après (`6845F6FD1F7341DBA782FB4D33DAAB4C`), `pytest -q` → 16 passed, registre MLflow : version 2 avec alias `challenger`, **métriques identiques à la production** (reproductibilité).
   - **7(ii) `src/05_porte_qualite.py` ✅ validée** : verdict ACCEPTE, code retour 0, note « équivalent à la référence ».
   - **7(iii) remote DVC + `dvc.yaml` ✅ validé chez l'utilisateur** : remote local synchronisé ; `dvc commit -f` avait d'abord affiché `WinError 32` (CSV brut ouvert dans un autre programme) ; après fermeture d'Excel, relancé **sans erreur** : `dvc.lock` contient les 4 stages, MD5 du CSV brut identique à celui du `.dvc`, `dvc status` « up to date », `dvc status -c` « in sync ». **Reste le commit Git de `dvc.yaml`/`dvc.lock`.**
2. **`01_audit_nettoyage.py` corrigé et validé** : l'utilisateur utilisait encore l'ancienne version (chemins `../data/...`). Le script corrigé reproduit exactement `telecom_clean.csv` (hash MD5 identique).
3. **`.gitignore` racine créé** (il n'existait pas ; le schéma de la v2 le supposait à tort). **Remote DVC `stockage_local`** = `D:\dvc-stockage-impayes`. **Commit Git `beb8797`** (branche `master`).
4. **Le plan est validé** : l'utilisateur a répondu « oui vas y » au plan CI minimale + CT sur vraies données + monitoring par rejeu (D13, D15, D16 → D22). Tu n'as plus à redemander ce plan ; tu dois toujours **expliquer chaque sous-étape avant de livrer** (règle 13) mais n'as pas besoin de re-obtenir l'accord sur le plan global.
5. Restent : commits Git de 7(iii) **et des livrables de l'étape 6 jamais commités** (voir section 6), **copie et exécution de `tests/test_ct_logique.py` chez l'utilisateur (le flow Prefect, lui, est validé)**, 7(v) CI minimale, étape 8 (monitoring), étape 9 (rapport). **L'étape E (reproductibilité dans un clone) a été sautée à la demande de l'utilisateur** (optionnelle).
6. Rappels des principes v2, toujours valables : D12 (pas de livrables factices) et « comprendre avant d'exécuter ».
7. Les fichiers de ce dépôt priment sur ce fichier en cas de contradiction. Mets à jour la section 10 à la fin de chaque étape.

Dernière mise à jour : 2026-10-30 (v3, fin de la session 2 : étape 7 (i) à (iii) faites et validées ; reste le commit Git de 7(iii))

---

## 1. Contexte et objectif

- **Cadre** : examen / projet de la matière *Machine Learning et Modélisation des Données*, Cycle Ingénieur informatique (3IL, EADL), semestre I 2026-2027, enseignant Dr KUATE Alain François. Barème total /80.
- **Rôle simulé** : Data Scientist / ML Engineer qui industrialise un modèle de prédiction des impayés (`default_risk`) pour un opérateur télécom, de l'expérimentation jusqu'au déploiement et au monitoring, en appliquant les principes MLOps.
- **Jeu de données** : *Telecom Credit Scoring Data (Nigeria)*, 500 000 lignes, 15 colonnes, **synthétique** (HuggingFace : `electricsheepafrica/africa-synth-telecom-telecom-credit-scoring-data-nigeria`). Fichier local : `data/telecom_credit_scoring_nigeria.csv`.
- **Cible** : `default_risk` ∈ {`low`, `medium`, `high`}, encodée 0 / 1 / 2 (classification multiclasse).
- **Modèle choisi par l'utilisateur** : **XGBoost** (avec un `DummyClassifier` comme plancher de référence).
- **Utilisateur** : étudiant en 5e année d'ingénierie. Il veut **avancer étape par étape, avec une explication à chaque étape** (le quoi, le pourquoi, comment lancer, ce qu'il doit te recoller). Il écrit en français et doit pouvoir justifier chaque choix dans son rapport.

### Contenu de l'examen à couvrir (résumé de `cc.md`)

| Bloc | Contenu | Points |
|---|---|---|
| Partie I | Questions de cours R1 à R4 (MLOps, ML vs MLOps, CI/CD/CT, outils) et cas pratique R5 à R10 (impayés, dérive, latence) | 5 + 9 + 6 |
| Partie II, QCM | Q1 à Q5 | 15 |
| Partie II, développement | Q6 (pipeline MLOps et outils, 8 pts), Q7 (data drift vs concept drift, 6 pts), Q8 (redéploiement canari, A/B, rollback, 6 pts) | 20 |
| Exercice de conception | i. schéma d'architecture (8), ii. pipeline CI/CD/CT (6), iii. plan de monitoring (6), iv. cycle de vie : versioning, gouvernance, auditabilité (5) | 25 |

**Réponses QCM déjà déterminées** : Q1 = B, Q2 = B, Q3 = C, Q4 = B, Q5 = B.

### Livrables techniques exigés (checklist officielle)

| # | Livrable | État |
|---|---|---|
| 1 | Dépôt Git structuré (code, notebooks, scripts, tests, doc) | 🟡 Git local : commit `beb8797` (branche `master`), `.gitignore` racine OK, 16 tests ; `notebooks/` vide ; doc des prétraitements à rédiger ; `dvc.yaml` / `dvc.lock` à commiter ; push GitHub à confirmer |
| 2 | Données versionnées DVC + doc des prétraitements | ✅ DVC 3.67.1, remote local `stockage_local` synchronisé (`dvc status -c`) ; `dvc.yaml` en place ; ⏳ doc des prétraitements à rédiger |
| 3 | Modèle entraîné et évalué (.joblib, précision, rappel, F1, matrice de confusion, AUC, rapport MLflow) | ✅ entraînement (étape 3) et évaluation test (étape 4) exécutés ; `models/model_final.joblib` produit |
| 4 | API de serving FastAPI/Flask + OpenAPI | ✅ validée par l'utilisateur : `pytest -q` → 16 passed ; `/predict` OK en local et dans Docker |
| 5 | Image Docker | ✅ construite et testée par l'utilisateur (étape 6) |
| 6 | Pipeline CI/CD/CT (lint, tests, entraînement, validation, build, déploiement) | 🟡 faits : isolation des candidats, porte de qualité, `dvc.yaml` ; à faire : flow Prefect (CT), CI minimale ruff + pytest. Déploiement = promotion manuelle (D15) |
| 7 | Registre de modèles (version, statut staging/production, métadonnées) | ✅ `default_risk_xgb` version 1, alias `staging` ; version 2 = candidat d'essai (alias `challenger`), identique à la v1, **à ne pas promouvoir** (aucun gain) |
| 8 | Dashboard de monitoring (métriques techniques et métier, data drift, concept drift, alertes) | ❌ étape 8 |
| 9 | Rapport PDF/DOCX (réponses, architecture, pipeline, monitoring, gouvernance) | ❌ étape 9 |

---

## 2. Environnement de l'utilisateur

- **OS** : Windows. IDE : **PyCharm**. Éviter les commandes bash-only dans les instructions ; préférer des commandes valables dans PowerShell/cmd (ou donner les deux).
- **Racine du projet** : `D:\cours 5e annee\marchine learning\projet_elctricite_nigeria_ml`
  **Les fautes de frappe dans les noms de dossiers (`marchine`, `elctricite`) sont réelles : ne rien renommer.**
- **Environnement virtuel** : `D:\cours 5e annee\marchine learning\marchin_learning_projet_dev\.venv` (situé HORS du dossier du projet).
- **Versions (Windows, Python 3.12), épinglées dans `requirements-api.txt`** : fastapi 0.143.2, joblib 1.6.0, numpy 2.5.3, pandas 3.0.6 (colonnes texte en dtype `str`), pydantic 2.14.1, scikit-learn 1.9.1, uvicorn 0.54.0, xgboost 3.4.1. Versions de mlflow, matplotlib, prefect, evidently, dvc chez l'utilisateur : **inconnues** (demander `pip freeze | findstr /i "mlflow matplotlib prefect evidently dvc"` avant de coder contre leur API).
- **Docker** : Docker Desktop **installé** et fonctionnel (image construite et testée). **Python 3.12** (`C:\Program Files\Python312`). Code **déjà sur GitHub**. **DVC 3.67.1.** **Remote DVC** `stockage_local` = `D:\dvc-stockage-impayes` (local, hors du projet : protège d'une suppression du projet ou du cache, pas d'une panne de disque ; **inutilisable par GitHub Actions**) ; `dvc push` fait (7 objets, car `model_final.joblib` est une copie de `xgb_comportement.joblib`), `dvc status -c` : synchronisé. **`.gitignore` racine** créé en session 2 (.venv, caches, `mlflow.db`, `mlartifacts/`, `mlruns/`, `logs/`, `models/candidats/`, `.idea/`). Git : branche `master`, avertissements « LF will be replaced by CRLF » sans gravité. Le shell de l'utilisateur est **PowerShell**.
- Excel affiche le CSV dans une seule colonne (séparateur `,` contre `;` régional) : sans conséquence pour pandas.

### Structure actuelle du dépôt

```
projet_elctricite_nigeria_ml/
├── .dvc/  .dvcignore  .gitignore (racine, créé en v3)
├── dvc.yaml  dvc.lock (dvc.lock : voir point ouvert, section 6)
├── data/
│   ├── telecom_credit_scoring_nigeria.csv (+ .dvc)      <- brut, ENTRÉE, NE JAMAIS MODIFIER NI ENREGISTRER DEPUIS EXCEL
│   └── processed/    telecom_clean.csv  train.csv  val.csv  test.csv   <- sorties de stages (plus de .dvc individuels)
├── notebooks/        (vide)
├── pipeline/         ct_logique.py (logique pure, sans Prefect), flow_ct.py (flow Prefect) : livrés, à copier/valider
├── reports/          01_audit.txt, 02_decoupage.json, 03_entrainement.csv, 04_evaluation_test.json,
│                     04_matrice_confusion_test.png   (cache: false dans dvc.yaml -> dans Git)
├── src/
│   ├── 01_audit_nettoyage.py   (corrigé v3 : ROOT)
│   ├── features.py
│   ├── 02_decoupage.py
│   ├── 03_entrainement.py      (--modeles-dir, --reports-dir)
│   ├── 04_evaluation.py        (--modeles-dir, --reports-dir, --alias, --promouvoir)
│   ├── 05_porte_qualite.py     (v3)
│   └── api/                    __init__.py, schemas.py, main.py
├── tests/            test_api.py (16 tests) ; test_ct_logique.py (28 cas, livré, jamais lancé avec le vrai pytest)
├── pytest.ini        (racine ; pythonpath = . src ; testpaths = tests)
├── Dockerfile  .dockerignore  requirements-api.txt  requirements.txt   <- NON COMMITÉS au 2026-10-30 (voir section 6)
├── logs/             predictions.jsonl (créé par l'API, ignoré par Git)
├── models/           xgb_comportement.joblib, xgb_complet.joblib, model_final.joblib (DVC) ; decision_poids.json (Git)
│   └── candidats/essai1/   candidat d'essai (ignoré par Git), avec reports/ (dont 05_porte_qualite.json)
└── mlflow.db  mlartifacts/                                <- suivi MLflow local (ignoré par Git)
```

**Jamais livrés / jamais adoptés** (ne pas supposer qu'ils existent) : `.github/workflows/ci.yml`, `ruff.toml`, `src/generer_donnees_ci.py` (brouillon synthétique abandonné). `pipeline/flow_ct.py` : pas encore écrit.
Le fichier `.dvc` du CSV brut reste ; les 7 autres `.dvc` (`telecom_clean`, `train`, `val`, `test`, `model_final`, `xgb_complet`, `xgb_comportement`) ont été retirés avec `dvc remove` (sans `--outs`, les données sont intactes).

---

## 3. Règles NON NÉGOCIABLES

1. **Ne jamais modifier le CSV brut** et ne jamais l'enregistrer depuis Excel (le hash DVC changerait, la reproductibilité serait cassée).
2. **Le jeu de test (`test.csv`) n'est utilisé qu'UNE SEULE FOIS**, à l'étape 4, pour l'évaluation finale. Aucun choix de modèle, de seuil ou d'hyperparamètre ne se fait sur le test (la validation sert à ça).
3. **Tout prétraitement est ajusté (`fit`) sur le train uniquement** et vit **dans un `Pipeline` sklearn** avec le modèle. Il est défini à **un seul endroit : `src/features.py`** (évite le *training-serving skew*).
4. **Graine fixe `SEED = 42`** partout. Tout entraînement est tracé dans MLflow (paramètres, métriques, empreinte MD5 des données, artefacts).
5. **Métriques : F1 macro et rappel de `high` en priorité.** L'accuracy seule est trompeuse (54,6 % rien qu'en répondant toujours `low`). Plancher de référence : F1 macro = 0,236.
6. **Ne jamais utiliser `customer_id`, `phone_number` ni `assessment_date` comme variables.**
7. **Ne pas inventer de résultats** et ne pas promettre un F1 de 0,88 (c'est un exemple fictif de l'énoncé). Les données sont synthétiques et faiblement prédictives (voir section 5).
8. **Ne pas changer silencieusement une décision antérieure** (déduplication, découpage, jeux de variables). Si un changement est nécessaire, l'expliquer à l'utilisateur et demander son accord : un nouveau découpage invalide tous les runs MLflow existants.
9. **Chemins robustes** : dans les scripts, calculer `ROOT = Path(__file__).resolve().parents[1]`. Pas de chemins relatifs du type `../data` (le script 01 a été corrigé en v3 et validé : tous les scripts calculent maintenant `ROOT`).
10. **Aucun secret** (token, mot de passe) dans Git ou dans les images Docker.
11. **Tester avant de livrer** : exécuter chaque script, même sur des données synthétiques de même structure, et dire clairement ce qui a été testé et ce qui ne l'a pas été.

12. **Données réelles d'abord (D12)** : tout chiffre du rapport (F1, dérive, alertes) vient des vraies données. Les données synthétiques (CI) ne prouvent que « le code tourne » et sont toujours étiquetées comme telles. Toute dérive **simulée** est marquée « SIMULÉ » dans les sorties, le dashboard et le rapport.
13. **Comprendre avant d'exécuter** : l'utilisateur veut saisir le pourquoi de chaque étape. Explique, propose, **attends son accord**, puis livre. Ne produis pas de fichiers « pour voir ».
14. **Ne jamais écraser le modèle validé** (`models/model_final.joblib`, `models/decision_poids.json`, MD5 `6845F6FD1F7341DBA782FB4D33DAAB4C`, `version_modele` 6845f6fd1f73) : pour tout candidat utiliser l'isolation implémentée en v3 (`--modeles-dir models/candidats/<nom> --reports-dir models/candidats/<nom>/reports --alias challenger`). **Ne JAMAIS lancer `dvc repro` sans `dvc repro --dry` d'abord** : le stage `evaluation` écrase `model_final.joblib`, ajoute une version au registre MLflow et déplace l'alias `staging`.
15. **Avant toute commande DVC/Git destructive** (`dvc remove`, `git rm`, `dvc repro`) : vérifier `dvc status -c` (cache et remote synchronisés) et connaître le hash du modèle validé. Donner à l'utilisateur d'abord des commandes en **lecture seule** et lui demander de coller la sortie avant toute commande qui modifie le dépôt.

### Conventions de code

- Scripts numérotés dans `src/` : `01_`, `02_`, `03_`… ; un script = une étape ; docstring en tête qui dit ce qu'il lit, ce qu'il écrit, comment le lancer.
- Code et commentaires en **français sans accents** (ASCII) dans les `.py`, pour éviter tout problème d'encodage sous Windows. Les `.md` et le rapport sont en français avec accents.
- Rapports produits dans `reports/` ; modèles dans `models/`.
- Un commit Git par étape, message en français, par exemple `Etape 2 : decoupage stratifie 70/15/15 et module de pretraitement`. Données et modèles versionnés avec `dvc add`, puis commit des fichiers `.dvc`.

### Façon de travailler avec l'utilisateur

- Répondre en français, ton chaleureux et direct, **une étape à la fois**.
- Pour chaque étape : (1) ce qu'on fait et pourquoi (en lien avec le barème et les livrables), (2) le code livré sous forme de fichier, (3) la commande exacte à lancer, (4) **ce que l'utilisateur doit te recoller** (sortie du terminal), (5) l'interprétation honnête des résultats, (6) la suite.
- Les explications doivent aussi servir son rapport (justifications de choix d'outils, risques, alternatives).
- Quand l'utilisateur pose une question de fond (« sur quoi fait-on le monitoring ? », « à quoi ça sert ? »), réponds d'abord à la question, honnêtement, y compris pour reconnaître qu'un choix précédent était discutable. Ne défends pas un brouillon qu'il juge inutile.
- Dis toujours ce qui a été **testé** et ce qui ne l'a pas été (ex. GitHub Actions n'a jamais tourné).

---

## 4. Ce qui a été fait

### Étape 1 : audit et nettoyage (`src/01_audit_nettoyage.py`) ✅ exécutée

Résultats de l'audit sur `telecom_credit_scoring_nigeria.csv` :

- 500 000 lignes, 15 colonnes ; **aucune valeur manquante, aucune valeur invalide, aucune ligne entièrement dupliquée.**
- `assessment_date` est constante (une seule valeur) → supprimée. `phone_number` supprimée (donnée personnelle sans valeur prédictive ; point de gouvernance à citer dans le rapport).
- **13 661 `customer_id` dupliqués** → première occurrence gardée → **486 339 lignes** dans `telecom_clean.csv` (13 colonnes).
  *Hypothèse non vérifiée* : ce sont probablement des **collisions d'identifiants aléatoires** (clients différents), car ~13 600 collisions sont attendues si les ID sont 7 chiffres tirés au hasard. Vérification proposée à l'utilisateur (fraction des `customer_id` dupliqués ayant des `phone_number` différents). Décision à documenter dans le rapport ; la perte est de 2,7 %.
- **Cible** : low 54,58 % / medium 23,16 % / high 22,26 % (déséquilibre modéré).
- Distributions numériques **uniformes** (jeu synthétique) : `credit_score` 300–1000, `tenure_months` 1–120, `avg_monthly_spend_ngn` ≈ 500–50 000, `recharge_frequency_per_month` 1–30, `payment_consistency_score` 0–1, `late_payment_count` 0–10, `max_loan_amount_ngn` 0–149 588.
- Modalités : `operator` ∈ {MTN, Airtel, Glo, 9mobile} ; `account_type` ∈ {prepaid, postpaid} (85 % / 15 %) ; `credit_rating` ∈ {poor, fair, good, excellent} ; `loan_eligibility` converti en 0/1.
- **Test de fuite (arbre de profondeur 4, une colonne à la fois)** : aucune colonne ne dépasse un F1 macro de **0,454** → pas de fuite directe. `credit_rating`, `loan_eligibility`, `credit_score`, `max_loan_amount_ngn` obtiennent **exactement** 0,454 : elles portent la même information. Les autres colonnes sont entre 0,18 et 0,42.
  Limite : ce test ne détecte pas une fuite visible seulement en combinant des colonnes.

### Étape 2 : découpage et prétraitement (`src/02_decoupage.py`, `src/features.py`) ✅ exécutée

- Découpage **70 / 15 / 15, stratifié sur la cible, aléatoire** (pas de dimension temporelle : date constante), `random_state = 42`. Tailles : **train 340 437 / val 72 951 / test 72 951**, proportions de cible identiques dans les trois (54,59 / 23,16 / 22,25 %). **0 client partagé entre les morceaux.**
- Les splits conservent `customer_id` (traçabilité) ; `features.separer_X_y` le retire avant modélisation.

**Deux jeux de variables (décision méthodologique centrale)** :

| Jeu | Variables | Rôle |
|---|---|---|
| **A « comportement »** (7) | `tenure_months`, `avg_monthly_spend_ngn`, `recharge_frequency_per_month`, `payment_consistency_score`, `late_payment_count`, `operator`, `account_type` | Observable avant toute décision de crédit. **Jeu de référence a priori.** |
| **B « complet »** (11) | A + `credit_score`, `credit_rating`, `loan_eligibility`, `max_loan_amount_ngn` | Ces 4 colonnes ressemblent à des **sorties** d'un scoring (une limite de prêt n'existe qu'après l'évaluation du risque). Servent de comparaison. |

Règle de décision : si B n'améliore presque rien par rapport à A, **on garde A** (plus simple, plus défendable). Si B améliore beaucoup, **en discuter avec l'utilisateur** (c'est un point de gouvernance pour le rapport) au lieu de trancher seul.

**`features.py`** expose : `SEED`, `TARGET`, `ID_COL`, `CLASSES = ["low","medium","high"]`, `CLASS_TO_INT`, `FEATURE_SETS = ("comportement","complet")`, `colonnes(jeu)`, `build_preprocessor(jeu, scale=False)` (ColumnTransformer : imputation médiane / plus fréquente, one-hot pour `operator` et `account_type`, encodage ordinal pour `credit_rating` poor<fair<good<excellent, `set_output("pandas")`), `separer_X_y(df)`.

### Étape 3 : entraînement XGBoost + MLflow (`src/03_entrainement.py`) ✅ exécutée le 2026-10-30

**Résultats réels sur la VALIDATION** (72 951 lignes) :

| Run | Accuracy | F1 macro | AUC | Rappel low | Rappel medium | Rappel high | Arbres |
|---|---|---|---|---|---|---|---|
| dummy_reference | 0,5459 | 0,2354 | 0,5000 | 1,000 | 0,000 | 0,000 | - |
| **xgb_comportement (A)** | 0,6197 | **0,4589** | **0,6549** | 0,852 | 0,027 | 0,666 | 437 |
| xgb_complet (B) | 0,6382 | 0,4543 | 0,6477 | 0,886 | 0,000 | 0,695 | 89 |

Lectures (hypothèses d'interprétation, pas des faits prouvés) :
- Les deux modèles battent nettement le plancher (F1 macro 0,235 → ~0,46), mais le signal est **faible** (AUC ~0,65).
- **B n'apporte rien par rapport à A** (F1 macro et AUC légèrement plus bas, écart non significatif) : les colonnes de scoring sont redondantes avec le comportement. Règle D4 appliquée : **jeu A retenu** (D9).
- Le meilleur arbre à une seule colonne de l'audit (0,454) fait presque aussi bien que XGBoost sur 7 variables : le plafond semble tenir au jeu de données, pas au modèle. Un réglage d'hyperparamètres apportera très peu.
- **La classe `medium` n'est quasiment jamais prédite** (rappel 0,027 et 0,000) malgré les poids « balanced » : le modèle se comporte comme un détecteur low vs high. Hypothèse : `medium` est la bande intermédiaire et sa probabilité n'est presque jamais la plus élevée. C'est ce qui plafonne le F1 macro. Traité à l'étape 4 par des poids de décision (D10).
- Avertissements MLflow vus et **sans gravité** : `artifact_path` déprécié (remplacé par `name`) et avertissement de sécurité sur le format pickle/cloudpickle (ne charger que des modèles de confiance ; à citer dans la partie gouvernance/sécurité du rapport).

Détails techniques :

- 3 runs MLflow : `dummy_reference` (classe majoritaire), `xgb_comportement`, `xgb_complet`.
- XGBoost : `multi:softprob`, `n_estimators=1000` avec **arrêt anticipé (50) surveillé sur la validation**, `learning_rate=0.05`, `max_depth=5`, `subsample=0.8`, `colsample_bytree=0.8`, `min_child_weight=5`, `tree_method="hist"`, **poids d'échantillons « balanced »** (classes rares plus pesantes ; compromis : rappel de `low` plus faible).
- Le prétraitement est ajusté sur le train, puis `Pipeline([("prep", pre), ("model", modele)])` est assemblé à partir d'éléments déjà ajustés (pour permettre l'arrêt anticipé).
- Métriques loguées sur la validation : accuracy, F1 macro, précision et rappel macro, AUC un-contre-tous, log-loss, rappel par classe. Artefacts : `importance_variables_gain.json`, `.joblib`, modèle MLflow, MD5 des fichiers train et val.
- Choix techniques MLflow : backend **SQLite** (`sqlite:///mlflow.db`, nécessaire pour le registre), `artifact_location` explicite (`mlartifacts/`), `serialization_format=CLOUDPICKLE` (le format par défaut de MLflow 3.x demande le paquet `skops`, absent).
- Commandes : `pip install xgboost mlflow matplotlib joblib` ; `python src/03_entrainement.py` ; UI : `mlflow ui --backend-store-uri sqlite:///mlflow.db` (depuis la racine), puis http://127.0.0.1:5000.

---

## 5. Attentes réalistes sur les résultats

- Les variables sont tirées uniformément et chaque colonne seule plafonne à 0,454 de F1 macro ; s'attendre à un **F1 macro modeste**, probablement très loin de 0,88. Ce n'est pas un échec : le rapport doit expliquer l'écart par rapport au plancher (0,236), la comparaison A/B et le rappel de `high`.
- Ne pas optimiser à outrance : si le signal est faible, un gain d'hyperparamètres sera faible aussi. Un réglage (Optuna ou grille réduite, évalué sur la validation) est **optionnel**, à proposer seulement après l'étape 4 si le temps le permet.

---

## 6. PROCHAINE ACTION IMMÉDIATE

1. ✅ **`dvc.lock` vérifié (point ouvert levé le 2026-10-30).** L'erreur `[WinError 32]` venait d'un CSV brut ouvert dans un autre programme (Excel ou aperçu PyCharm). Après fermeture : `dvc commit -f` sans erreur ; `dvc.lock` contient `audit_nettoyage`, `decoupage`, `entrainement`, `evaluation` ; MD5 du CSV brut = md5 du `.dvc` (`3b13408c5fc88ebd4b7f3bf8ebfa8b3a`, 51 426 327 octets) ; `dvc status` « up to date » ; `dvc status -c` « in sync ». **Ne jamais lancer `dvc repro` pour « réparer »** (règle 14).
2. **Commits Git (état constaté par `git status --short` le 2026-10-30).** Suppressions non indexées : les 7 `.dvc` retirés. **Non suivis** : `.dockerignore`, `Dockerfile`, `cc.md`, `data/.gitignore`, `dvc.lock`, `dvc.yaml`, `pytest.ini`, `requirements-api.txt`, `requirements.txt`. **Suivi par erreur** : `src/__pycache__/features.cpython-312.pyc` (commité avant le `.gitignore`). **Constat important : les livrables de l'étape 6 (Dockerfile, .dockerignore, pytest.ini, requirements*) n'avaient jamais été commités**, contrairement à ce que supposait la v2 (« code déjà sur GitHub ») : GitHub ne les a pas, et une CI sans `pytest.ini` (pythonpath) ni `requirements` échouerait. `reports/*` et `models/decision_poids.json` ne sont pas listés : déduits suivis et inchangés (à confirmer par `git ls-files models reports`). Commandes données à l'utilisateur : 2 commits à chemins explicites (jamais `git add .`) : (a) `git rm` des 7 `.dvc` + `git add dvc.yaml dvc.lock data/.gitignore` ; (b) `git rm --cached` du `.pyc` + `git add Dockerfile .dockerignore pytest.ini requirements-api.txt requirements.txt`. À vérifier ensuite : `git status --short` (attendu : seulement `?? cc.md`), `git check-ignore -v` sur le CSV brut, `data/processed/train.csv` et `models/model_final.joblib` (ils doivent être ignorés), `git remote -v` (un push GitHub est-il configuré ?). `cc.md` (sujet de l'examen) : à l'utilisateur de décider (idéalement `docs/`).
3. **Étape E : SAUTÉE** à la demande de l'utilisateur (2026-10-30). Optionnelle ; la proposer à nouveau avant le rapport. En attendant, ne pas affirmer dans le rapport qu'un clone a été rejoué ; citer seulement que le candidat `essai1` a redonné exactement les métriques de la production.
4. **7(iv) flow Prefect : VALIDÉ chez l'utilisateur (2026-10-30), reste la copie des tests.** Exécuté avec Prefect 3.9.0 sur les vraies données : validation des données OK (train 340 437 / val 72 951 / test 72 951 lignes, écart de proportions 0,00 point, clients disjoints), 03 puis 04 (version 3 du registre, alias `challenger`) puis 05 (ACCEPTE, F1 0,4813, rappel high 0,6217), verdict `ACCEPTE_EQUIVALENT`, durée ≈ 95 s dont ≈ 20 s de démarrage du serveur Prefect temporaire. Après : MD5 de `model_final.joblib` inchangé (`6845F6FD1F7341DBA782FB4D33DAAB4C`) et `dvc status` « up to date ». **Point non réglé : `pytest -q` a affiché 16 passed au lieu de 44 → `tests/test_ct_logique.py` n'a pas été copié (ou mal placé).** À demander : copier le fichier dans `tests/`, relancer `pytest -q` (attendu : 44 passed). À vérifier aussi : `Get-Content models\candidats\20261030_093450_manuel\ct_resume.json` (`version_registre_mlflow` doit valoir 3, MD5 des données présents). Le flow a laissé le dossier `models/candidats/20261030_093450_manuel/` et la version 3 du registre (avec la version 2 de `essai1`) : deux candidats équivalents à la production, **à ne pas promouvoir** ; la production reste la version 1 (`staging`).
5. **7(v) CI minimale** : `.github/workflows/ci.yml` (job ruff + pytest uniquement) + `ruff.toml` (voir 7.4).
6. **Étape 8** (monitoring par rejeu, dérive simulée étiquetée « SIMULÉ ») puis **étape 9** (rapport).
7. Question toujours sans réponse : le rapport doit-il inclure la Partie I (R1 à R10) et le QCM ?
8. À chaque étape : un commit Git en français, mise à jour de la section 10, et la liste de ce que l'utilisateur doit te recoller.

---

## 7. Étapes restantes (cahier des charges détaillé)

### Étape 4 : évaluation finale sur le test + registre de modèles → `src/04_evaluation.py` ✅ exécuté le 2026-10-30

**Résultats réels** :
- Validation : F1 macro argmax simple 0,4589 → avec poids de décision `(1.0, 1.25, 1.0)` 0,4794 (**+0,0205 ≥ 0,01 : poids adoptés**).
- **TEST (une seule évaluation)** : accuracy 0,5830 | **F1 macro 0,4813** (argmax simple : 0,4587) | **AUC 0,6492** | log-loss 0,9651.

| Classe | Précision | Rappel | F1 |
|---|---|---|---|
| low | 0,698 | 0,761 | 0,728 |
| medium | 0,233 | 0,127 | 0,164 |
| high | 0,496 | 0,622 | 0,552 |

- Registre MLflow : modèle `default_risk_xgb`, **version 1, alias `staging`**. Fichiers produits : `models/model_final.joblib`, `models/decision_poids.json`, `reports/04_matrice_confusion_test.png`, `reports/04_evaluation_test.json`.
- Lectures : (1) gain des poids confirmé sur le test (+0,023 vs +0,021 en validation) et F1 macro validation 0,479 / test 0,481 : **pas de surajustement à la validation**. (2) Pour `high`, précision 0,50 pour une base de 22 % : le modèle repère 62 % des mauvais payeurs mais la moitié des alertes sont de fausses alertes. (3) Pour `medium`, la précision (0,233) est égale à la fréquence de la classe (23 %) : **le modèle ne sait pas identifier `medium`** ; le gain de F1 macro vient en partie du fait de prédire un peu plus de `medium`. À présenter honnêtement dans le rapport comme une limite (le modèle discrimine surtout low vs high). (4) Log-loss peu informative : les probabilités ne sont pas calibrées (poids « balanced »), d'où le nom `scores` dans l'API. (5) Les avertissements MLflow (schéma d'entiers, cloudpickle) sont sans gravité.

**(Description du script livré)**

**Ce que fait le script livré** :
- Charge `models/xgb_comportement.joblib` (option `--jeu complet`).
- Cherche sur la **validation seulement** des **poids de décision** `(1, w_medium, w_high)` multipliant les probabilités avant l'argmax (grille, objectif = F1 macro). Ils ne sont adoptés que si le gain en validation est ≥ 0,01 ; sinon argmax simple (D10).
- Évalue **une seule fois** sur `test.csv` avec la règle figée ; logue aussi l'argmax simple sur le test à titre de transparence (la décision a été prise sur la validation).
- Produit : `reports/04_matrice_confusion_test.png`, `reports/04_evaluation_test.json`, `models/model_final.joblib`, **`models/decision_poids.json`**, un run MLflow `evaluation_test_<jeu>`.
- Enregistre le modèle dans le registre sous **`default_risk_xgb`** avec alias **`staging`** et tags (`statut`, `jeu_variables`, `poids_decision`, `f1_macro_test`, `auc_test`, `train_md5`, `seed`).
- `python src/04_evaluation.py --promouvoir N` pose l'alias `production` et le tag `statut=production` sur la version N. La porte de qualité automatique viendra à l'étape 7.
- **Conséquence pour l'étape 5 (API)** : l'API doit appliquer `argmax(probabilités × poids)` en lisant `models/decision_poids.json` (poids `[1,1,1]` = argmax simple). Sinon les prédictions de l'API diffèrent de celles évaluées.

Rappel du cahier des charges initial de l'étape 4 :
- Charger le modèle retenu, évaluer **une seule fois sur `test.csv`** : précision, rappel, F1 (par classe et macro), **matrice de confusion (image PNG, matplotlib backend `Agg`)**, **AUC** un-contre-tous, rapport de classification. Logger le tout dans MLflow (nouveau run « évaluation test »).
- Enregistrer le modèle dans le **MLflow Model Registry** (backend SQLite) avec métadonnées (jeu de variables, MD5 des données, métriques, graine). Représenter les statuts **staging / production** : vérifier ce que la version de MLflow installée supporte (les *stages* sont dépréciés en 2.9+, les **alias** et **tags** sont la voie actuelle ; utiliser alias `champion` / `challenger` + tag `statut=staging|production` pour satisfaire l'énoncé).
- Sauvegarder le modèle final sous un nom stable (ex. `models/model_final.joblib`) et le versionner avec DVC.
- Optionnel : importance des variables, courbes, analyse des seuils de décision selon le coût d'un faux négatif sur `high`.

### Étape 5 : API de serving (FastAPI) → `src/api/` ✅ validée par l'utilisateur (2026-10-30 : `pytest -q` → 16 passed ; `/predict` OK)

**Livré** : `src/api/schemas.py`, `src/api/main.py`, `tests/test_api.py`, `pytest.ini`.
- Endpoints : `GET /health`, `GET /model-info`, `POST /predict` ; OpenAPI sur `/docs`. Lancement : `uvicorn api.main:app --app-dir src --reload` depuis la racine.
- Le modèle est chargé **une fois au démarrage** (lifespan). Configuration par variables d'environnement : `MODEL_DIR`, `MODEL_VERSION`, `PREDICTION_LOG`, `EVAL_REPORT` (utile pour Docker).
- Règle de décision identique à l'étape 4 : `argmax(probabilités × poids)` avec `models/decision_poids.json`. L'API **refuse de démarrer** si le modèle n'a pas été entraîné avec le jeu `comportement`.
- Entrées : 7 champs ; erreurs 422 pour les valeurs impossibles ; valeurs hors plage d'entraînement **acceptées avec avertissement** (signal de dérive). `operator` et `account_type` insensibles à la casse.
- Sortie : `classe_predite`, `scores` (**pas des probabilités calibrées**), `version_modele` (= md5 du fichier modèle, identique au hash DVC), `avertissements`, `latence_ms`.
- Journal `logs/predictions.jsonl` (une ligne JSON par prédiction ; `customer_id` **hashé**, jamais en clair) pour le monitoring de l'étape 8. Ajouter `logs/` au `.gitignore`.
- Tests : 16 tests pytest sur un **modèle temporaire** (pas besoin de DVC), prêts pour la CI.
- Non fait volontairement à ce stade : endpoint `/metrics` (Prometheus) et traitement par lots ; à ajouter à l'étape 8 si besoin.
- Versions : le `.joblib` dépend des versions exactes de scikit-learn, xgboost, pandas, numpy. Générer `requirements-api.txt` avec les versions de l'utilisateur : `pip freeze | findstr /i "scikit-learn xgboost pandas numpy joblib fastapi uvicorn pydantic"`.

Rappel du cahier des charges initial de l'étape 5 :
- Endpoints : `POST /predict` (entrées brutes d'un client, sortie : classe prédite, probabilités des 3 classes, version du modèle), `GET /health`, `GET /model-info`. Documentation OpenAPI automatique (`/docs`).
- Schéma Pydantic avec contraintes issues de l'audit : `tenure_months` 1–120, `avg_monthly_spend_ngn` > 0, `recharge_frequency_per_month` 1–30, `payment_consistency_score` 0–1, `late_payment_count` 0–10, `operator` ∈ {MTN, Airtel, Glo, 9mobile}, `account_type` ∈ {prepaid, postpaid} (+ les 4 colonnes de scoring seulement si le jeu B est retenu).
- Le pipeline attend un **DataFrame pandas avec les noms de colonnes exacts** ; l'API doit le construire.
- **Le `.joblib` dépend des versions exactes de scikit-learn, xgboost, pandas, numpy** : les épingler à l'identique dans `requirements.txt` et l'image Docker. (Le pipeline ne référence pas `features.py`, il n'a besoin que des bibliothèques.)
- Journaliser chaque requête (horodatage, entrées, prédiction, version du modèle, latence) dans un fichier/SQLite : ces logs alimenteront le monitoring (étape 8).
- Tests `pytest` avec `TestClient` (cas valide, valeur hors plage, champ manquant).

### Étape 6 : Docker ✅ fait et validé par l'utilisateur (2026-10-30)

- Fichiers à la racine : `Dockerfile`, `.dockerignore`, `requirements-api.txt`.
- Image `python:3.12-slim` ; `libgomp1` (OpenMP, requis par XGBoost sous Linux) ; utilisateur non-root ; `HEALTHCHECK` en Python (pas de curl dans l'image slim) ; variables `MODEL_DIR`, `PREDICTION_LOG`, `EVAL_REPORT`. N'embarque que `src/` (sans les scripts `0*.py`), `models/model_final.joblib`, `models/decision_poids.json`, `reports/04_evaluation_test.json` ; exclut données, `.dvc`, `mlflow.db`, `mlartifacts/`, `logs/`, tests.
- **`xgboost-cpu` dans l'image (D14)** : sous Linux, `xgboost==3.4.1` tire ~300 Mo de paquets NVIDIA (vérifié par `pip install --dry-run`). Le Dockerfile remplace `xgboost==` par `xgboost-cpu==` (même version, même module `xgboost`) avec un `sed`. **Validé** : même `version_modele` (6845f6fd1f73) que le serveur local.
- Résultat utilisateur : `/predict` dans le conteneur → `classe_predite: high`, scores {low 0,0959 ; medium 0,3237 ; high 0,5804}, latence ≈ 30 ms (≈ 90 ms sur le premier appel local).
- Commandes : `docker build -t default-risk-api:0.1 .` puis `docker run --rm -p 8000:8000 default-risk-api:0.1`.
- Optionnel, à décider à l'étape 8 : `docker-compose.yml` (API + MLflow + monitoring).

### Étape 7 : pipeline CI/CD/CT 🟡 7(i), 7(ii) et 7(iii) faits et validés ; restent 7(iv) flow Prefect et 7(v) CI minimale

**Plan validé par l'utilisateur le 2026-10-30 (« oui vas y »)**, après son refus des livrables factices (citation : « si le pipeline n'utilise pas les vraies données, ça ne sert à rien ; sur quoi allons-nous faire le monitoring ? [...] faire un truc factice ne servira à rien »). Principe : tout chiffre vient des vraies données (D12) ; la CI ne teste que le code et n'utilise aucune donnée. Sous-étapes : (i) isolation des candidats ✅ → (ii) porte de qualité ✅ → (iii) `dvc.yaml` 🟡 → (iv) flow Prefect → (v) CI minimale.

#### 7.0 Qui utilise quelles données (à expliquer à l'utilisateur, puis à reprendre dans le rapport)

| Couche | Rôle | Données | Où ça tourne |
|---|---|---|---|
| **CI** (intégration continue) | Vérifier que le **code** ne casse pas à chaque `git push` (lint, tests) | Aucune donnée réelle nécessaire : les 16 tests utilisent un petit modèle jetable. C'est un test de plomberie, pas une évaluation | GitHub Actions |
| **Pipeline reproductible** (`dvc.yaml`, `dvc repro`) | Rejouer à l'identique audit → découpage → entraînement → évaluation | **Vraies données** (CSV versionné par DVC) | Poste de l'utilisateur (ou CI si remote DVC accessible) |
| **CT** (entraînement continu) | Réentraîner un **challenger** et le soumettre à la porte de qualité | **Vraies données** | Flow Prefect, poste de l'utilisateur |
| **Porte de qualité** | Bloquer un candidat trop faible ou pire que la production | **Métriques réelles** (jeu de test réel) | Fin du flow CT |
| **Monitoring** (étape 8) | Détecter dérive et dégradation | Référence = `train.csv` réel ; « production » = **rejeu de lignes réelles** via l'API (voir étape 8) | Poste de l'utilisateur |

#### 7(i) Isolation des candidats ✅ (validée chez l'utilisateur)

- `03_entrainement.py` : `--modeles-dir` (défaut `models/`), `--reports-dir` (défaut `reports/`). `04_evaluation.py` : mêmes options + `--alias {staging,challenger}` (défaut `staging` ; le tag `statut` suit l'alias) ; il lit `xgb_<jeu>.joblib` dans `--modeles-dir`, y écrit `model_final.joblib` et `decision_poids.json`, et écrit ses rapports dans `--reports-dir`. `--promouvoir N` retire aussi les alias `staging` et `challenger` de la version promue. **Sans option, comportement identique à avant.**
- Validation utilisateur : MD5 de `models\model_final.joblib` identique avant et après un run candidat ; `pytest -q` 16 passed ; candidat `models/candidats/essai1` : F1 macro 0,4813, AUC 0,6492, poids `(1.0, 1.25, 1.0)`, version 2 du registre avec alias `challenger`. **Résultats identiques à la production** : l'entraînement est reproductible sur cette machine (même données, même graine). Le hash du candidat n'a pas été comparé à celui de `model_final.joblib` (à faire : `Get-FileHash models\candidats\essai1\model_final.joblib`).
- Tests de l'agent : uniquement sur doublures de `xgboost`/`mlflow` et données jetables (plomberie des chemins) ; la validation réelle est celle de l'utilisateur.
- Commande type : `python src/03_entrainement.py --modeles-dir models/candidats/X --reports-dir models/candidats/X/reports` puis `python src/04_evaluation.py --modeles-dir models/candidats/X --reports-dir models/candidats/X/reports --alias challenger`.

#### 7.1 `dvc.yaml` : pipeline reproductible sur les vraies données ✅ (reste le commit Git)

Fichier à la racine, 4 stages : `audit_nettoyage` → `decoupage` → `entrainement` → `evaluation` (graphe vérifié par `dvc dag`). Entrées/sorties :
- `audit_nettoyage` : deps `01_audit_nettoyage.py` + CSV brut ; outs `data/processed/telecom_clean.csv`, `reports/01_audit.txt` (cache: false).
- `decoupage` : deps `02_decoupage.py`, `features.py`, `telecom_clean.csv` ; outs `train.csv`, `val.csv`, `test.csv`, `reports/02_decoupage.json` (cache: false).
- `entrainement` : deps `03_entrainement.py`, `features.py`, `train.csv`, `val.csv` ; outs `models/xgb_comportement.joblib`, `models/xgb_complet.joblib`, `reports/03_entrainement.csv` (cache: false).
- `evaluation` : deps `04_evaluation.py`, `features.py`, `xgb_comportement.joblib`, `val.csv`, `test.csv` ; outs `models/model_final.joblib` ; `models/decision_poids.json` et `reports/04_matrice_confusion_test.png` (cache: false) ; metrics `reports/04_evaluation_test.json` (cache: false).
- Les rapports `cache: false` restent dans Git (lisibles dans les commits). `mlflow.db` et `mlartifacts/` ne sont pas des sorties (gitignorés). Le CSV brut reste une entrée suivie par son `.dvc`.
- **Mise en place (faite chez l'utilisateur)** : commit Git `beb8797` ; `dvc remove` des 7 `.dvc` de sorties ; `dvc dag` conforme ; `dvc status` listait tout « modified » (normal sans `dvc.lock`) ; `dvc commit -f` pour **adopter l'état existant sans rien exécuter** → erreur WinError 32, puis `dvc status` et `dvc repro --dry` propres. **Lock vérifié le 2026-10-30 (voir section 6, point 1).**
- Vérifié statiquement par l'agent (YAML, pas de cycle, pas de sortie dupliquée, cohérence avec les écritures des scripts). **DVC n'est pas installé dans le bac à sable de l'agent** : aucune commande DVC n'a été testée par lui.
- Conséquences à connaître : modifier un script (même un commentaire) fait considérer les stages concernés comme « changed » et `dvc repro` les rejouera ; `dvc.lock` enregistre les empreintes des entrées et sorties = **auditabilité** (quelles données et quel code ont produit quel modèle) ; `dvc metrics show` affiche les métriques de test (capture utile pour le rapport).
- **Limite à écrire dans le rapport** : `model_final.joblib` a été produit avant l'ajout des options d'isolation à 03/04 ; le lock enregistre les empreintes des scripts actuels. Le comportement par défaut est inchangé et le run candidat a redonné les mêmes métriques.
- **Étape E, preuve de reproductibilité (SAUTÉE à la demande de l'utilisateur le 2026-10-30 ; optionnelle, à reproposer avant le rapport)** dans un **clone jetable** (son `mlflow.db` est séparé et gitignoré : le registre réel n'est pas touché) : `git clone <dépôt local> D:\clone_repro` ; `cd D:\clone_repro` ; (venv activé) `dvc pull` (le remote est dans `.dvc/config`) ; `dvc repro --force` ; `Get-FileHash models\model_final.joblib -Algorithm MD5` à comparer à `6845F6FD1F7341DBA782FB4D33DAAB4C`. XGBoost n'est pas garanti identique bit-à-bit : si le hash diffère, comparer les métriques (`reports/04_evaluation_test.json`) et le dire honnêtement.

#### 7.2 CT : flow Prefect `pipeline/flow_ct.py` sur les vraies données ✅ livré et validé chez l'utilisateur (tests de logique à copier)

**Décision A (2026-10-30)** : le CT réutilise les splits existants `train/val/test` ; il ne rejoue pas `02`. Livré : `ct_logique.py` (validation des données, isolation, commandes, verdict ; aucune dépendance à Prefect), `flow_ct.py` (tâches `valider_donnees` et `lancer_script`, flow `ct_impayes`), `tests/test_ct_logique.py`. Étapes : validation des données (colonnes requises, effectifs minimums 10 000/2 000/2 000, cible valide, proportions des classes à ±5 points de `reports/02_decoupage.json`, clients disjoints) → `03` → `04 --alias challenger` → `05` (contre `reports/04_evaluation_test.json`) → verdict → `ct_resume.json` (raison, MD5 des données et du modèle de production, contrôles, version MLflow, verdict). Verdicts : `REFUSE` (code 1 de 05 : résultat métier, le flow se termine normalement), `ACCEPTE_EQUIVALENT` (|écart F1| < 0,0005 : rien à promouvoir), `ACCEPTE_AVEC_ECART` (examen humain ; message distinct si l'écart est négatif ou positif), `DONNEES_INVALIDES` (aucun entraînement, le flow échoue), `ECHEC_<script>` (erreur technique, le flow échoue). Options : `--raison {manuel,planning,derive_donnees,degradation_performance}`, `--planifier CRON`, `--f1-min`, `--rappel-high-min`, `--tolerance` (non transmis à 05 si absents : un seul endroit pour les seuils par défaut). Garde-fou : `verifier_isolation` refuse tout dossier hors `models/candidats/` (testé, y compris `..`).
**Testé par l'agent (bac à sable, sans vrai Prefect, XGBoost ni MLflow)** : 28 cas avec un exécuteur imitant pytest ; 3 mutations de la logique détectées ; flow de bout en bout avec faux `prefect`, doublures `xgboost`/`mlflow` et données jetables de 30 000 lignes : `REFUSE` sur étiquettes aléatoires, `ACCEPTE_EQUIVALENT` avec seuils à 0, `DONNEES_INVALIDES` sur `val.csv` corrompu (code de sortie 1), `models/`, `reports/` et `data/` identiques octet pour octet après les lancements. **Validé ensuite chez l'utilisateur le 2026-10-30** : le flow avec Prefect 3.9.0 sur les vraies données (verdict `ACCEPTE_EQUIVALENT`, modèle validé intact, `dvc status` propre ; voir section 6, point 4). **Toujours non testé** : `test_ct_logique.py` avec le vrai pytest (fichier pas encore copié par l'utilisateur), ruff, `--planifier`, les verdicts `REFUSE` et `DONNEES_INVALIDES` sur le vrai dépôt (testés seulement en bac à sable).
**Limites à écrire dans le rapport** : avec les mêmes données le candidat est identique à la production (gain nul, déjà observé) ; en production le CT serait alimenté par les impayés réellement constatés ; chaque lancement ajoute des runs MLflow, une version au registre (alias `challenger` déplacé) et un dossier `models/candidats/...` (ignoré par Git) ; 03 entraîne aussi le jeu « complet », inutile pour le CT.
**Ancien plan (conservé pour mémoire, remplacé par le texte ci-dessus)** :

- Prefect **3.9.0** installé chez l'utilisateur (`pip show prefect`, 2026-10-30) ; **absent du bac à sable de l'agent** : consulter la doc de cette version avant de coder, séparer la logique (fonctions pures : validation des données, verdict, chemins) du décorateur Prefect, et tester la logique avec un faux module `prefect`. Tâches : valider les données (schéma, plages, manquants, comme l'audit 01) → préparer (02) → entraîner (03) → évaluer (04) → porte de qualité (05) → enregistrer le **challenger** (alias `challenger`) → **s'arrêter** : validation humaine, puis promotion manuelle (`python src/04_evaluation.py --promouvoir N`).
- Appeler les scripts numérotés avec `subprocess.run([sys.executable, "src/0X_....py", ...], check=True)` (un nom de fichier commençant par un chiffre n'est pas importable normalement). **Attention** : `05` renvoie 1 en cas de REFUS : le flow doit interpréter ce code comme un résultat métier (candidat refusé), pas comme une erreur technique (code 2 = erreur d'entrée).
- **Isolation (prérequis levé en v3)** : le flow écrit dans `models/candidats/<horodatage>/` avec `--modeles-dir`, `--reports-dir` (sous-dossier `reports`) et `--alias challenger`, puis appelle `05_porte_qualite.py --candidat models/candidats/<horodatage>/reports/04_evaluation_test.json --reference reports/04_evaluation_test.json`. Le script 02 (découpage) ne doit **pas** être rejoué dans le CT sur les mêmes données : cela réécrirait `train/val/test` (suivis par DVC) ; à décider avec l'utilisateur (le CT peut réutiliser les splits existants ou travailler sur une « fenêtre récente »).
- Déclencheurs : planning (cron/schedule Prefect) OU alerte de dérive (étape 8). En démonstration : `python pipeline/flow_ct.py --raison derive_donnees`.
- ~~Données de réentraînement : décision en attente~~ → **tranchée : A** (voir ci-dessus). Option B (« nouvelles données ») écartée : jeu statique de 486 339 lignes sans date, toutes déjà dans train/val/test ; une « fenêtre récente » serait simulée et risquerait une fuite vers le test.
- Limite à écrire dans le rapport : le jeu de test sert à juger chaque challenger (légère optimisation) ; en production on réserverait un jeu de validation frais ; une porte qui refuse puis un nouveau réglage réutilise le test.

#### 7.3 Porte de qualité : `src/05_porte_qualite.py` ✅ (validée chez l'utilisateur)

- Entrée : `--candidat` (obligatoire : `04_evaluation_test.json` du candidat, clés `metriques_test.test_f1_macro` et `metriques_test.test_recall_high`), `--reference` (optionnel : même format, modèle en production), `--f1-min` (0,45), `--rappel-high-min` (0,55), `--tolerance` (0,01), `--sortie` (défaut : `05_porte_qualite.json` à côté du rapport du candidat).
- Contrôles : F1 macro ≥ 0,45 (plancher « toujours low » = 0,236 ; modèle actuel 0,481) ; rappel de `high` ≥ 0,55 (actuel 0,622) ; si référence : F1 macro ≥ F1 de la référence − 0,01 (pas de régression). Note automatique si le candidat est équivalent à la référence (écart < 0,0005).
- Codes retour : **0 ACCEPTE, 1 REFUSE, 2 erreur d'entrée** (fichier illisible, métriques absentes ou NaN). Réussir la porte **ne déploie rien**.
- Testé par l'agent sur 6 scénarios (bac à sable) et par l'utilisateur sur le vrai candidat : ACCEPTE, code 0, note « équivalent ». Seuils à **justifier dans le rapport** (marge de 0,03 au-dessus de la production, rappel de `high` = métrique métier, tolérance 0,01 ≈ bruit entre deux entraînements). `ruff` n'a pas été exécuté sur ce fichier.

#### 7.4 CI GitHub Actions : minimale (à faire, D22)

- **Retenu** : un seul job `qualite` (ruff + pytest), sans aucune donnée. **Abandonnés** : le job d'entraînement sur données synthétiques et le job `docker` (le modèle est dans DVC, absent de Git, et le remote local est inaccessible depuis GitHub : `docker build` échouerait). L'image se construit en local.
- Fichiers à écrire : `.github/workflows/ci.yml` et `ruff.toml` (ruff ≥ 0.17 a un jeu de règles par défaut très large : fournir `select = ["E","F","W"]`, `line-length = 125`, `ignore = ["W292"]`, `extend-exclude = [".venv","mlartifacts","data","notebooks","models","logs"]`). Ne pas reformater le code de l'utilisateur pour contenter le linter.
- Dépendances à installer en CI : celles des tests (`src.api.main` et `src.features` : fastapi, pydantic, pandas, numpy, scikit-learn, joblib, **xgboost-cpu** (pas `xgboost` sous Linux), pytest, et **httpx** requis par `TestClient`). Vérifier le contenu de `requirements-api.txt` (a-t-il pytest/httpx ?) avant d'écrire le workflow. Python 3.12, versions épinglées comme chez l'utilisateur.
- Versions d'actions vérifiées par recherche le 2026-10-30 (à revérifier au moment d'écrire) : `checkout@v6` (une source mentionne `v7`), `setup-python@v6`. GitHub Actions n'a **jamais été exécuté** sur ce dépôt : le dire dans le rapport tant que ce n'est pas le cas.
- Pas de job de déploiement automatique (D15).

#### 7.5 Canari / test A/B / rollback

Traités dans le rapport (Q8) et dans la partie « cycle de vie » de l'exercice de conception. Démonstration optionnelle (à **proposer**, pas imposer) : un script qui route 90 % / 10 % des requêtes rejouées entre deux versions du modèle et compare latence, taux d'avertissements et F1 sur étiquettes retardées.

### Étape 8 : monitoring et dérive 🔲 à faire (plan validé par « oui vas y » ; détailler avec l'utilisateur avant de coder)

**Question de l'utilisateur : « sur quoi fait-on le monitoring ? »** Réponse à lui donner clairement : il n'y a **pas de vrai trafic de production** (jeu synthétique, aucun client réel). On fait donc du **rejeu** :

- **Référence** = `data/processed/train.csv` (vraies données d'entraînement).
- **« Production »** = lignes **réelles** tirées de `test.csv` (jamais vues à l'entraînement), envoyées **une par une à l'API réelle** par un script de rejeu (ex. `src/06_rejeu_production.py`) en fournissant `customer_id`, pour que `client_hash = sha256(customer_id)[:16]` (calculé par l'API) permette de rejoindre les étiquettes. L'API écrit `logs/predictions.jsonl` exactement comme en vrai. Les étiquettes réelles (`default_risk`) sont conservées à part par le script de rejeu, pour simuler des **étiquettes qui arrivent en retard**.
- Quatre scénarios, tous **étiquetés « SIMULÉ »** (sorties, dashboard, rapport) :
  - **A. Normal (témoin)** : lignes de test inchangées → le monitoring ne doit PAS alerter (prouve l'absence de fausses alertes).
  - **B. Data drift simulé** : décaler `avg_monthly_spend_ngn` (ex. × 1,6) et/ou baisser `recharge_frequency_per_month` → drift détecté sur ces variables ; l'API doit renvoyer des avertissements « hors plage » (> 50 000).
  - **C. Concept drift simulé** : features inchangées mais le lien features → étiquette change (ex. modifier une part des étiquettes `high` en `low` pour un segment) → F1 macro et rappel `high` chutent **sans** drift des entrées.
  - **D. Incident technique** : latence (délai artificiel contrôlé par variable d'environnement, ou charge) pour illustrer R10 (100 ms → 900 ms) : problème d'infrastructure, distinct du drift.
- Ce montage répond à R10 (latence ×9 = service ; distribution des montants = data drift ; F1 0,88 → 0,65 = dégradation de performance) et à Q7 (exemples concrets de data drift et de concept drift).
- **Surveiller** : technique (latence p50/p95, taux d'erreur, débit) ; données (data drift par variable : PSI et/ou test KS, valeurs manquantes, hors plage) ; modèle/métier (F1 macro, rappel de `high`, distribution des classes prédites, concept drift via les étiquettes qui arrivent avec retard).
- **Seuils proposés** (à justifier dans le rapport, ajustables) : PSI ≥ 0,1 avertissement, ≥ 0,2 alerte ; baisse de F1 macro > 0,05 vs validation = avertissement, > 0,10 = alerte ; latence p95 > 2× la référence = alerte.
- **Outils** : Evidently (son API a beaucoup changé selon les versions : `pip show evidently`, lire la doc de **cette** version, vérifier la compatibilité avec pandas 3.x / Python 3.12) ; **repli** : PSI et KS calculés avec numpy/scipy (faciles à expliquer dans le rapport). Dashboard : rapport HTML Evidently + petit tableau de bord (Streamlit ou HTML) à choisir avec l'utilisateur. Alertes : règles de seuil → `reports/alertes.json` + message en console/log. `/metrics` (Prometheus) non implémenté (optionnel).
- **Chaîne alerte → CT** : alerte → flow Prefect → challenger → porte de qualité → (canari) → **validation humaine** → promotion ou rollback.
- **Limites à écrire dans le rapport** : pas de trafic réel ; la dérive est injectée par nous (donc détectée par construction : on démontre la **capacité de détection** et les **seuils**, pas la fréquence réelle de dérive) ; variables tirées uniformément et indépendantes, donc peu de structure à faire dériver.

### Étape 9 : rapport DOCX/PDF 🔲 à faire

- Contenu : réponses **Q6, Q7, Q8**, **exercice de conception (i à iv)**, doc des prétraitements et du feature engineering, résultats (tableau A/B, métriques test, matrice de confusion), décisions et limites (jeu synthétique, déduplication, colonnes de scoring, `phone_number` retiré, test réutilisé par la porte, monitoring par rejeu avec dérive simulée), tableau « réel vs simulé ». Captures utiles : MLflow UI, Swagger `/docs`, `docker ps` (healthy), dashboard, matrice de confusion.
- Demander à l'utilisateur s'il faut aussi inclure les réponses de la Partie I (R1 à R10) et du QCM.
- Éléments d'appui déjà établis : QCM (section 1) ; pour R10 : latence ×9 = problème de service/infrastructure à traiter séparément, distribution des montants = **data drift**, F1 0,88 → 0,65 = **dégradation de performance** (conséquence probable de dérive de données ou de concept) ; on ne remplace pas automatiquement le modèle car dérive ≠ dégradation prouvée, risque d'entraîner sur des données non validées ou sans étiquettes fiables, besoin de validation, de canari et de rollback.
- Si le rapport doit être un `.docx`, lire le skill `docx` avant de créer le fichier. Les schémas d'architecture peuvent être produits en Mermaid ou SVG puis intégrés.


---

## 8. Pièges déjà rencontrés (ne pas les reproduire)

- `01_audit_nettoyage.py` : chemins par défaut `../data/...` ; ne s'exécute correctement que depuis `src/` (cas par défaut de PyCharm). Les nouveaux scripts utilisent `ROOT` calculé depuis `__file__`.
- MLflow 3.x : `log_model` sklearn utilise `skops` par défaut → forcer `serialization_format=cloudpickle`. Le paramètre `artifact_path` devient `name` dans les versions récentes (vérifier le warning).
- Un `Pipeline` assemblé à partir de composants déjà ajustés fonctionne pour `predict` / `predict_proba` ; ne pas le ré-ajuster (`fit`) par erreur.
- Au moment de l'inférence, les colonnes doivent porter exactement les noms d'entraînement.
- Installation de `mlflow` complet lente : `mlflow-skinny` suffit pour tester sans l'interface.
- `pytest.ini` **à la racine** avec `pythonpath = . src` ; placé dans `tests/` il provoque `ModuleNotFoundError: No module named 'src'` (les tests font `from src.api.main import app`).
- `src/api/main.py` utilise un import **relatif** (`from .schemas import ...`) et n'importe pas `features.py` : il se lance avec `uvicorn api.main:app --app-dir src` et se teste avec `src.api.main`. Le `.joblib` ne référence pas `features.py`.
- PowerShell : `>` écrit en **UTF-16** ; pour générer un fichier de dépendances, utiliser `| Set-Content -Encoding ascii fichier.txt`.
- Sous Linux, `pip install xgboost` tire ~300 Mo de bibliothèques NVIDIA : utiliser `xgboost-cpu` (même version) dans Docker/CI. Invisible sous Windows.
- `04_evaluation.py` écrase `models/model_final.joblib` et `decision_poids.json` et déplace l'alias `staging` : **ne pas le relancer sur le dépôt réel** sans isolation (`--modeles-dir`, `--reports-dir`, `--alias challenger`, disponibles depuis la v3) et ne pas faire `dvc repro` sans `--dry`. L'entraînement XGBoost n'est pas garanti identique bit-à-bit, donc le hash `version_modele` peut changer.
- `01_audit_nettoyage.py` avait des chemins `../data/...` (cassé depuis la racine, pouvait écrire hors du dépôt) : corrigé avec `ROOT = Path(__file__).resolve().parents[1]`. Validé en v3 : l'utilisateur a remplacé le fichier et le hash du CSV propre est identique à celui suivi par DVC.
- ruff ≥ 0.17 : jeu de règles par défaut très large ; imposer `ruff.toml` (voir 7.4).
- Evidently : l'API change fortement d'une version à l'autre ; ne jamais coder de mémoire.
- Environnement d'outils de l'agent (bac à sable) : `pkill -f <motif>` peut tuer son propre shell ; `/bin/sh` n'accepte ni l'expansion d'accolades `{a,b}` ni `time ( ... )` ; utiliser `bash` ou des commandes simples. Pour tester l'API en arrière-plan, lancer `uvicorn` avec `nohup ... &` et l'arrêter par PID.
- Les fichiers livrés en conversation ne sont pas dans le dépôt de l'utilisateur tant qu'il ne les a pas copiés : vérifier leur présence avant de s'y appuyer (exemple réel : l'utilisateur avait encore l'ancienne version de `01_audit_nettoyage.py` ; le schéma de la v2 supposait un `.gitignore` racine qui n'existait pas).
- **Excel et fichiers verrouillés (Windows)** : un CSV ouvert dans Excel (ou un aperçu PyCharm) provoque `[WinError 32]` lors d'opérations DVC (`dvc commit -f`). Fermer le fichier ; **ne jamais enregistrer le CSV brut depuis Excel** (le hash DVC changerait).
- **`sys.exit("message")` renvoie le code 1**, pas 2 : dans `05_porte_qualite.py`, les erreurs d'entrée passent par `print(..., file=sys.stderr)` puis `sys.exit(2)` pour ne pas être confondues avec un refus (code 1).
- **DVC** : sans `dvc.lock`, `dvc status` liste tout en « modified » (normal). `dvc commit -f` enregistre l'état existant dans `dvc.lock` **sans exécuter** les scripts, alors que `dvc repro` les rejoue (et `evaluation` a des effets de bord : voir règle 14). `dvc remove <fichier>.dvc` conserve les données (sans `--outs`). Un fichier suivi par un `.dvc` individuel ne peut pas être la sortie d'un stage. `dvc dag` affiche « Unable to find `less` in the PATH » : sans gravité. 7 objets poussés pour 8 sorties : `model_final.joblib` et `xgb_comportement.joblib` sont identiques.
- **PowerShell** : pour créer un fichier texte multiligne, utiliser un here-string `@' ... '@ | Set-Content -Encoding ascii fichier` (`>` écrit en UTF-16). Les avertissements Git « LF will be replaced by CRLF » sont sans gravité.
- **Prefect 3.9.0 (constaté chez l'utilisateur)** : un lancement direct démarre un serveur temporaire (≈ 20 s) puis l'arrête ; un avis de télémétrie anonyme s'affiche (désactivation : variable `DO_NOT_TRACK=1`) ; le journal du flow n'affiche que les 12 dernières lignes de chaque script, souvent des avertissements MLflow (cosmétique : le résumé complet n'est pas perdu, `ct_resume.json` garde les informations utiles).
- **Bac à sable de l'agent** : pas de réseau, ni `xgboost`, ni `mlflow`, ni `dvc`, ni `prefect`, ni `pytest`, ni `ruff` (le shell de l'outil est `sh` : `PIPESTATUS` n'existe pas, écrire un script et le lancer avec `bash`). Les tests de 03/04 se font avec des **doublures** (faux `xgboost` héritant de `BaseEstimator` et enveloppant un RandomForest ; faux `mlflow` en `MagicMock`) sur données jetables : ils ne valident que la plomberie des chemins. Le système de fichiers est réinitialisé entre les sessions (les doublures sont à recréer). Toujours dire ce qui a été testé et ce qui ne l'a pas été.

---

## 9. Décisions prises (registre)

| # | Décision | Justification | Réversible ? |
|---|---|---|---|
| D1 | Suppression de 13 661 `customer_id` dupliqués (première occurrence gardée) | Éviter l'ambiguïté et toute fuite client entre morceaux ; perte de 2,7 % | Oui (relancer 01, mais re-découper tout) |
| D2 | Retrait de `phone_number` et `assessment_date` | Donnée personnelle sans valeur prédictive ; colonne constante | Non souhaitable |
| D3 | Découpage 70/15/15 stratifié, aléatoire, seed 42 | Pas de dimension temporelle ; proportions de cible conservées | Non (invalide les runs) |
| D4 | Deux jeux de variables A et B, A par défaut | B contient des sorties de scoring (risque de circularité) | Oui, après comparaison |
| D5 | XGBoost comme modèle principal ; Dummy comme plancher | Choix de l'utilisateur ; adapté au tabulaire | Oui |
| D6 | Poids d'échantillons « balanced » + arrêt anticipé sur la validation | Déséquilibre modéré ; éviter le sur-apprentissage | Oui |
| D7 | Métriques de pilotage : F1 macro et rappel de `high` | L'accuracy est trompeuse | Non |
| D8 | MLflow avec backend SQLite local | Nécessaire au registre de modèles | Oui |
| D9 | **Jeu A « comportement » retenu** comme modèle principal | B ne dépasse pas A sur la validation (F1 macro 0,4543 vs 0,4589) et contient des sorties de scoring | Oui, à confirmer avec l'utilisateur si besoin |
| D10 | Poids de décision cherchés sur la validation (gain min 0,01) pour compenser la classe `medium` quasi jamais prédite | Le F1 macro est plafonné par `medium` ; réglage sur la validation, test évalué une seule fois ; à appliquer aussi dans l'API | Oui |
| D11 | L'API renvoie des `scores` (pas des probabilités), accepte les valeurs hors plage avec avertissement, hashe `customer_id` dans les journaux | Modèle non calibré ; détection précoce de dérive ; gouvernance des données personnelles | Oui |
| D12 | Évaluation, porte de qualité, CT et monitoring reposent sur les **vraies données** ; le synthétique n'est admis que pour tester du code et est toujours signalé | Retour de l'utilisateur ; sinon chiffres sans valeur pour le rapport | Non |
| D13 | Monitoring par **rejeu de lignes réelles de test** via l'API, avec dérives **simulées** et étiquetées (normal, data drift, concept drift, latence) | Pas de trafic réel ; permet de montrer détection et seuils | Oui, si l'utilisateur préfère un autre montage |
| D14 | `xgboost-cpu` (même version) dans l'image Docker | Évite ~300 Mo NVIDIA sous Linux ; `version_modele` identique (validé) | Oui |
| D15 | Aucune promotion ni déploiement automatique : promotion humaine après porte de qualité (+ canari) | R10.3, Q8 | Non |
| D16 | CI minimale (lint + tests) recommandée ; jobs smoke synthétiques en option | Retour de l'utilisateur ; **tranchée en D22** | Oui |
| D17 | Isolation des candidats par options CLI (`--modeles-dir`, `--reports-dir`, `--alias`), défauts inchangés ; alias `challenger` ; `--promouvoir` retire `staging` et `challenger` | Ne jamais écraser le modèle validé ; modification de 03/04 annoncée à l'utilisateur (règle 8) | Oui |
| D18 | Porte de qualité : F1 macro ≥ 0,45, rappel `high` ≥ 0,55, tolérance de régression 0,01 ; codes 0/1/2 ; réussir ne déploie rien | Marge au-dessus du plancher 0,236 et de la production 0,481 ; rappel de `high` = métrique métier ; tolérance ≈ bruit | Oui (seuils réglables, à justifier) |
| D19 | Remote DVC **local** `stockage_local` (`D:\dvc-stockage-impayes`, hors du projet) | Filet de sécurité avant restructuration ; limites : pas de protection contre une panne de disque, inaccessible depuis GitHub Actions | Oui (DagsHub, S3, etc.) |
| D20 | `dvc.yaml` à 4 stages ; petits rapports et `decision_poids.json` en `cache: false` (dans Git) ; `mlflow.db`/`mlartifacts` hors DVC ; CSV brut = entrée avec son `.dvc` | Lisibilité dans Git, reproductibilité, auditabilité via `dvc.lock` | Oui |
| D21 | `dvc commit -f` pour adopter l'état existant, **pas** `dvc repro` | `repro` rejouerait `evaluation` (écrase `model_final`, nouvelle version MLflow, déplace `staging`) | Non pour cette adoption |
| D23 | CT sur les splits existants (option A) ; verdict à 3 issues + données invalides ; le CT ne promeut jamais ; logique pure séparée du décorateur Prefect | Honnêteté (jeu statique), testabilité en CI sans Prefect, règle 14 | Oui (nouvelles données réelles en production) |
| D22 | CI minimale : un job ruff + pytest ; ni entraînement synthétique ni job docker | D12 ; le modèle est dans DVC et le remote local est inaccessible à GitHub | Oui, si remote accessible |

---

## 10. Journal d'avancement (à compléter par l'agent)

| Date | Étape | Ce qui a été fait | Résultats clés | Décisions / points ouverts |
|---|---|---|---|---|
| 2026-10-29 | 1 | Audit + nettoyage exécutés | 500 000 → 486 339 lignes ; pas de fuite directe (max F1 0,454) | D1, D2 |
| 2026-10-29 | 2 | Découpage 70/15/15 + `features.py` exécutés | train 340 437 / val 72 951 / test 72 951 ; 0 chevauchement | D3, D4 |
| 2026-10-30 | 3 | `03_entrainement.py` exécuté | A : F1 macro 0,4589 / AUC 0,6549 / rappel high 0,666 ; B : 0,4543 / 0,6477 / 0,695 ; dummy 0,2354 ; `medium` quasi jamais prédite | D5, D6, D8, D9 |
| 2026-10-30 | 4 | `04_evaluation.py` exécuté | Test : F1 macro 0,4813 (argmax 0,4587), AUC 0,6492 ; precision/rappel high 0,496/0,622 ; medium non identifiable (précision 0,233) ; registre v1 `staging` | D10 |
| 2026-10-30 | 5 | API FastAPI + tests écrits, testés (16 tests + serveur réel sur modèle synthétique) | **Lancement chez l'utilisateur en attente** | D11 |
| 2026-10-30 | 5 | Validation par l'utilisateur : `pytest -q` → 16 passed (après déplacement de `pytest.ini` à la racine) ; `/predict` testé en local | local : classe `high`, latence 89,5 ms, version modèle 6845f6fd1f73 | `pytest.ini` : `pythonpath = . src` |
| 2026-10-30 | 1 | Correctif de `01_audit_nettoyage.py` (chemins basés sur `ROOT`) | Non ré-exécuté chez l'utilisateur ; vérifier 486 339 lignes | règle 9 |
| 2026-10-30 | 6 | `Dockerfile`, `.dockerignore`, `requirements-api.txt` ; build et run chez l'utilisateur | `/predict` OK dans le conteneur, même version de modèle, latence ≈ 30 ms | D14 |
| 2026-10-30 | 7a | Brouillon CI livré en conversation (ci.yml, ruff.toml, generer_donnees_ci.py, 05_porte_qualite.py), validé en bac à sable | ruff OK, 16 tests OK, scripts 02-04 OK sur synthétique (23 s), 4 scénarios de la porte OK ; Actions jamais exécuté | **Non adopté** (retour utilisateur) : D12, D16. Étape 7 à redéfinir |
| 2026-10-30 | 7(i) | `03`/`04` : `--modeles-dir`, `--reports-dir`, `--alias` ; testé sur doublures (bac à sable) puis chez l'utilisateur | MD5 `model_final.joblib` inchangé (6845F6FD…), 16 tests OK, candidat v2 alias `challenger`, métriques identiques à la production | D17 ; `models/candidats/` à ignorer par Git |
| 2026-10-30 | 7(ii) | `05_porte_qualite.py` livré, testé sur 6 scénarios (bac à sable), validé chez l'utilisateur | ACCEPTE, code 0, note « équivalent à la référence », rappel high 0,6217 | D18 ; correctif : erreurs d'entrée = code 2 |
| 2026-10-30 | 7(iii) prép. | Constats : pas de remote DVC, pas de `.gitignore` racine, `01` en ancienne version. Correctifs livrés et exécutés | Hash du CSV propre identique à celui suivi par DVC ; `dvc push` 7 objets ; `dvc status -c` synchronisé ; commit `beb8797` | D19 |
| 2026-10-30 | 7(iii) | `dvc.yaml` livré (vérifié statiquement) ; `dvc remove` des 7 `.dvc` de sorties ; `dvc dag` conforme ; `dvc status` tout « modified » faute de lock | Données intactes | D20 |
| 2026-10-30 | 7(iii) D | `dvc commit -f` → `WinError 32` sur le CSV brut, puis `dvc status` « up to date » et `dvc repro --dry` : 4 stages inchangés | Lock probablement complet (hypothèse) | **Point ouvert** : vérifier `dvc.lock` et le CSV brut, fermer Excel ; D21 |
| 2026-10-30 | 7(iii) D | Vérification après fermeture d'Excel : `dvc commit -f` sans erreur ; `dvc.lock` : 4 stages ; MD5 CSV brut = md5 du `.dvc` ; `dvc status` up to date ; `dvc status -c` in sync | Point ouvert levé | Reste : commit Git (`git status --short` et vérification des fichiers indésirables non collés) |
| 2026-10-30 | 7(iii) git | `git status --short` : 7 `.dvc` supprimés ; non suivis : Dockerfile, .dockerignore, pytest.ini, requirements*.txt, data/.gitignore, dvc.yaml, dvc.lock, cc.md ; `.pyc` suivi par erreur | **Livrables de l'étape 6 jamais commités** | Commits à chemins explicites donnés ; exécution en attente |
| 2026-10-30 | — | Étape E sautée à la demande de l'utilisateur ; `pip show prefect` → 3.9.0 | — | Décision A/B (données du CT) en attente |
| 2026-10-30 | 7(iv) | Décision A (mêmes splits). `pipeline/ct_logique.py`, `pipeline/flow_ct.py`, `tests/test_ct_logique.py` livrés ; doc Prefect 3 consultée (flow/task, get_run_logger, serve+cron) | Testé en bac à sable seulement (faux prefect, doublures) : 28 cas, 3 mutations détectées, bout en bout REFUSE / ACCEPTE_EQUIVALENT / DONNEES_INVALIDES, production simulée intacte | D23 ; exécution chez l'utilisateur en attente (vrai Prefect 3.9.0, vrai pytest) |
| 2026-10-30 | 7(iv) | Validation chez l'utilisateur : `python pipeline/flow_ct.py --raison manuel` avec Prefect 3.9.0 sur les vraies données | `ACCEPTE_EQUIVALENT`, version 3 du registre (`challenger`), MD5 du modèle inchangé, `dvc status` propre. **`pytest -q` : 16 passed (attendu 44) : `test_ct_logique.py` non copié** | Reste : copier les tests ; vérifier `ct_resume.json` ; commits Git (état non communiqué) |
| 2026-10-30 | — | Handoff v3 rédigé | — | Prochaine session : section 6 |
| | | | | |
