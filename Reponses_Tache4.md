# Analyse des Expérimentations Random Forest (Tâche 4)

Voici le rapport détaillé des expérimentations effectuées sur le modèle **Random Forest**, suivies sur **MLflow** dans le cadre de votre projet de classification (Hit/Flop).

---

## 1. Feature Importances (Importance des variables)

Le graphique des importances a été généré par le script Python `src/run_rf_analysis.py` (sauvegardé sous `src/feature_importances.png` et sur MLflow).

**Répartition des scores d'importance :**
- **vote_count :** 0.3079
- **popularity :** 0.2390
- **budget :** 0.1800
- **runtime :** 0.1432
- **vote_average :** 0.1299

**Quelles sont les 3 variables les plus importantes pour la classification ? Cela correspond-il à votre compréhension des données ?**
> Les 3 variables les plus décisives sont **vote_count**, **popularity**, et **budget**.
> Cela correspond parfaitement à la réalité de l'industrie cinématographique : 
> le *vote_count* (volume d'engagement global) et la *popularity* sont les reflets directs de la taille de l'audience touchée. Plus un film suscite d'engagement public, plus ses chances de rentabilité sont élevées. Le *budget* en troisième position définit la barre que les revenus doivent dépasser (la profitabilité dépend directement du coût initial).

---

## 2. Stabilité des prédictions

En testant différents états aléatoires (`random_state = [1, 23, 42, 99, 2024]`), nous obtenons les exactitudes suivantes (sur le Test Set) :
- Random State `1` : 0.7817
- Random State `23` : 0.7833
- Random State `42` : 0.7802
- Random State `99` : 0.7817
- Random State `2024` : 0.7848

**Moyenne :** 0.7824 | **Écart-type :** 0.0016

**En utilisant random_state différent, observez-vous une grande variabilité dans les résultats ? Que cela dit-il sur la robustesse du modèle ?**
> Non, la variabilité est quasi-nulle (l'écart-type n'est que de 0.0016). Cela prouve que le modèle Random Forest est extrêmement **robuste**. Grâce à sa structure ensembliste (*bagging*), le résultat final résulte d'un vote majoritaire sur de multiples arbres, ce qui neutralise la forte variance que l'on aurait pu observer sur un arbre de décision unique.

---

## 3. Analyse des erreurs

Trois exemples mal classés par le modèle aléatoire (`random_state=42`) :

**Exemple 1 : *Saving Silverman*** (Vrai = 0 (Flop), Prédit = 1 (Hit))
- **Stats :** Budget: 22M $, Popularité: ~13, Votes: 154
- **Analyse :** Le modèle a vu un budget confortable et des indicateurs de communauté passables, supposant un "Hit". Cependant, le film n'a pas couvert son budget assez élevé.

**Exemple 2 : *Beyond the Valley of the Dolls*** (Vrai = 1 (Hit), Prédit = 0 (Flop))
- **Stats :** Budget très bas: 2M $, Popularité: ~4.3, Votes: 53
- **Analyse :** Le modèle a vu des indicateurs d'engagement (popularité / votes) très faibles et a immédiatement prédit un Flop. Mais c'était un film à très micro-budget qui fut facile à rentabiliser malgré une petite base de fans.

**Exemple 3 : *Attack the Block*** (Vrai = 0 (Flop), Prédit = 1 (Hit))
- **Stats :** Budget: 14.3M $, Popularité très haute: 31, Votes très hauts: 733.
- **Analyse :** Tous les signaux extérieurs criaient "Succès" (fort intérêt critique/publicistique), poussant l'algorithme vers 1. Toutefois, il n'a financièrement pas rapporté plus de 14.3M $.

**Y a-t-il des patterns dans ces erreurs ?**
> **Oui.** Le modèle échoue souvent sur les *outliers (anomalies)* de l'industrie :
> 1. Les films indépendants / cultes à micro-budget ("Hit" facile en ratio, mais métriques très faibles qui trompent le modèle).
> 2. Les succès d'estime ou viraux qui ont une popularité massive (souvent sur internet) mais de très mauvaises ventes de billets en salle.

---

## 4. Biais et Variance

Tableau d'analyse :

| n_estimators | max_depth | Train Accuracy | Test Accuracy | Biais | Variance |
| :---: | :---: | :---: | :---: | :---: | :---: |
| 10 | 2 | 0.7894 | 0.7864 | Haut | Faible |
| 100 | 10 | 0.9253 | 0.7926 | Faible | Modérée |
| 200 | None | 1.0000 | 0.7740 | Très faible | Haute |

**Quel paramétrage montre overfitting ?**
> `n_estimators = 200, max_depth = None`. Le score d'entraînement est parfait (1.0000 - le modèle a appris les données par cœur), mais le score de test chute (0.7740), ce qui indique une très haute variance (surapprentissage).

**Quel paramétrage montre underfitting ?**
> `n_estimators = 10, max_depth = 2`. Le modèle est beaucoup trop bridé. L'accuracy d'entraînement est relativement faible et proche de celle de test (0.7894). L'algorithme a un "Biais haut", il simplifie trop la réalité.

**Quel paramétrage semble équilibré ?**
> `n_estimators = 100, max_depth = 10`. Il donne les meilleures performances en test (0.7926) tout en gardant une compréhension fluide du set d'entraînement. C'est le meilleur compromis Biais/Variance.

---

## 5. Comparaison avec l'algorithme Arbre de Décision

- **Arbre de décision standard (Test Accuracy) :** 0.7523
- **Random Forest par défaut (Test Accuracy) :** 0.7802

**Conclusion :** 
Le `Random Forest` surpasse l'arbre de décision classique d'environ **3 points** de précision. L'arbre de décision unique est très vulnérable au surapprentissage (variance élevée) sur des données multidimensionnelles, ses feuilles mémorisant trop vite le bruit. En faisant la moyenne (*bagging*) de dizaines d'arbres légèrement différents, le Random Forest atténue naturellement cette variance, généralise beaucoup mieux et offre donc des résultats de Test supérieurs.
