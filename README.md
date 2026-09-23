# DataVine Analytics


DataVine Analytics is a data science workspace for exploratory data analysis, statistical modeling, and actionable business insights.

## Summative Lab: Three Client ML Prototypes

This repo contains the Summative Lab notebook built for DataVine Analytics' boutique consulting practice. As a junior data scientist, the notebook develops three prototype machine learning solutions for three client projects, each following the same standardized workflow: **data preparation → dimensionality reduction (PCA) → model implementation & hyperparameter tuning → evaluation & visualization.**

| # | Client Project | Dataset | Technique |
|---|---|---|---|
| 1 | Wine Classification System | Wine | k-NN + PCA + `GridSearchCV` |
| 2 | Agricultural Feed Recommendation Engine | Chickwts | PCA + Cosine Similarity |
| 3 | Regional Crime Pattern Analysis | USArrests | K-Means & GMM + PCA + Feature Selection |

## Key Results

- **Wine k-NN:** PCA reduces 13 correlated chemical features to 10 components (96.2% variance retained); `GridSearchCV`-tuned k-NN (`euclidean`, `n_neighbors=9`, `distance`-weighted) achieves **100% test accuracy**.
- **Chickwts Recommender:** Feeds split into a high-growth group (sunflower, casein, meatmeal) and a low-growth group (soybean, linseed, horsebean) based on a PCA-compressed weight profile, with cosine-similarity recommendations for each feed type.
- **USArrests Clustering:** Top 3 features (`Assault`, `UrbanPop`, `Rape`) selected by variance, reduced to 2 PCA components (89.6% variance retained). The elbow method (`kneed`) and BIC are used to size K-Means and GMM respectively; K-Means achieves a silhouette score of 0.416 across 4 interpretable regional crime clusters.

## Project Structure

```text
├── Data/                          # Raw datasets (wine.csv, chickwts.csv, USArrests.csv)
├── Notebook/
│   └── DataVine_Analytics.ipynb   # Main analysis notebook (all 3 client projects)
├── environment.yml                # Conda environment definition
├── .gitignore                     # Git ignore rules
└── README.md                      # Project documentation
```

## Setup

```bash
conda env create -f environment.yml
conda activate datavine
```

**Note:** the notebook additionally depends on two packages not yet listed in `environment.yml`'s `pip:` section — add them before running:

```bash
pip install kneed rdatasets
```

`kneed` is used to programmatically detect the K-Means elbow point (`KneeLocator`) rather than reading it off the plot by eye. `rdatasets` is used to load the Chickwts and USArrests datasets; the Wine dataset is loaded via `sklearn.datasets.load_wine()`. (Local copies of all three datasets are also provided under `Data/` if you'd prefer to load from CSV instead.)

Then launch:

```bash
jupyter lab
```

and open `Notebook/DataVine_Analytics.ipynb`.

## Notebook Workflow

1. **Data Preparation** — load Wine, Chickwts, and USArrests; check for missing values, duplicates, and inconsistent categories; standardize numeric features (z-score); summarize structure.
2. **k-NN Classification (Wine)** — encode target labels, apply PCA retaining 95% variance, tune k-value/distance metric/weighting via `GridSearchCV`, train and evaluate the final classifier (classification report, accuracy, confusion matrix, PCA scatter).
3. **Recommendation System (Chickwts)** — standardize weight, reduce to 1 principal component, compute cosine similarity between feed-type profiles, generate top-N feed recommendations.
4. **Clustering (USArrests)** — standardize features, select the top 3 by variance, reduce to 2 principal components, determine cluster count via the elbow method (K-Means) and BIC (GMM), fit both models, visualize and compare, profile clusters by state.
5. **Evaluation & Interpretation** — a stakeholder-facing summary pulling every reported metric directly from the fitted models and computed variables above.

## Known Limitations

- The Chickwts dataset has only one numeric attribute (chick weight), so the PCA-reduced feed profile is effectively 1-dimensional; cosine similarity on this single axis distinguishes "above/below average growth" rather than a nuanced multi-attribute similarity. A production recommender would benefit from additional feed attributes (protein %, cost, ingredient composition).
- On USArrests (50 states, 2 PCA dimensions), BIC's complexity penalty favors a single Gaussian component (k=1) for GMM, diverging from the K-Means elbow (k=4). The notebook documents this discrepancy explicitly and uses k=4 for both models to allow a direct, interpretable comparison.

## License

MIT
