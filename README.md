# Data Science Portfolio — David Membreno

CS student at California Lutheran University (B.S. Computer Science, Minor in Data Science) 
with a focus on building end-to-end analytical workflows: cleaning messy real-world data, 
surfacing patterns through visualization, and communicating findings clearly. These projects 
span Python and R, covering EDA, statistical modeling, and machine learning across Kaggle datasets.

---

## Projects

### 🎮 Steam Game Rating Classification (`steam_game_rating_classification.ipynb`)
**Tools:** Python, pandas, scikit-learn, seaborn, plotly  
Cleaned and processed 50K+ Steam game records, engineered features from datetime and boolean 
columns, and handled class imbalance across three rating categories. Compared Random Forest, 
Gradient Boosting, and KNN classifiers using cross-validation. GBM came out on top at 71.5% 
accuracy. Feature importance analysis identified user review count and release year as the 
strongest predictors.

### ☕ Coffee Reviews EDA (`coffee_reviews_eda_python.ipynb`)
**Tools:** Python, pandas, seaborn, plotly, nltk, WordCloud  
Explored 1,200+ coffee reviews from CoffeeReview.com. Cleaned nulls, standardized categories, 
and analyzed distributions across roast type, origin country, price, and rating. Built a word 
cloud from lemmatized review text and used choropleth maps to visualize roaster and origin 
density globally. Found that medium-light roasts dominate and Ethiopia leads in bean origin 
by a wide margin.

### 🎬 IMDb Movie EDA (`imdb_movie_eda_r.ipynb`)
**Tools:** R, ggplot2, dplyr, tidyverse  
Analyzed 10K+ IMDb movie records covering budget, revenue, score, genre, language, and country. 
Cleaned and recoded variable types, removed duplicates, and built univariate and bivariate 
visualizations. Found a weak budget-to-revenue relationship and noted Australia's outsized 
presence in the dataset as an open question worth further investigation.

### ☕ Coffee Quality Regression (`coffee_quality_regression_r.ipynb`)
**Tools:** R, ggplot2, randomForest, gbm, corrplot  
Used Coffee Quality Institute data to predict flavor scores. Applied double Z-score outlier 
removal, built a correlation heatmap, and compared Linear Regression, Random Forest, and GBM 
models. GBM performed best (R² = 0.842), with Aftertaste contributing 56% of predictive power. 
Written findings include practical recommendations for coffee producers and roasters.

### 🎙️ YouTube Toxicity Pipeline v1 (`youtube_toxicity_pipeline_v1.ipynb`)
**Tools:** Python, transformers, pandas, sklearn  
First iteration of a multi-stage YouTube comment moderation pipeline. Combined and cleaned 
multiple Kaggle datasets, benchmarked several classification approaches, and selected a 
pretrained RoBERTa toxicity classifier as the core model. This prototype was iterated into 
a full deployment-ready system with a Streamlit UI — see 
[youtube-comment-moderation](https://github.com/DavidMembreno/youtube-comment-moderation) 
for the complete version.

---

## Data
All datasets are stored in the `data/` directory. File paths in each notebook reference 
this location via raw GitHub URLs — no local setup needed, just run the cells.

---

## Stack
- **Python:** pandas, NumPy, scikit-learn, seaborn, matplotlib, plotly, nltk
- **R:** ggplot2, dplyr, tidyverse, randomForest, gbm, corrplot
