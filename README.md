# movie-recommender-system-tmdb-dataset
A content based movie recommender system using cosine similarity

## Project Overview VideoLink =

## Dataset Link = https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata




# 🔹 Movie Recommendation System (Content-Based) 🔹

[![Github Repo](https://img.shields.io/badge/GitHub-Tetrapico%2FML_Project-blue?logo=github)](https://github.com/Tetrapico/ML_Project)



## 🔹 Project Overview 🔹

In today’s digital era, choosing a movie to watch can be a challenging task due to the vast number of options. This project aims to solve that by providing a way for users to discover new films based on their preferences.
Using the **TMDb Movie Metadata** dataset, we extract semantic information (genres, keywords, overviews) for each movie and compute their similarity scores to enable personalized recommendations.


## 🔹 Methodology 🔹

* **Data Preprocessing:**

  * Text cleaned by lowercasing and removing punctuation
  * Stopwords removed
* **Vectorization (TF-IDF):**
  Each movie’s content is represented as a numerical vector.
* **Similarity Computation (Cosine Similarity):**
  Pair-wise similarity scores are computed to identify the most similar films.
* **Recommendation:**
  Given a movie title, the algorithm ranks and produces the most similar films.


## 🔹 Tech Stack 🔹

* **Python 3.x**
* **Pandas** — for data manipulation
* **Scikit-learn** — for TF-IDF vectorization and cosine similarity
* **Jupyter notebook** — for interactive exploration
* **TMDb Movie Metadata** — for the dataset (available on Kaggle)


## 🔹 Installation 🔹

1. Clone this repository:

```bash
git clone https://github.com/Tetrapico/ML_Project.git
cd ML_Project
```

2. Install the required packages:

```bash
pip install -r requirements.txt
```



## 🔹 How to Use 🔹

* Open `movie_recommender.ipynb` in **Jupyter notebook**.
* Run all cells in the notebook.
* Call the `recommend()` function with a movie title to retrieve its top 5 most similar films.



## 🔹 Future Improvement Ideas 🔹

* Implement **collaborative filtering** alongside content-based filtering.
* Utilize **Word Embeddings (Word2Vec, BERT)** for semantic understanding.
* Include additional metadata (actor, crew, box office) to aid in recommendations.



## 🔹 References 🔹

* [TMDb Movie Metadata](https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata)
* [Scikit-learn TF-IDFVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)
* [Scikit-learn Cosine Similarity](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.cosine_similarity.html)



