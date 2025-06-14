# movie-recommender-system-tmdb-dataset

## A content based movie recommender system using cosine similarity
![file_00000000922861f9914a6fd7676b03a4](https://github.com/user-attachments/assets/31919d4c-e784-4d4b-a681-3723ec6973c0)
![20250614_064836](https://github.com/user-attachments/assets/a01033e9-00b8-4c1f-9ebd-a5faad42485a)
![20250614_063720](https://github.com/user-attachments/assets/52dc25ce-da61-458b-81b7-fe28fd560d6c)
![20250614_063842](https://github.com/user-attachments/assets/09239a4f-4373-4ccb-a5b0-c40c2959b36e)
![20250614_064003](https://github.com/user-attachments/assets/18362c33-0f9e-4a63-bcb3-c3d990d1d417)
![20250614_064059](https://github.com/user-attachments/assets/472da3dd-e39a-4d99-a4e4-fa6f0ec2924a)
![Screenshot_20250614-062806](https://github.com/user-attachments/assets/cd434380-0289-40e8-89c5-e30600137023)
![file_00000000922861f9914a6fd7676b03a4](https://github.com/user-attachments/assets/31919d4c-e784-4d4b-a681-3723ec6973c0)



## Project Overview VideoLink  https://drive.google.com/file/d/1_DSbAr7hg9L06LuBAsqyoxvKmy0C7qKC/view?usp=drivesdk

## Dataset Link  https://www.kaggle.com/datasets/tmdb/tmdb-movie-metadata




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



