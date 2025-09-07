# =============================================================================
# Step 1: Setup and Data Loading (Upgraded to MovieLens 1M)
#
# We are now using the ml-1m dataset, which is larger and uses a different
# format. We need to adjust how we read the files.
# =============================================================================

import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import zipfile
from io import BytesIO

print("--- Step 1: Setup and Data Loading (MovieLens 1M) ---")

# Download the MovieLens 1M dataset
print("Downloading MovieLens 1M dataset...")
url = "https://files.grouplens.org/datasets/movielens/ml-1m.zip"
response = requests.get(url)
with zipfile.ZipFile(BytesIO(response.content)) as z:
    z.extractall(".")
print("Dataset downloaded and extracted successfully.")

# Define column names for the ratings data (UserID::MovieID::Rating::Timestamp)
r_cols = ['user_id', 'movie_id', 'rating', 'timestamp']
# The file has no header and uses '::' as a separator
ratings = pd.read_csv('ml-1m/ratings.dat', sep='::', names=r_cols, engine='python', encoding='latin-1')

# Define column names for the movie data (MovieID::Title::Genres)
m_cols = ['movie_id', 'title', 'genres']
movies = pd.read_csv('ml-1m/movies.dat', sep='::', names=m_cols, engine='python', encoding='latin-1')

# Merge the ratings and movies dataframes
data = pd.merge(ratings, movies, on='movie_id')

# Display the first few rows of the merged dataset
print("\n--- First 5 rows of the merged dataset ---")
print(data.head())


# =============================================================================
# Step 2: Data Preprocessing and User-Item Matrix Creation
# =============================================================================

print("\n--- Step 2: Creating the User-Item Matrix ---")

user_movie_matrix = data.pivot_table(index='user_id', columns='title', values='rating')


# =============================================================================
# Step 3: Filter Movies Based on Popularity Threshold
#
# With a larger dataset, filtering by popularity is even more crucial to
# ensure recommendations are based on a reasonable number of opinions.
# =============================================================================

print("\n--- Step 3: Filtering Movies by Popularity ---")

# Calculate the number of ratings for each movie
movie_ratings_count = data['title'].value_counts()

# Increase the minimum ratings threshold for the larger dataset
MINIMUM_RATINGS = 50
popular_movies = movie_ratings_count[movie_ratings_count >= MINIMUM_RATINGS].index

# Filter the user-movie matrix to only include popular movies
user_movie_matrix_popular = user_movie_matrix[popular_movies]

print(f"Original number of movies: {user_movie_matrix.shape[1]}")
print(f"Number of movies after filtering (rated by at least {MINIMUM_RATINGS} users): {user_movie_matrix_popular.shape[1]}")


# =============================================================================
# Step 4: Calculating User Similarity
#
# This step may take slightly longer due to the larger number of users
# (approx. 6000 vs. 943 in the 100k dataset).
# =============================================================================

print("\n--- Step 4: Calculating User Similarity on Filtered Data ---")

# We fill NaN with 0 before calculating similarity
user_movie_matrix_filled = user_movie_matrix_popular.fillna(0)
user_similarity = cosine_similarity(user_movie_matrix_filled)
user_similarity_df = pd.DataFrame(user_similarity,
                                  index=user_movie_matrix_popular.index,
                                  columns=user_movie_matrix_popular.index)

print("\n--- User Similarity Matrix (first 5 users) ---")
print(user_similarity_df.head())


# =============================================================================
# Step 5: Generating Robust Movie Recommendations
# The function remains the same, but it now operates on the larger,
# more reliable dataset.
# =============================================================================

print("\n--- Step 5: Defining the Robust Recommendation Function ---")

def get_movie_recommendations(user_id, num_recommendations=10):
    """
    Generates robust movie recommendations using the filtered ml-1m dataset.
    """
    print(f"\n--- Generating Top {num_recommendations} Recommendations for User {user_id} ---")

    similar_users = user_similarity_df[user_id].sort_values(ascending=False).iloc[1:]
    movies_watched = user_movie_matrix_popular.loc[user_id].dropna().index

    recommendations = {}

    for movie in user_movie_matrix_popular.columns:
        if movie not in movies_watched:
            weighted_sum = 0
            similarity_sum = 0
            for other_user, similarity_score in similar_users.items():
                if pd.notna(user_movie_matrix_popular.loc[other_user, movie]):
                    weighted_sum += similarity_score * user_movie_matrix_popular.loc[other_user, movie]
                    similarity_sum += similarity_score
            if similarity_sum > 0:
                recommendations[movie] = weighted_sum / similarity_sum

    recommendations_df = pd.DataFrame.from_dict(recommendations, orient='index', columns=['predicted_rating'])
    recommendations_df['match_score_percent'] = (recommendations_df['predicted_rating'] / 5 * 100).round(2)
    recommendations_df.sort_values(by='predicted_rating', ascending=False, inplace=True)

    if recommendations_df.empty:
        return "Could not generate recommendations for this user."

    return recommendations_df.head(num_recommendations)

# =============================================================================
# Step 6: Getting and Visualizing the Final Recommendations
#
# Let's get recommendations for the same user (user_id=1) and see how they
# differ with the larger dataset.
# =============================================================================

print("\n--- Step 6: Generating and Visualizing Robust Recommendations ---")

sample_user_id = 1
top_recs = get_movie_recommendations(sample_user_id, num_recommendations=10)

print(f"\nTop 10 Recommended Movies for User {sample_user_id} (from MovieLens 1M):")
print(top_recs)

# Plotting the recommendations
if isinstance(top_recs, pd.DataFrame):
    plt.figure(figsize=(12, 8))
    sns.barplot(x=top_recs['predicted_rating'], y=top_recs.index, palette='plasma')
    plt.title(f'Top 10 Movie Recommendations for User {sample_user_id}')
    plt.xlabel('Predicted Rating (1-5 Scale)')
    plt.ylabel('Movie Title')
    plt.xlim(0, 5)
    plt.show()
