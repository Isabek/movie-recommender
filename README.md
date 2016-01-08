# Simple Movie Recommendation System

It is a Movie Recommender System on Spark. It uses Collobrative Filtering.

Also it containes simple examples for Spark modules.

## Find best parametres for model(Collobrative Filtering)

First of all you should find parameteres like `lambda`, `iteration number`, `rank` for the ALS in Spark (`movie/model_train.py`)

## Reccommend movies for users based on their ratings.

Lets say, we want to recommend movies for user with id = 0 which does not exist in our database. (`movie/movie_recommender.py`)
