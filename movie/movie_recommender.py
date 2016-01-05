import os
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS

conf = SparkConf()
sc = SparkContext()


def parse_line_by_comma(line):
    line = line.strip()
    start = line.find('"')

    if start == -1:
        return line.split(",")
    end = line.find('"', start + 1)
    return [line[:start - 1], line[start:end + 1], line[end + 2:]]


data_path = os.path.join('.', 'data/small')

ratings_raw_data = sc.textFile(os.path.join(data_path, 'ratings.csv'))
ratings_raw_data_header = ratings_raw_data.first()

ratings_data = ratings_raw_data \
    .filter(lambda line: line != ratings_raw_data_header) \
    .map(lambda line: line.split(",")) \
    .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2])))

movies_raw_data = sc.textFile(os.path.join(data_path, 'movies.csv'))
movies_raw_data_header = movies_raw_data.first()

movies_data = movies_raw_data \
    .filter(lambda line: line != movies_raw_data_header) \
    .map(parse_line_by_comma) \
    .map(lambda tokens: (int(tokens[0]), tokens[1]))

print "\nAll movies quantity=%s" % movies_data.count()

user_id = 0
user_ratings = [
    (user_id, 260, 4),  # Star Wars (1977)
    (user_id, 318, 3),  # Shawshank Redemption, The (1994)
    (user_id, 16, 3),  # Casino (1995)
    (user_id, 101142, 1),  # "Croods, The (2013)
    (user_id, 379, 1),  # Timecop (1994)
    (user_id, 296, 3),  # Pulp Fiction (1994)
    (user_id, 858, 5),  # Godfather, The (1972)
    (user_id, 7458, 4)  # Troy (2004)
]

user_ratings_data = sc.parallelize(user_ratings)

ratings = ratings_data.union(user_ratings_data)

best_lambda = 0.1
best_rank = 30
best_iteration = 20

model = ALS.train(ratings=ratings, rank=best_rank, iterations=best_iteration, lambda_=best_lambda, seed=5L)

user_rated_movies_ids = map(lambda r: r[1], user_ratings)

user_unrated_movies = movies_data \
    .filter(lambda movie: movie[0] not in user_rated_movies_ids) \
    .map(lambda x: (user_id, x[0]))

recommended_movies_data = model \
    .predictAll(user_unrated_movies) \
    .map(lambda r: (r[1], r[2]))

recommended_movies_titles_data = recommended_movies_data \
    .join(movies_data) \
    .map(lambda x: (x[1][1], x[1][0]))

recommended_20_movies = recommended_movies_titles_data \
    .takeOrdered(20, key=lambda x: -x[1])

print "\n\r------------Recommended movies for user with id = %s ----------------------" % user_id

i = 0
for movie_title, movie_rating in recommended_20_movies:
    i += 1
    print("%2d: %s %5f" % (i, movie_title.encode('utf-8'), movie_rating))

print "----------------------------------------------------------------------------"