import itertools
import math
import datetime
from pyspark import SparkContext, SparkConf
from pyspark.mllib.recommendation import ALS
from termcolor import colored


def parse_line_by_comma(line):
    line = line.strip()
    start = line.find('"')

    if start == -1:
        return line.split(",")
    end = line.find('"', start + 1)
    return [line[:start - 1], line[start:end + 1], line[end + 2:]]


def get_day_from_timestamp(timestamp):
    return datetime.datetime.fromtimestamp(timestamp=timestamp).strftime("%A")

sc = SparkContext('local')

ratings_raw_data = sc.textFile('./data/small/ratings.csv')
ratings_raw_data_header = ratings_raw_data.first()

ratings_data = ratings_raw_data \
    .filter(lambda line: line != ratings_raw_data_header) \
    .map(lambda line: line.split(",")) \
    .map(lambda tokens: (int(tokens[0]), int(tokens[1]), float(tokens[2]))) \
    .cache()

training_data, validation_data, test_data = ratings_data.randomSplit([6, 2, 2])

validation_data_for_predict = validation_data.map(lambda x: (x[0], x[1]))
test_data_for_predict = test_data.map(lambda x: (x[0], x[1]))

ranks = [10, 20, 30]
lambdas = [0.1, 0.01]
iterations = [10, 20]

best_validation_rmse = float('inf')
best_rank = 0
best_lambda = -1.0
best_num_iteration = -1
best_model = None

for lambda_, iteration, rank in itertools.product(lambdas, iterations, ranks):
    model = ALS.train(training_data, rank=rank, iterations=iteration, lambda_=lambda_, seed=5L)

    predictions = model \
        .predictAll(validation_data_for_predict) \
        .map(lambda r: ((r[0], r[1]), r[2]))

    rates_and_predictions = validation_data \
        .map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))) \
        .join(predictions)

    rmse = math.sqrt(rates_and_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

    print '\rFor lambda=%s, iteration=%s, rank=%s, RMSE=%s' % (
        colored(lambda_, 'magenta'), colored(iteration, 'yellow'), colored(rank, 'blue'), colored(rmse, 'green'))

    if best_validation_rmse > rmse:
        best_validation_rmse = rmse
        best_rank = rank
        best_num_iteration = iteration
        best_lambda = lambda_
        best_model = model

print colored('------------------------Best Parameters--------------------------', 'red')
print 'The best model was trained with rank=%s, lambda=%s, iteration=%s' % (best_rank, best_lambda, best_num_iteration)
print colored('-----------------------------------------------------------------', 'red')

print colored('----------------------RMSE for validation------------------------', 'blue')
print 'For validation data the RMSE=%s' % best_validation_rmse
print colored('-----------------------------------------------------------------', 'blue')

test_predictions = best_model \
    .predictAll(test_data_for_predict) \
    .map(lambda r: ((r[0], r[1]), r[2]))

test_rates_and_predictions = test_data \
    .map(lambda r: ((int(r[0]), int(r[1])), float(r[2]))) \
    .join(test_predictions)

test_error = math.sqrt(test_rates_and_predictions.map(lambda r: (r[1][0] - r[1][1]) ** 2).mean())

print colored('-------------------------RMSE for test---------------------------', 'yellow')
print 'For test data the RMSE=%s' % test_error
print colored('-----------------------------------------------------------------', 'yellow')