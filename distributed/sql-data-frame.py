from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext('local')
sqlContext = SQLContext(sparkContext=sc)

df = sqlContext.read.json('./data/country-population.json')

df.filter(df.population >= 500000000).sort(df.population.desc()).show(n=40)