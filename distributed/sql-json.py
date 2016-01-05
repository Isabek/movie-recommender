from pyspark import SparkContext
from pyspark.sql import SQLContext

sc = SparkContext('local')
sqlContext = SQLContext(sc)

df = sqlContext.read.json('./data/country-population.json')
df.registerTempTable('cp')

countries_rdd = sqlContext \
    .sql("SELECT * FROM cp WHERE population >= 5000000 AND country LIKE 'T%'") \
    .cache()

countries_rdd.show()