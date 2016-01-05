from pyspark import SparkContext
from pyspark.streaming import StreamingContext

sc = SparkContext()
ssc = StreamingContext(sparkContext=sc, batchDuration=5)

lines = ssc.socketTextStream('localhost', 9999)

word_counts = lines \
    .flatMap(lambda line: line.split(" ")) \
    .map(lambda word: (word, 1)) \
    .reduceByKey(lambda x, y: x + y)

word_counts.pprint()

ssc.start()
ssc.awaitTermination()

# to connect rc -lk 9999