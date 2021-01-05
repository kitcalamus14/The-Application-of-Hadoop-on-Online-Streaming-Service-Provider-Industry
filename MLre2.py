#Import Relevant Packages
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.functions import lit
import subprocess
#Loading and transforming movie info file from hdfs
def loadMovieNames():
    movieNames = {}
    cat = subprocess.Popen(["hadoop", "fs", "-cat", "/user/maria_dev/ml-100k/u.item"], stdout=subprocess.PIPE)
    with cat.stdout as f:
        for line in f:
            fields = line.split('|')
            movieNames[int(fields[0])] = fields[1].decode('ascii', 'ignore')
    return movieNames
#Function converting u.data file into a structured format
def parseInput(line):
    fields = line.value.split()
    return Row(userID = int(fields[0]), movieID = int(fields[1]), rating = float(fields[2]))
#main ML process
if __name__ == "__main__":
    spark = SparkSession.builder.appName("MovieRecs").getOrCreate()
    #load movie ID and movie names
    movieNames = loadMovieNames()
    # Get the raw data hdfs filename:u.data
    lines = spark.read.text("hdfs:///user/maria_dev/ml-100k/u.data").rdd
    # Convert it to a RDD of Row objects with (userID, movieID, rating)
    ratingsRDD = lines.map(parseInput)
    # Convert to a DataFrame and cache it for iteration process when needed
    ratings = spark.createDataFrame(ratingsRDD).cache()
    # Create an ALS collaborative filtering model from the complete data set
    als = ALS(maxIter=5, regParam=0.01, userCol="userID", itemCol="movieID", ratingCol="rating")
    model = als.fit(ratings)
    # Print out ratings from user 0:
    print("\nRatings for user ID 0:")
    userRatings = ratings.filter("userID = 0")
    for rating in userRatings.collect():
        print movieNames[rating['movieID']], rating['rating']
    user0list = []
    for rating in userRatings.collect():
        user0list.append(rating['movieID'])
        
    print("\nTop 40 movie recommendations:")
    # Find movies rated more than 200 times
    ratingCounts = ratings.groupBy("movieID").count().filter("count > 200")
    # Construct a "test" dataframe for user 0 with every movie rated more than 100 times
    popularMovies = ratingCounts.select("movieID").withColumn('userID', lit(0))

    # Run our model on that list of popular movies for user ID 0
    recommendations = model.transform(popularMovies)

    # Get the top 20 movies with the highest predicted rating for this user
    topRecommendations = recommendations.sort(recommendations.prediction.desc()).take(40)

    for recommendation in topRecommendations:
        if recommendation['movieID'] in user0list:
            pass
        else:
            print (movieNames[recommendation['movieID']], recommendation['prediction'])

    spark.stop()
