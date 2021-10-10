import matplotlib 
matplotlib.use('Agg')
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib.pyplot as plt
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.sql.window import Window
from pyspark.ml.clustering import KMeans
from pyspark.ml.clustering import KMeansModel
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors




spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Q1 Assignment") \
        .config("spark.local.dir","/fastdata/acp20eww") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 


# Code doesn't work in batch file, only works when submitting the python file


Q2_ratings = spark.read.load('../Data/ml-latest/ratings.csv', format = 'csv', inferSchema = "true", header = "true").cache()
#Q2_ratings.show(20,False)


Q2_sorted = Q2_ratings.orderBy('timestamp', ascending=True).cache()
#Q2_sorted.show(20, False)

#print((Q2_sorted.count(), len(Q2_sorted.columns)))

#  Sorting the dataframe in ascending order of timestamp
windowSpec  = Window.orderBy(Q2_ratings['timestamp'].asc())
# Then getting the percentage ranks to split the data
Q2_ranks = Q2_ratings.withColumn("percent_rank", F.percent_rank().over(windowSpec))
Q2_ranks.show(20, False)

# Splitting the data into the three splits
train1 = Q2_ranks.filter(Q2_ranks["percent_rank"]<0.5).cache()
train1.count()
test1 = Q2_ranks.filter(Q2_ranks["percent_rank"]>=0.5).cache()
test1.count()


train2 = Q2_ranks.filter(Q2_ranks["percent_rank"]<0.65).cache()
train2.count()
test2 = Q2_ranks.filter(Q2_ranks["percent_rank"]>=0.65).cache()
test2.count()

train3 = Q2_ranks.filter(Q2_ranks["percent_rank"]<0.8).cache()
train3.count()
test3 = Q2_ranks.filter(Q2_ranks["percent_rank"]>=0.8).cache()
test3.count()


als1 = ALS(userCol = "userId", itemCol = "movieId", seed = 200206688, coldStartStrategy = "drop")
rmse_evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",predictionCol="prediction")
mse_evaluator = RegressionEvaluator(metricName="mse", labelCol="rating",predictionCol="prediction")
mae_evaluator = RegressionEvaluator(metricName="mae", labelCol="rating",predictionCol="prediction")

#train1.show(20)
#test1.show(20)

model1 = als1.fit(train1)
predictions1 = model1.transform(test1)
model2 = als1.fit(train2)
predictions2 = model2.transform(test2)
model3 = als1.fit(train3)
predictions3 = model3.transform(test3)

metrics = []

def run_model(_train, _test, _als, rmse_evaluator, mse_evaluator, mae_evaluator):
    
    metrics = []
    model = _als.fit(_train)
    predictions = model.transform(_test)
    rmse = rmse_evaluator.evaluate(predictions)
    mse = mse_evaluator.evaluate(predictions)
    mae = mae_evaluator.evaluate(predictions)

    #print("Root-mean-square error = " + str(rmse))
    #print("Mean-square error = " + str(mse))
    #print("Mean absolute error = " + str(mae))
    
    metrics.append(rmse)
    metrics.append(mse)
    metrics.append(mae)

    return metrics

# Carrying out the als and evaluating the performance metrics using three different als settings
output1 = run_model(train1, test1, als1, rmse_evaluator, mse_evaluator, mae_evaluator)
output2 = run_model(train2, test2, als1, rmse_evaluator, mse_evaluator, mae_evaluator)
output3 = run_model(train3, test3, als1, rmse_evaluator, mse_evaluator, mae_evaluator)
als2 = als1.setRank(20)

output4 = run_model(train1, test1, als2, rmse_evaluator, mse_evaluator, mae_evaluator)
output5 = run_model(train2, test2, als2, rmse_evaluator, mse_evaluator, mae_evaluator)
output6 = run_model(train3, test3, als2, rmse_evaluator, mse_evaluator, mae_evaluator)

als3 = als1.setRegParam(0.05)
output7 = run_model(train1, test1, als3, rmse_evaluator, mse_evaluator, mae_evaluator)
output8 = run_model(train2, test2, als3, rmse_evaluator, mse_evaluator, mae_evaluator)
output9 = run_model(train3, test3, als3, rmse_evaluator, mse_evaluator, mae_evaluator)

print(output1, output4, output7)
print(output2, output5, output8)
print(output3, output6, output9)

# Getting the user factors vector from the three als models
def featuresvec(model):
  User_facts = model.userFactors.collect()
  df = spark.createDataFrame(User_facts, ["userid", "features"])
  id_rdd = df.rdd.map(lambda row:row[0])
  features_rdd = df.rdd.map(lambda row:row[1])
  new_df = id_rdd.zip(features_rdd.map(lambda x:Vectors.dense(x))).toDF(schema=['userid','features'])
  #features_df = new_df.drop('id')
  #return features_df
  return new_df
#vecAssembler = VectorAssembler(inputCol="features", outputCol="features")
#df2 = df.drop("id")

dfFeatureVec1 = featuresvec(model1)
dfFeatureVec2 = featuresvec(model2)
dfFeatureVec3 = featuresvec(model3)

kmeans = KMeans(k=20, seed=200206688)

#Using kmeans on the three als settings 
kmeans1 = kmeans.fit(dfFeatureVec1.select('features'))
kpreds1_tr = kmeans1.transform(dfFeatureVec1.select('features'))
#dfFeatureVec1.show(5)
#kpreds1.show(5)
#preds_with_ids1 = dfFeatureVec1.join(kpreds1, dfFeatureVec1.features == kpreds1.features, "leftouter")
preds_with_ids1 = dfFeatureVec1.join(kpreds1_tr, ["features"], "leftouter").cache()
#preds_with_ids1.show(5)
summary1 = kmeans1.summary
Cluster_sizes1 = summary1.clusterSizes
Cluster_sizes1sort = sorted(summary1.clusterSizes, reverse=True)
print(Cluster_sizes1sort[0:3])
#print(Cluster_sizes1)
#print(Cluster_sizes1sort)
largest_cluster1 = Cluster_sizes1.index(max(Cluster_sizes1))
#print(largest_cluster1)
largest_cluster_ids1 =  preds_with_ids1.filter(preds_with_ids1["prediction"]==largest_cluster1).cache()
largest_cluster_ids1.show(5)

kmeans2 = kmeans.fit(dfFeatureVec2.select('features'))
kpreds2_tr = kmeans2.transform(dfFeatureVec2.select('features'))
#dfFeatureVec2.show(5)
#kpreds2.show(5)
#preds_with_ids2 = dfFeatureVec2.join(kpreds2, dfFeatureVec2.features == kpreds2.features, "leftouter")
preds_with_ids2 = dfFeatureVec2.join(kpreds2_tr, ["features"], "leftouter").cache()
#preds_with_ids2.show(5)
summary2 = kmeans2.summary
Cluster_sizes2 = summary2.clusterSizes
Cluster_sizes2sort = sorted(summary2.clusterSizes, reverse=True)
print(Cluster_sizes2sort[0:3])
#print(Cluster_sizes2)
#print(Cluster_sizes2sort)
largest_cluster2 = Cluster_sizes2.index(max(Cluster_sizes2))
#print(largest_cluster2)
largest_cluster_ids2 =  preds_with_ids2.filter(preds_with_ids2["prediction"]==largest_cluster2).cache()
#largest_cluster_ids2.show(5)




kmeans3 = kmeans.fit(dfFeatureVec3.select('features'))
kpreds3 = kmeans3.transform(dfFeatureVec3.select('features'))
#dfFeatureVec3.show(5)
#kpreds3.show(5)
#preds_with_ids3 = dfFeatureVec3.join(kpreds3, dfFeatureVec3.features == kpreds3.features, "leftouter")
preds_with_ids3 = dfFeatureVec3.join(kpreds3, ["features"], "leftouter").cache()
#preds_with_ids3.show(5)
summary3 = kmeans3.summary
Cluster_sizes3 = summary3.clusterSizes
Cluster_sizes3sort = sorted(summary3.clusterSizes, reverse=True)
print(Cluster_sizes3sort[0:3])
#print(Cluster_sizes3)
#print(Cluster_sizes3sort)
largest_cluster3 = Cluster_sizes3.index(max(Cluster_sizes3))
#print(largest_cluster3)
largest_cluster_ids3 =  preds_with_ids3.filter(preds_with_ids3["prediction"]==largest_cluster3).cache()
#largest_cluster_ids3.show(5)


movie_data = spark.read.load('../Data/ml-latest/movies.csv', format = 'csv', inferSchema = "true", header = "true").cache()
#movie_data.show(20, False)

#ratings_above4 = Q2_ratings.filter(Q2_ratings.rating>=4).cache()
#ratings_above4.show(20)

# Filtering the ratings for the train set and test set
# Then selecting only the movieids in the clusters
# Now splitting the movies into the genres and counting them to get the top 5

users_ids1 = largest_cluster_ids1.select("userid").rdd.flatMap(lambda x: x).collect()
#user_ids.show(50)
tr_ratings1 = train1.filter(train1.rating>=4).cache()
tr_filtered_ids1 = tr_ratings1.filter(F.col('userId').isin(users_ids1)).cache()
#filtered_ids1.show(20)
tr_filtered_movies1 = movie_data.join(tr_filtered_ids1, movie_data.movieId == tr_filtered_ids1.movieId, "leftanti")
#filtered_movies.show(20, False)
tr_genre_split1 = tr_filtered_movies1.select("title",F.split("genres", "\|").alias("genres"), F.posexplode(F.split("genres","\|")).alias("pos", "genre")).drop("genre").select("title",F.expr("genres[pos]").alias("genre"))
#genre_split1.show(10, False)
tr_genre_counts1 = tr_genre_split1.groupBy("genre").count().sort(F.col("count").desc())
tr_top5_genre1 = tr_genre_counts1.take(5)
print("The top 5 movie genres for train split 1",tr_top5_genre1)

#user_ids.show(50)
tst_ratings1 = test1.filter(test1.rating>=4).cache()
tst_filtered_ids1 = tst_ratings1.filter(F.col('userId').isin(users_ids1)).cache()
#filtered_ids1.show(20)
tst_filtered_movies1 = movie_data.join(tst_filtered_ids1, movie_data.movieId == tst_filtered_ids1.movieId, "leftanti")
#filtered_movies.show(20, False)
tst_genre_split1 = tst_filtered_movies1.select("title",F.split("genres", "\|").alias("genres"), F.posexplode(F.split("genres","\|")).alias("pos", "genre")).drop("genre").select("title",F.expr("genres[pos]").alias("genre"))
#genre_split1.show(10, False)
tst_genre_counts1 = tst_genre_split1.groupBy("genre").count().sort(F.col("count").desc())
tst_top5_genre1 = tst_genre_counts1.take(5)
print("The top 5 movie genres for test split 1",tst_top5_genre1)


users_ids2 = largest_cluster_ids2.select("userid").rdd.flatMap(lambda x: x).collect()
#user_ids.show(50)
tr_ratings2 = train2.filter(train2.rating>=4).cache()
tr_filtered_ids2 = tr_ratings2.filter(F.col('userId').isin(users_ids2)).cache()
#filtered_ids1.show(20)
tr_filtered_movies2 = movie_data.join(tr_filtered_ids2, movie_data.movieId == tr_filtered_ids2.movieId, "leftanti")
#filtered_movies.show(20, False)
tr_genre_split2 = tr_filtered_movies2.select("title",F.split("genres", "\|").alias("genres"), F.posexplode(F.split("genres","\|")).alias("pos", "genre")).drop("genre").select("title",F.expr("genres[pos]").alias("genre"))
#genre_split2.show(10, False)
tr_genre_counts2 = tr_genre_split2.groupBy("genre").count().sort(F.col("count").desc())
tr_top5_genre2 = tr_genre_counts2.take(5)
print("The top 5 movie genres for train split 2",tr_top5_genre2)

#user_ids.show(50)
tst_ratings2 = test2.filter(test2.rating>=4).cache()
tst_filtered_ids2 = tst_ratings2.filter(F.col('userId').isin(users_ids2)).cache()
#filtered_ids1.show(20)
tst_filtered_movies2 = movie_data.join(tst_filtered_ids2, movie_data.movieId == tst_filtered_ids2.movieId, "leftanti")
#filtered_movies.show(20, False)
tst_genre_split2 = tst_filtered_movies2.select("title",F.split("genres", "\|").alias("genres"), F.posexplode(F.split("genres","\|")).alias("pos", "genre")).drop("genre").select("title",F.expr("genres[pos]").alias("genre"))
#genre_split2.show(10, False)
tst_genre_counts2 = tst_genre_split2.groupBy("genre").count().sort(F.col("count").desc())
tst_top5_genre2 = tst_genre_counts2.take(5)
print("The top 5 movie genres for test split 2", tst_top5_genre2)




users_ids3 = largest_cluster_ids3.select("userid").rdd.flatMap(lambda x: x).collect()
#user_ids.show(50)
tr_ratings3 = train3.filter(train3.rating>=4).cache()
tr_filtered_ids3 = tr_ratings3.filter(F.col('userId').isin(users_ids3)).cache()
#filtered_ids1.show(20)
tr_filtered_movies3 = movie_data.join(tr_filtered_ids3, movie_data.movieId == tr_filtered_ids3.movieId, "leftanti")
#filtered_movies.show(20, False)
tr_genre_split3 = tr_filtered_movies3.select("title",F.split("genres", "\|").alias("genres"), F.posexplode(F.split("genres","\|")).alias("pos", "genre")).drop("genre").select("title",F.expr("genres[pos]").alias("genre"))
#genre_split3.show(10, False)
tr_genre_counts3 = tr_genre_split3.groupBy("genre").count().sort(F.col("count").desc())
tr_top5_genre3 = tr_genre_counts3.take(5)
print("The top 5 movie genres for train split 2",tr_top5_genre3)

#user_ids.show(50)
tst_ratings3 = test3.filter(test3.rating>=4).cache()
tst_filtered_ids3 = tst_ratings3.filter(F.col('userId').isin(users_ids3)).cache()
#filtered_ids1.show(20)
tst_filtered_movies3 = movie_data.join(tst_filtered_ids3, movie_data.movieId == tst_filtered_ids3.movieId, "leftanti")
#filtered_movies.show(20, False)
tst_genre_split3 = tst_filtered_movies3.select("title",F.split("genres", "\|").alias("genres"), F.posexplode(F.split("genres","\|")).alias("pos", "genre")).drop("genre").select("title",F.expr("genres[pos]").alias("genre"))
#genre_split3.show(10, False)
tst_genre_counts3 = tst_genre_split3.groupBy("genre").count().sort(F.col("count").desc())
tst_top5_genre3 = tst_genre_counts3.take(5)
print("The top 5 movie genres for test split 3",tst_top5_genre3)




spark.stop()
