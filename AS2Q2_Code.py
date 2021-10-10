import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql.types import StringType
from pyspark.ml.feature import OneHotEncoder, VectorAssembler, StringIndexer
from pyspark.ml.regression import GeneralizedLinearRegression
from pyspark.ml.evaluation import RegressionEvaluator
from pyspark.ml import Pipeline
from pyspark.ml.regression import LinearRegression
import time
from pyspark.ml.classification import DecisionTreeClassifier




spark = SparkSession.builder \
    .master("local[4]") \
    .appName("COM6012 Assignment 2 Q1") \
    .config("spark.driver.memory", "10g") \
    .config("spark.local.dir", "/fastdata/acp20eww") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")


# Loading the dataset

print("")
print("")
print("")
print("")

missing_values = ["?" or " "]
df = spark.read.csv('../Data/train_set.csv', inferSchema="true", header="true", nanValue=missing_values, nullValue=missing_values, encoding='UTF-8').cache()

#df = spark.read.csv('../Data/train_set.csv', inferSchema="true", header="true", encoding='UTF-8').cache()
df = df.replace('?', None).cache()

# Getting the count of null values for each column in the dataset
#df2.select([F.count(F.when(F.isnan(c) | F.col(c).isNull(), c)).alias(c) for c in df2.columns]).show()

# Getting the total count i.e. the number of occurences in the dataset
feature_modes = {i: df.groupby(i).count().orderBy("count", ascending=False).first()[0] for i in df.columns}

# Shows that the modal value for Cat7 is a missing value so this column will be removed
# & for the other features the missing values will be replaced the modal value
df2 = df.drop("Cat7").cache()

del feature_modes['Cat7']

# Now need to replace the missing values by the mode
df3 = df2.na.fill(feature_modes).cache()
# Converting the categorical variables to one hot encoding

# First need to convert the cateogrical variables using string indexer



cat_cols = ['Blind_Make', 'Blind_Model', 'Blind_Submodel', 'Cat1', 'Cat2', 'Cat3', 'Cat4', 'Cat5',\
 'Cat6', 'Cat8', 'Cat9', 'Cat10', 'Cat11', 'Cat12','NVCat', 'OrdCat']
stage_string = [StringIndexer(inputCol=c, outputCol=c+"_index") for c in cat_cols]
stage_ohe = [OneHotEncoder(inputCol=c+"_index", outputCol=c+"ohe") for c in cat_cols]

stages = stage_string + stage_ohe
pipeline = Pipeline(stages=stages)
df4 = pipeline.fit(df3).transform(df3)

#df4.printSchema()
num_cols = [ 'Row_ID', 'Household_ID', 'Vehicle', 'Calendar_Year', 'Model_Year', 'Var1', 'Var2', 'Var3', 'Var4', 'Var5', 'Var6', 'Var7', 'Var8', 'NVVar1', 'NVVar2', 'NVVar3', 'NVVar4']
features_cols = [c + "ohe" for c in cat_cols] + num_cols                    
assembler = VectorAssembler(inputCols = features_cols, outputCol = 'features')

# Accouting for the imbalanced dataset 

# First need to partition into the training/test sets
(Q2_train, Q2_test) = df4.randomSplit([0.7, 0.3], 21)

Q2_zeros = Q2_train.filter(Q2_train["Claim_Amount"]==0).cache()
total_claims = Q2_train.count()


Q2_non_zeros = Q2_train.filter(Q2_train["Claim_Amount"]>0).cache()

Q2_sample_frac = Q2_non_zeros.count() / Q2_zeros.count()
Q2_zeros = Q2_zeros.sample(fraction=Q2_sample_frac, seed=3)
Q2_train_sampled = Q2_zeros.union(Q2_non_zeros).cache()
Q2_train_sampled.write.mode("overwrite").parquet('../Data/Q2_training.parquet')
Q2_test.write.mode("overwrite").parquet('../Data/Q2_test.parquet')
train = spark.read.parquet('../Data/Q2_training.parquet')
test = spark.read.parquet('../Data/Q2_test.parquet')

# Linear regression

LR = LinearRegression(featuresCol='features', labelCol='Claim_Amount', maxIter=10, regParam=0.01,
                          elasticNetParam=1)
# Getting the training time
LR_start = time.time()
stages2 =[assembler, LR]
pipeline = Pipeline(stages=stages2)
LR_model = pipeline.fit(train)
LR_end = time.time()
LR_time = LR_end - LR_start
print("LR training time", LR_time, "seconds") 
LR_preds = LR_model.transform(test)

# Evaluate using mse and mae

mse_eval = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="prediction", metricName="mse")
mae_eval = RegressionEvaluator(labelCol="Claim_Amount", predictionCol="prediction", metricName="mae")

LR_mse = mse_eval.evaluate(LR_preds)
print("Linear Regression mean absolute error = %g " % LR_mse)

LR_mae = mae_eval.evaluate(LR_preds)
print("Linear Regression mean squared error = %g " % LR_mae)

print('='*50)

# Combination of two models
# Starting with a binary classifier


# First model need to form a new column where if claim amount is greater than zero == 1 else 0

binary_tr = train.withColumn("binary", F.when(F.col("Claim_Amount") > 0, 1).otherwise(0)).cache()
binary_tst = test.withColumn("binary", F.when(F.col("Claim_Amount") > 0, 1).otherwise(0)).cache()

print(binary_tr.head())
print('='*50)

DT = DecisionTreeClassifier(labelCol="binary", featuresCol="features", maxDepth=10, impurity='entropy')
GLM_start = time.time()
DT_stages = [assembler, DT]
DT_pipeline = Pipeline(stages=DT_stages)
DT_model = DT_pipeline.fit(binary_tr)
#DT_end = time.time()
#DT_time = DT_end - DT_start
#print("DT training time", DT_time, "seconds") 
DT_preds = DT_model.transform(binary_tst)
#print(DT_preds.show(5))


# Second model if Claim Amount > 0

fin_tr = train[(train['Claim_Amount'] > 0)].cache()
fin_tst = test[(test['Claim_Amount'] > 0)].cache()


print(fin_tr.head())
print('='*50)

print(fin_tr.count())
GLM = GeneralizedLinearRegression(featuresCol="features", labelCol="Claim_Amount", maxIter=10, regParam=0.01, family='gamma')


feature_col = [c + "_index" for c in cat_cols] + num_cols                    
GLM_assembler = VectorAssembler(inputCols = feature_col, outputCol = 'features')

GLM_stages = [GLM_assembler, GLM]
GLM_pipeline = Pipeline(stages=GLM_stages)
GLM_model = GLM_pipeline.fit(fin_tr)
GLM_end = time.time()
GLM_time = GLM_end - GLM_start
print("GLM training time", GLM_time, "seconds") 
GLM_preds = GLM_model.transform(fin_tst)



GLM_mse = mse_eval.evaluate(GLM_preds)
print("GLM mean absolute error = %g " % GLM_mse)

GLM_mae = mae_eval.evaluate(GLM_preds)
print("GLM mean squared error = %g " % GLM_mae)


















