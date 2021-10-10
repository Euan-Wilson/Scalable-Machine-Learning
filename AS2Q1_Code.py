import pyspark
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import numpy as np
from pyspark.sql.types import StringType
from pyspark.sql.functions import col
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import MulticlassClassificationEvaluator
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.classification import GBTClassifier
from pyspark.ml.classification import MultilayerPerceptronClassifier
import re
import time



spark = SparkSession.builder \
    .master("local[4]") \
    .appName("COM6012 Assignment 2 Q1") \
    .config("spark.driver.memory", "10g") \
    .config("spark.local.dir", "/fastdata/acp20eww") \
    .getOrCreate()

sc = spark.sparkContext
sc.setLogLevel("ERROR")

import warnings
warnings.filterwarnings("ignore")

print("")
print("")
print("")
print('='*50)
Q1_rawdata = spark.read.csv('../Data/HIGGS.csv.gz')


features = ['label','lepton_pT','lepton_eta','lepton_phi', 'missing_energy_magnitude','missing_energy_phi', 'jet_1_pt', 'jet_1_eta', 'jet_1_phi', 'jet_1_b-tag', 'jet_2_pt', 'jet_2_eta', 'jet_2_phi', 'jet_2_btag', 'jet_3_pt', 'jet_3_eta','jet_3_phi', 'jet_3_btag', 'jet_4_pt', 'jet_4_eta', 'jet_4_phi', 'jet_4_btag', 'mjj', 'mjjj', 'mlv', 'mjlv', 'mbb', 'mwbb', 'mwwbb']
ncolumns = len(Q1_rawdata.columns)
schemaNames = Q1_rawdata.schema.names
for i in range(ncolumns):
    Q1_rawdata = Q1_rawdata.withColumnRenamed(schemaNames[i], features[i])
    
StringColumns = [x.name for x in Q1_rawdata.schema.fields if x.dataType == StringType()]
for c in StringColumns:
    Q1_rawdata = Q1_rawdata.withColumn(c, col(c).cast("double"))
    
   
Q1_sampled = Q1_rawdata.sample(False, 0.01, 42).cache()
(Q1subset_train, Q1subset_test) = Q1_sampled.randomSplit([0.7, 0.3], 21)

Q1subset_train.write.mode("overwrite").parquet('../Data/Q1subset_training.parquet')
Q1subset_test.write.mode("overwrite").parquet('../Data/Q1subset_test.parquet')
subset_train = spark.read.parquet('../Data/Q1subset_training.parquet')
subset_test = spark.read.parquet('../Data/Q1subset_test.parquet')

vecAssembler = VectorAssembler(inputCols = features[1:], outputCol = 'features') 
RF = RandomForestClassifier(labelCol="label", featuresCol="features", maxDepth=10, impurity='entropy')

RF_stages = [vecAssembler, RF]
RF_pipeline = Pipeline(stages=RF_stages)

    
RF_paramGrid = ParamGridBuilder() \
    .addGrid(RF.maxDepth, [1, 5, 10]) \
    .addGrid(RF.maxBins, [10, 20, 50]) \
    .addGrid(RF.numTrees, [1, 5, 10]) \
    .addGrid(RF.featureSubsetStrategy, ['all','sqrt', 'log2']) \
    .addGrid(RF.subsamplingRate, [0.1, 0.5, 0.9]) \
    .build()
    
RF_crossval = CrossValidator(estimator=RF_pipeline,
                          estimatorParamMaps=RF_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)

RF_cvModel = RF_crossval.fit(subset_train)
RF_predictions = RF_cvModel.transform(subset_test)


Acc_evaluator = MulticlassClassificationEvaluator\
      (labelCol="label", predictionCol="prediction", metricName="accuracy")
      
        
Area_evaluator = BinaryClassificationEvaluator\
      (labelCol="label", rawPredictionCol="prediction", metricName="areaUnderROC")
      
      
      
RF_accuracy = Acc_evaluator.evaluate(RF_predictions)
print("RF accuracy = %g " % RF_accuracy)
RF_area = Area_evaluator.evaluate(RF_predictions)
print("RF area under the curve = %g " % RF_area)
print('='*50)

# for gradient boosting

GBT = GBTClassifier(maxIter=5, maxDepth=2, labelCol="label", seed=42,
    featuresCol="features", lossType='logistic')


GBT_stages = [vecAssembler, GBT]
GBT_pipeline = Pipeline(stages=GBT_stages)

    
GBT_paramGrid = ParamGridBuilder() \
    .addGrid(GBT.maxDepth, [1, 5, 10]) \
    .addGrid(GBT.maxIter, [10, 20, 30]) \
    .addGrid(GBT.stepSize, [0.1, 0.2, 0.05]) \
    .addGrid(GBT.subsamplingRate, [0.1, 0.5, 0.9]) \
    .build()
    
GBT_crossval = CrossValidator(estimator=GBT_pipeline,
                          estimatorParamMaps=GBT_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)

GBT_cvModel = GBT_crossval.fit(subset_train)
GBT_predictions = GBT_cvModel.transform(subset_test)


GBT_accuracy = Acc_evaluator.evaluate(GBT_predictions)
print("GBT accuracy = %g " % GBT_accuracy)
GBT_area = Area_evaluator.evaluate(GBT_predictions)
print("GBT area under the curve = %g " % GBT_area)
print('='*50)



MLP = MultilayerPerceptronClassifier(maxIter=5, labelCol="label", seed=42, featuresCol="features")
      
MLP_stages = [vecAssembler, MLP]
MLP_pipeline = Pipeline(stages=MLP_stages)

    
MLP_paramGrid = ParamGridBuilder() \
    .addGrid(MLP.tol, [1e-6, 1e-5, 1e-4]) \
    .addGrid(MLP.layers, [[len(subset_train.columns)-1, 20, 10, 2], [len(subset_train.columns)-1, 8, 2], [len(subset_train.columns)-1, 15, 6, 2]]) \
    .addGrid(MLP.stepSize, [0.03, 0.02, 0.05]) \
    .addGrid(MLP.blockSize, [128, 100, 150]) \
    .addGrid(MLP.solver, ['l-bfgs', 'gd']) \
    .addGrid(MLP.maxIter, [5, 10, 20]) \
    .build()
    
MLP_crossval = CrossValidator(estimator=MLP_pipeline,
                          estimatorParamMaps=MLP_paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5)

MLP_cvModel = MLP_crossval.fit(subset_train)


MLP_predictions = MLP_cvModel.transform(subset_test)

MLP_accuracy = Acc_evaluator.evaluate(MLP_predictions)
print("MLP accuracy = %g " % MLP_accuracy)
MLP_area = Area_evaluator.evaluate(MLP_predictions)
print("MLP area under the curve = %g " % MLP_area)
print('='*50)




# Extracting the best parameters for each model to use on the whole dataset

RF_hyper = RF_cvModel.getEstimatorParamMaps()[np.argmax(RF_cvModel.avgMetrics)]
GBT_hyper = GBT_cvModel.getEstimatorParamMaps()[np.argmax(GBT_cvModel.avgMetrics)]
MLP_hyper = MLP_cvModel.getEstimatorParamMaps()[np.argmax(MLP_cvModel.avgMetrics)]

def extract_params(hyperparams):
  hyper_list = []
  for i in range(len(hyperparams.items())):
      hyper_name = re.search("name='(.+?)'", str([x for x in hyperparams.items()][i])).group(1)
      hyper_value = [x for x in hyperparams.items()][i][1]
      hyper_list.append({hyper_name: hyper_value})
      
  return hyper_list
      
RF_params = extract_params(RF_hyper)     
print(RF_params) 
print('='*50)  

GBT_params = extract_params(GBT_hyper)     
print(GBT_params)
print('='*50)     

MLP_params = extract_params(MLP_hyper)     
print(MLP_params)
print('='*50)  



# Splitting to get the train and test set for the entire dataset
# Saving to parquet to use the same splits for all of the models
# Commented this out when testing using 5 and 10 Cores to ensure I used the same splits

(Q1train, Q1test) = Q1_sampled.randomSplit([0.7, 0.3], 21)
Q1train.write.mode("overwrite").parquet('../Data/Q1training.parquet')
Q1test.write.mode("overwrite").parquet('../Data/Q1test.parquet')


train = spark.read.parquet('../Data/Q1training.parquet')
test = spark.read.parquet('../Data/Q1test.parquet')


# Training the models on the training data & getting the training time for each model

# Using the best params

RF_start = time.time()
RF_best = RF_cvModel.bestModel
RF_stage = [vecAssembler, RF_best]
RF_pipe = Pipeline(stages=RF_stage)

RF_Model = RF_pipe.fit(train)
RF_end = time.time()
RF_time = RF_end - RF_start
print("Training time for Random Forests is", RF_time, "seconds")
#RF_predictions = RF_Model.transform(test)


GBT_start = time.time()                    
GBT_best = GBT_cvModel.bestModel
GBT_stage = [vecAssembler, GBT_best]
GBT_pipe = Pipeline(stages=GBT_stage)

GBT_Model = GBT_pipe.fit(train)
GBT_end = time.time()
GBT_time = GBT_end - GBT_start
print("Training time for Gradient Boosting is", GBT_time, "seconds")
#GBT_predictions = GBT_Model.transform(test)





MLP_start = time.time()
MLP_best = MLP_cvModel.bestModel
MLP_stage = [vecAssembler, MLP_best]
MLP_pipe = Pipeline(stages=MLP_stage)

MLP_Model = MLP_pipe.fit(train)
MLP_end = time.time()
MLP_time = MLP_end - MLP_start
print("Training time for Neural network is", MLP_time, "seconds")
#MLP_predictions = MLP_cvModel.transform(test)
print('='*50) 


# Getting the 3 most relevant features for each model

RF_Mod = RF_Model.stages[-1].featureImportances
GBT_Mod = GBT_Model.stages[-1].featureImportances


RF_index = np.argsort(np.abs(RF_Mod))[-3:]
RF_features = [features[x] for x in RF_index]
print("Top 3 features for Random Forests", RF_features)

GBT_index = np.argsort(np.abs(GBT_Mod))[-3:]
GBT_features = [features[x] for x in GBT_index]
print("Top 3 features for Gradient boosting", GBT_features)





