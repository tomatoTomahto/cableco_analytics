# # Customer Churn Analysis
# ## Summary
# This script analyzes customer data and builds a machine learning model to predict customer churn
# based on various attributes, including:
# * Demographics - age, sex, household size, income, location
# * Value - monthly spend, subscribers in the household, duration of contract
# * Usage - number of services, monthly data usage, monthly tv usage

# ## Initialization
#!pip install vincent

# ### Spark Library Imports
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from math import log
import seaborn as sb, pandas as pd, sys, cdsw

# ### Configuration Settings
kuduMaster = '10.0.0.25'
pd.options.display.html.table_schema = True
sample_size = 0.01
maxIter=10
regParam=0.3
num_trees = 10
experimentType = 'none'

# ### Read in parameters
args = len(sys.argv)
if args == 2:
  num_trees = int(sys.argv[1])
  experimentType = 'rf'
if args == 3:
  maxIter = int(sys.argv[1])
  regParam =  float(sys.argv[2])
  experimentType = 'lr'

!rm -rf models/*

# ### Create a Spark Session
spark = SparkSession.builder.appName("Analyze Customer Data").getOrCreate()
sc = spark.sparkContext
sqc = SQLContext(sc)

# ## Read in Customer Information from Kudu
customersAll = sqc.read.format('org.apache.kudu.spark.kudu')\
    .option('kudu.master',kuduMaster)\
    .option('kudu.table','impala::network.model_data').load()\
    .withColumn('churn',F.when(F.col('churned')==1,'yes').otherwise('no'))\
    .withColumn('change_data',(F.col('last_month_data')-F.col('avg_monthly_data'))/F.col('avg_monthly_data'))\
    .withColumn('change_tv',(F.col('last_month_watch')-F.col('avg_monthly_watch'))/F.col('avg_monthly_watch'))

customers = customersAll.sample(False, sample_size)
  
customers.toPandas().head()

# ## Data Analysis
# ### Correlations Between Various Customer Attributes
numericFeatures = customers.select('age','income','avg_monthly_data','sex','monthly_spend')
sb.pairplot(numericFeatures.toPandas(), hue='sex', palette="husl", 
            diag_kind="kde", diag_kws=dict(shade=True))

# ### Monthly Spend vs. Monthly Data
spendVsUsage = customers.select('monthly_spend','avg_monthly_data')
sb.jointplot(x="monthly_spend", y="avg_monthly_data", data=spendVsUsage.toPandas(),
             kind="hex", color="#4CB391")

# ### Monthly Spend vs. Income Bracket
spendByIncome = customers.withColumn('income_bracket', F.when(customers.income<30000,'0-30K')\
                                        .when(customers.income.between(30000,60000),'30K-60K')\
                                        .when(customers.income.between(60000,90000),'60K-90K')\
                                        .when(customers.income.between(90000,120000),'90K-120K')\
                                        .otherwise('>120000'))\
  .select('income_bracket','sex','monthly_spend').orderBy('income_bracket')
  
sb.boxplot(x="income_bracket", y="monthly_spend", hue="sex", linewidth=2.5, data=spendByIncome.toPandas())

# ## Machine Learning: Customer Spend Prediction
# ### Feature Engineering - convert the data into a dataset that can be fed into a linear regression model (vector of numeric features)
# * Convert sex and state to integers - String Indexer
# * Convert categorical attributes to encoded vectors - One Hot Encoder
# * Combine all features into a vector - Vector Assembler
# * Normalize the features to a common scale
# * Feed data into a Linear regression model to predict average spend
from pyspark.ml.regression import LinearRegression
from pyspark.ml import Pipeline
from pyspark.ml.feature import OneHotEncoder, StringIndexer, VectorAssembler, StandardScaler, VectorIndexer

features = ['age','income','avg_monthly_data','sex_vector','state_vector','hh_members','services']
label = 'monthly_spend'

pipelineStages = []

for feature in ['sex','state']:
  pipelineStages.append(StringIndexer(inputCol=feature, outputCol="%s_index"%feature))
  pipelineStages.append(OneHotEncoder(inputCol="%s_index"%feature, outputCol="%s_vector"%feature))
                              
pipelineStages.append(VectorAssembler(inputCols=features,outputCol="features"))
pipelineStages.append(StandardScaler(inputCol="features", outputCol="scaledFeatures"))
pipelineStages.append(LinearRegression(featuresCol='scaledFeatures', labelCol=label,
                                       maxIter=maxIter, regParam=regParam))

# ### ML Pipeline - build a pipeline of feature transformations and model training
# * Split data into train and test datasets
# * Build a pipeline
# * Fit the pipeline with training data
(trainingData, testData) = customersAll.randomSplit([0.7, 0.3])

lrPipeline = Pipeline(stages=pipelineStages)

lrPipelineModel = lrPipeline.fit(trainingData)

# ### Evaluate Model
lrPipelineModel.write().overwrite().save("lr-pipeline-model")
!hdfs dfs -get lr-pipeline-model models/

lrModel = lrPipelineModel.stages[6]
trainingSummary = lrModel.summary
print("RMSE: %f" % trainingSummary.rootMeanSquaredError)
print("r2: %f" % trainingSummary.r2)
if experimentType == 'lr':
  cdsw.track_metric('RMSE',trainingSummary.rootMeanSquaredError)
  cdsw.track_metric('r-squared',trainingSummary.r2)
  cdsw.track_file("/home/cdsw/models/lr-pipeline-model")
  
# ### Visualize Predictions and Residuals
predictions = lrPipelineModel.transform(testData)\
  .withColumn('residuals',(F.col('prediction')-F.col('monthly_spend'))/F.col('prediction'))\
  .select(F.col('prediction').alias('predictedSpend'),'residuals')\
  .sample(False,sample_size)
  
sb.jointplot(x="predictedSpend", y="residuals", data=predictions.toPandas(), kind="reg")

# ## Machine Learning: Customer Churn Prediction
numericFeatures = customers.select('avg_monthly_data','monthly_spend','churn',
                                   'days_customer','change_data','income')
numericFeatures.printSchema()
sb.pairplot(numericFeatures.toPandas(),hue='churn',palette="husl",diag_kind="kde", diag_kws=dict(shade=True))

# ### Feature Engineering - convert the data into a dataset that can be fed into a random forest classification model (vector of numeric features)
# * Convert sex and state to integers - String Indexer
# * Combine all features into a vector - Vector Assembler
# * Feed data into a random forest classification model to predict average spend
from pyspark.ml.classification import RandomForestClassifier
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

features = ['state_index','age','sex_index','income','hh_members','services','monthly_spend',
            'days_customer','avg_monthly_data','change_data','avg_monthly_watch','change_tv']

label = 'churned'

pipelineStages = []
for feature in ['sex','state']:
  pipelineStages.append(StringIndexer(inputCol=feature, outputCol="%s_index"%feature))

pipelineStages.append(VectorAssembler(inputCols=features,outputCol="features"))
#pipelineStages.append(VectorIndexer(inputCol="features", outputCol="indexedFeatures", maxCategories=60))
pipelineStages.append(RandomForestClassifier(labelCol=label, featuresCol="features", numTrees=10, maxBins=64))

# ### ML Pipeline - build a pipeline of feature transformations and model training
# * Split data into train and test datasets
# * Build a pipeline
# * Fit the pipeline with training data
(trainingData, testData) = customersAll.randomSplit([0.7, 0.3])

rfPipeline = Pipeline(stages=pipelineStages)

rfPipelineModel = rfPipeline.fit(trainingData)

# ### Evaluate Model and Feature Importances
rfPipelineModel.write().overwrite().save("rf-pipeline-model")
!hdfs dfs -get rf-pipeline-model models/

predictions = rfPipelineModel.transform(testData)
predictions.select("prediction", label, "features").show(5)

evaluator = MulticlassClassificationEvaluator(
    labelCol=label, predictionCol="prediction", metricName="accuracy")
accuracy = evaluator.evaluate(predictions)
print("Test Error = %g" % (1.0 - accuracy))
if experimentType == 'rf':
  cdsw.track_metric('Accuracy',accuracy)
  cdsw.track_file("/home/cdsw/models/rf-pipeline-model")

featureImportances = pd.DataFrame({'feature': features,
                                   'importance': rfPipelineModel.stages[3].featureImportances.toArray()})\
  .sort_values(by=['importance'])

sb.barplot(y="feature", x="importance", data=featureImportances)