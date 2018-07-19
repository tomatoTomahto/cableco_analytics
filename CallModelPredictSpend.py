from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
      .appName("Customer Spend Prediction") \
      .master("local[*]") \
      .getOrCreate()
  
model = PipelineModel.load("file:///home/cdsw/models/lr-pipeline-model") 

features = ['age','income','avg_monthly_data','sex','state','hh_members','services']

def predict(args):
  customer=args["feature"].split(",")
  feature = spark.createDataFrame([map(int,customer[0:3])+customer[3:5]+map(int,customer[5:7])], features)
  feature.show()
  feature.printSchema()
  result=model.transform(feature).collect()[0].prediction
  return {"result" : result}