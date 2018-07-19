from pyspark.sql import SparkSession
from pyspark.sql.types import *
from pyspark.ml import PipelineModel

spark = SparkSession.builder \
      .appName("Customer Churn Prediction") \
      .master("local[*]") \
      .getOrCreate()
  
model = PipelineModel.load("file:///home/cdsw/models/rf-pipeline-model") 

features = ['state','age','sex','income','hh_members','services','monthly_spend',
            'days_customer','avg_monthly_data','change_data','avg_monthly_watch','change_tv']

def predict(args):
  customer=args["feature"].split(",")
  customerCols = [customer[:1]+map(int,customer[1:2])+customer[2:3]+map(int,customer[3:9])\
                  +map(float,customer[9:10])+map(int,customer[10:11])+map(float,customer[11:12])]
  feature = spark.createDataFrame(customerCols, features)
  feature.show()
  feature.printSchema()
  result=model.transform(feature).collect()[0].prediction
  return {"result" : result}