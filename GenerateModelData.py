# # Model Data Generator
# ## Summary
# This script generates model churn data to be fed into a classification algorithm. It pulls customer
# information from a table in Kudu, generates a churn score based on a specified set of churn factors,
# and writes the resulting model data to an S3 bucket. 

# ## Initialization
# ### Spark Library Imports
from pyspark.sql import SparkSession
from pyspark.sql import SQLContext
from pyspark.sql import functions as F
from math import log

# ### Configuration Settings
kuduMaster = 'cdsw-demo-4.vpc.cloudera.com'
churnThreshold = 0.5
trainingDataSizeM = 1
churnFactors = {
  'state':0.08,
  'age':0.07,
  'sex':0.07,
  'income':0.19,
  'hh_members':0.02,
  'services':0.15,
  'monthly_spend':0.26,
  'days_customer':0.28,
  'change_monthly_data':0.35,
  'change_monthly_watch':0.35,
}

# ### Create a Spark Session
spark = SparkSession.builder.appName("Generate Model Data").getOrCreate()
sc = spark.sparkContext
sqc = SQLContext(sc)

# ## Read in Customer Information from Kudu
customers = sqc.read.format('org.apache.kudu.spark.kudu')\
    .option('kudu.master',kuduMaster)\
    .option('kudu.table','impala::network.customers').load()\
    .withColumn('cust_sinceDT', F.to_date(F.from_unixtime(F.col('cust_since'),'yyyy-MM-dd')))\
    .withColumn('days_customer', F.datediff(F.lit('2018-01-01'),F.col('cust_sinceDT')))
    
customers.show(10)

# ## Build a Churn Likelihood Attribute
# ### For each record based on the weightings (churnFactors) for each customer attribute, create an artificial variable indicating if the customer has churned
churnData = customers.withColumn('churn_likelihood',
                                  (F.col('state').rlike('[A-M].*')).cast('integer')*churnFactors['state']+
                                  (F.col('age')/70)*churnFactors['age']+
                                  (F.col('sex')=='m').cast('integer')*churnFactors['sex']+
                                  (F.col('income')/300000)*churnFactors['income']+
                                  (F.col('hh_members')/5)*churnFactors['hh_members']+
                                  (F.col('services')/5)*churnFactors['services']+
                                  (F.col('monthly_spend')/300)*churnFactors['monthly_spend']+
                                  (F.col('days_customer')/3650)*churnFactors['days_customer']+
                                  F.abs(F.col('last_month_data')-F.col('avg_monthly_data'))/F.col('avg_monthly_data')*churnFactors['change_monthly_data']+
                                  F.abs(F.col('last_month_watch')-F.col('avg_monthly_watch'))/F.col('avg_monthly_watch')*churnFactors['change_monthly_watch'])\
  .withColumn('churned', (F.col('churn_likelihood')>churnThreshold).cast('integer'))\
  .drop('cust_sinceDT','cust_since')
  
churnData.show()
churnData.printSchema()

# ### Duplicate data (optional: just to grow the dataset size)
duplicatedData = churnData

for i in range(1,trainingDataSizeM):
  duplicatedData = duplicatedData.union(churnData.withColumn('id',churnData.id+i*1000000))

duplicatedData.count()
duplicatedData.select(F.min('id'),F.max('id')).show()
churnData.filter('churn_likelihood>0.5').count()

# ## Write Data to Kudu
duplicatedData.drop('churn_likelihood').write.format('org.apache.kudu.spark.kudu')\
  .option("kudu.master", kuduMaster)\
  .option("kudu.table", "impala::network.model_data")\
  .mode("append")\
  .save()