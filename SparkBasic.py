# -*- coding: utf-8 -*-
"""
Created on Sat Sep 26 08:57:24 2020

@author: BrownPlanning
"""

from pyspark.sql import SparkSession
import pyspark.sql.functions as F

spark = SparkSession.builder.appName("how to read csv file").getOrCreate()
print(spark.version)

PATH = "C:\\Users\\Brown Planning\\Documents\\Innova\\Datasets\\"
FILE = "foo.csv"
df = spark.read.csv(PATH + FILE, header=True)
df.show(5)

categ = df.select('OS').rdd.distinct().flatMap(lambda x: x).collect()
categ = df.select('OS').distinct().collect()
exprs = [F.when(F.col('OS') == cat, 1).otherwise(0)
         .alias(str(cat)) for cat in categ]
df = df.select(exprs + df.columns)

