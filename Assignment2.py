# coding: utf-8
#part 1 demo for 3 components-pca +kmeans
from pyspark.sql.functions import col
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pyspark.conf import SparkConf
from pyspark.sql import SparkSession 
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.feature import VectorAssembler, StandardScaler, PCA
from pyspark.sql.types import DoubleType
from sklearn import decomposition
from pyspark.sql.functions import *
import seaborn as sns

columns=["d1","d2","d3","d4","d5","d6"]
df_hd=sc.textFile("homework2.dat").flatMap (lambda x: [x.split(",")]).toDF(columns)
df_hd=df_hd.select(*(col(c).cast("double").alias(c) for c in df_hd.columns))
assembler = VectorAssembler( inputCols = columns, outputCol = 'features')
df_hd= assembler.transform(df_hd).select('features')
scaler = StandardScaler(inputCol ='features',outputCol = 'scaledFeatures',withMean = True,withStd = True).fit(df_hd)
df_hd_scaled = scaler.transform(df_hd)

n_components = 6
pca = PCA( k = n_components, inputCol = 'scaledFeatures', outputCol = 'pcaFeatures' ). fit(df_hd_scaled)
df_hd_pca = pca.transform(df_hd_scaled)
print(pca.explainedVariance)


silhouette_scores=[]
evaluator = ClusteringEvaluator(featuresCol='pcaFeatures', metricName='silhouette', distanceMeasure='squaredEuclidean')
cost = np.zeros(10)

for K in range(2,10):
    KMeans_=KMeans(featuresCol='pcaFeatures', k=K)
    KMeans_fit=KMeans_.fit(df_hd_pca)
    KMeans_transform=KMeans_fit.transform(df_hd_pca) 
    evaluation_score=evaluator.evaluate(KMeans_transform)
    silhouette_scores.append(evaluation_score)

print(silhouette_scores)

KMeans_=KMeans(featuresCol='pcaFeatures', k=5)
KMeans_Model=KMeans_.fit(df_hd_pca)


df_hd_kmeans=KMeans_Model.transform(df_hd_pca)
df_raw = df_hd_kmeans.rdd.map(lambda row: (row.pcaFeatures)).collect()

X=[x[0] for x in df_raw]
Y=[x[1] for x in df_raw]
Z=[x[2] for x in df_raw]
W=df_hd_kmeans.agg(collect_list(col("prediction"))).collect()[0][0]

fig = plt.figure(figsize=(25,25))
ax = fig.add_subplot(111, projection='3d')
ax.scatter(X, Y, Z, c=W, cmap='viridis', edgecolor='k', s=40, alpha=0.5)
X_c=[x[0] for x in  KMeans_Model.clusterCenters()]
Y_c=[x[1] for x in  KMeans_Model.clusterCenters()]
Z_c=[x[2] for x in  KMeans_Model.clusterCenters()]

ax.scatter(X_c, Y_c, Z_c, s=300, c='r', marker='*', label= 'Centroid')
for x, y, z in zip(X_c, Y_c, Z_c):
    text = str(x)+ ', ' + str(y)+ ', ' + str(z)
    ax.text(x, y, z, text, zdir=(1, 1, 1))
plt.show()



#the other way, PCA for each kmeans clusters first, demo for 4000-point cluster, adjust the cluster== from 0 to 4. Noted 3 and 4 are same cluster. 
df=spark.read.option("header",True).csv("df_3d_f.csv")
df_4000=df[df["cluster"]==0]
columns=["0","1","2","3","4","5"]
df_4000=df_4000.select(*(col(c).cast("double").alias(c) for c in df_4000.columns))
assembler = VectorAssembler( inputCols = columns, outputCol = 'features')
df_4000= assembler.transform(df_4000).select('features')
scaler = StandardScaler(inputCol ='features',outputCol = 'scaledFeatures',withMean = True,withStd = True).fit(df_hd)
df_4000_scaled = scaler.transform(df_4000)

n_components = 6
pca = PCA( k = n_components, inputCol = 'scaledFeatures', outputCol = 'pcaFeatures' ). fit(df_4000_scaled)
df_4000_pca = pca.transform(df_4000_scaled)
print(pca.explainedVariance)
