df_#pre-processing
columns=['ascension','declination','frequency','time']

from pyspark.sql.functions import col
from pyspark.sql.functions import *

def find_length(A):
        return len(A)
find_length_UDF=udf(lambda x: find_length(x))


df_pulsar_1=sc.textFile("pulsar.dat").flatMap(lambda x: [x.split()]).map(lambda x: (x[0],x[1],x[3],x[2])).toDF(columns)

df_pulsar_2=sc.textFile("pulsar.dat").flatMap(lambda x: [x.split()]).map(lambda x: (x[0],x[1],x[3],x[2])).toDF(columns)


df_pulsar_1=df_pulsar_1.select(*(col(c).cast("float").alias(c) for c in df_pulsar_1.columns))

df_pulsar_2=df_pulsar_2.select(*(col(c).cast("float").alias(c) for c in df_pulsar_2.columns))

df_pulsar_1=df_pulsar_1.select (*(col(x).alias(x+'1') for x in df_pulsar_1.columns))
df_pulsar_2=df_pulsar_2.select (*(col(x).alias(x+'2') for x in df_pulsar_2.columns))

#round to 0.1, we also treat freq with 0.1 for convenience (as Prof’s announcement)
for i in df_pulsar_1.columns:
         df_pulsar_1=df_pulsar_1.withColumn(i, round(i,1))

for i in df_pulsar_2.columns:
         df_pulsar_2=df_pulsar_2.withColumn(i, round(i,1))

#confirm signals per location (within error)
df_join =df_pulsar_1.join(df_pulsar_2,
(abs(df_pulsar_1.ascension1- df_pulsar_2.ascension2)<0.2) & (abs(df_pulsar_1.declination1- df_pulsar_2.declination2)<0.2) & (abs (df_pulsar_1.frequency1- df_pulsar_2.frequency2)<=1)).select(df_pulsar_1.ascension1.alias("ascension"), df_pulsar_1.declination1.alias("declination"), df_pulsar_1.frequency1. alias("frequency"), df_pulsar_2.time2.alias("time")).groupBy("ascension","declination","frequency").agg(sort_array(array_distinct(collect_list("time"))).alias("times")). dropDuplicates(["times"]).withColumn('maximum_length', find_length_UDF('times')).orderBy("maximum_length",ascending=False)

df1=df_join.limit(10)
df2=df_join.limit(10)
df1=df1.select (*(col(x).alias(x+'1') for x in df1.columns))
df2=df2.select (*(col(x).alias(x+'2') for x in df2.columns))

#searching nearby rough (~) location
df=df1.join(df2, (abs(df1.ascension1-df2.ascension2)<=0.2) & (abs(df1.declination1- df2.declination2)<=0.2)& (abs(df1.frequency1-df2.frequency2)<=1)).select(df1.ascension1.alias("ascension_rough"), df1.declination1.alias("declination_rough"), df1.frequency1. alias("frequency_rough"),df2.times2.alias("times")).groupBy("ascension_rough","declination_rough","frequency_rough").agg(sort_array(array_distinct(flatten(collect_list("times")))).alias("times")). dropDuplicates(["times"]).withColumn ('maximum_length',find_length_UDF('times')).orderBy("maximum_length")

#function to find maximum roughly (+-1) equally-spaced consecutive sequence, actually is not needed for our situation as we only have few timestamps per location per freq, which can compare manually. 
def find_sequence (A):
    Aset = set(A)
    if len(A)==1:   
        lmax =1
    else:
        lmax = 2
        for j, b in enumerate(A):
            for i in range(j):
                a = A[i]
                step = b - a
                if b + step-1 in Aset and a - step+1 not in Aset and step-1!=0:
                    c = b + step-1
                    count = 3
                    while (c + step in Aset)or (c+step+1 in Aset) or (c+step-1 in Aset and step-1!=0):
                        if c+step-1 in Aset and step-1!=0:
                            c += step-1
                        elif c+step in Aset:
                            c+= step
                        elif c+step+1 in Aset:
                            c+= step+1
                        count=count+1
                    if count > lmax:
                        lmax = count      
                elif b + step in Aset and a - step not in Aset:
                    c = b + step
                    count = 3
                    while (c + step in Aset)or (c+step+1 in Aset) or (c+step-1 in Aset and step-1!=0):
                        if c+step-1 in Aset and step-1!=0:
                            c+= step-1
                        elif c+step in Aset:
                            c+= step
                        elif c+step+1 in Aset:
                            c+= step+1
                        count=count+1
                    if count > lmax:
                        lmax = count
                elif b + step+1 in Aset and a - step-1 not in Aset:
                    c = b + step+1
                    count = 3         
                    while (c + step in Aset)or (c+step+1 in Aset) or (c+step-1 in Aset and step-1!=0):
                        if c+step-1 in Aset and step-1!=0:
                            c += step-1
                        elif c+step in Aset:
                            c+= step
                        elif c+step+1 in Aset and step+1!=0:
                            c+= step+1
                        count=count+1
                    if count > lmax:
                        lmax = count
    return lmax

from pyspark.sql.functions import expr
find_sequence_UDF=udf(lambda x: find_sequence(x))

#final result return longest one.
df=df.withColumn('times_int', expr('transform(times,x-> int(round(x)))')). withColumn('max_consec_subseq', find_sequence_UDF('times_int')). orderBy('max_consec_subseq') 
df=df.select(df.ascension_rough,df.declination_rough,df.frequency_rough,df.times,df.maximum_length, df.max_consec_subseq) .show(1,truncate=False)

