from pyspark.sql import SparkSession
import pyspark.sql.functions as F
import matplotlib
import numpy as np 
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns 


spark = SparkSession.builder \
        .master("local[4]") \
        .appName("Q1 Assignment") \
        .config("spark.local.dir","/fastdata/acp20eww") \
        .getOrCreate()
sc = spark.sparkContext
sc.setLogLevel("WARN") 


Q1_file = spark.read.text("../Data/NASA_access_log_Jul95.gz").cache()   
#Q1_file.show(20, False)


Q1_data = Q1_file.withColumn('host', F.regexp_extract('value', '^(.*) - -.*', 1)) \
                .withColumn('timestamp', F.regexp_extract('value', '.* - - \[(.*)\].*',1)) \
                .withColumn('request', F.regexp_extract('value', '.*\"(.*)\".*',1)) \
                .withColumn('HTTP reply code', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) -2).cast("int")) \
                .withColumn('bytes in the reply', F.split('value', ' ').getItem(F.size(F.split('value', ' ')) - 1).cast("int")).drop("value").cache()
#Q1_data.show(20,False)



print("---------------")
print(" ")
print(" ")
Jp_hosts = Q1_data.filter(Q1_data.host.endswith(".ac.jp")).cache()
Jp_counts = Jp_hosts.count()
print("The total number of requests for all hosts from Japanese Universities ending in .ac.jp is", Jp_counts)
# 13067

UK_hosts = Q1_data.filter(Q1_data.host.endswith(".ac.uk")).cache()
UK_counts = UK_hosts.count()
print("The total number of requests for all hosts from UK Universities ending in .ac.uk is", UK_counts)
# 25009

US_hosts = Q1_data.filter(Q1_data.host.endswith(".edu")).cache()
US_counts = US_hosts.count()
print("The total number of requests for all hosts from US Universities ending in .edu is", US_counts)
# 218449


objects = ('Japanese', 'UK', 'US')
y_pos = np.arange(len(objects))
performance = [Jp_counts, UK_counts, US_counts]

plt.bar(y_pos, performance, align='center', alpha=0.5)
plt.xticks(y_pos, objects)
plt.ylabel('Number of Hosts')
plt.title('Total number of hosts for universities')


plt.savefig("../Output/Q1_figA.png")

# Forming a new column 'Uni' to extract the university from the host domain
UK_Unis = UK_hosts.withColumn('Uni', F.regexp_extract(F.col('host'), '([^\.]+\.\w+\.\w+$)', 1))
#UK_Unis.show(5, False)

Jp_Unis = Jp_hosts.withColumn('Uni', F.regexp_extract(F.col('host'), '([^\.]+\.\w+\.\w+$)', 1))
#Jp_Unis.show(5, False)


US_Unis = US_hosts.withColumn('Uni', F.regexp_extract(F.col('host'), '([^\.]+\.\w+$)', 1))
#US_Unis.show(5, False)

# Getting the top 9 most frequent universities
Jp_top9 = Jp_Unis.select('Uni').groupBy('Uni').count().sort('count', ascending=False).take(9)
print(Jp_top9)
UK_top9 = UK_Unis.select('Uni').groupBy('Uni').count().sort('count', ascending=False).take(9)
print(UK_top9)
US_top9 = US_Unis.select('Uni').groupBy('Uni').count().sort('count', ascending=False).take(9)
print(US_top9)

# Forming lists of the top 9 universities and the top 9 scores
UK_unis = []
UK_scores = []
UK_tot = 0
for uni, counts in UK_top9:
	UK_unis.append(uni)
	UK_scores.append(counts)
	UK_tot += counts

UK_rest = UK_hosts.count() - UK_tot
UK_scores.append(UK_rest)
UK_unis.append('rest')

fig = plt.figure(figsize =(10, 7)) 	
plt.pie(UK_scores,labels=UK_unis,autopct='%1.1f%%')
plt.title('Percentage of the top 9 Universities and the rest')
plt.axis('equal')
plt.savefig("../Output/Q1_figB.png")

Jp_unis = []
Jp_scores = []
Jp_tot = 0
for uni, counts in Jp_top9:
	Jp_unis.append(uni)
	Jp_scores.append(counts)
	Jp_tot += counts

Jp_rest = Jp_hosts.count() - Jp_tot
Jp_scores.append(Jp_rest)
Jp_unis.append('rest')
Jp_scores
Jp_unis

fig = plt.figure(figsize =(10, 7)) 
plt.pie(Jp_scores,labels=Jp_unis,autopct='%1.1f%%')
plt.title('Percentage of the top 9 Universities and the rest')
plt.axis('equal')
plt.savefig("../Output/Q1_figC.png")	

US_unis = []
US_scores = []
US_tot = 0
for uni, counts in US_top9:
	US_unis.append(uni)
	US_scores.append(counts)
	US_tot += counts

US_rest = US_hosts.count() - US_tot
US_scores.append(US_rest)
US_unis.append('rest')
US_scores
US_unis

fig = plt.figure(figsize =(10, 7)) 
plt.pie(US_scores,labels=US_unis,autopct='%1.1f%%')
plt.title('Percentage of the top 9 Universities and the rest')
plt.axis('equal')
plt.savefig("../Output/Q1_figD.png")




UK_first = UK_Unis.filter(UK_Unis.Uni.contains(UK_unis[0])).cache()
#UK_first.show(5, False)


# Need to find out the days

UK_first.select('timestamp')
UK_max_date = UK_first.agg({"timestamp": "max"}).collect()[0]
UK_min_date = UK_first.agg({"timestamp": "min"}).collect()[0]



split_col = F.split(UK_first['timestamp'], '/')
split2 = F.split(UK_first['timestamp'], '\:')
UK_df2 = UK_first.withColumn('day', split_col.getItem(0)) \
       .withColumn('hour', split2.getItem(1))
UK_df2.show(5, False) 

# Need to check this & get it working
heat_UK_counts = UK_df2.groupBy("day", "hour").count()


heat_cols = ["day", "hour"]

sorted_UK_heat = heat_UK_counts.orderBy(heat_cols, ascending=True)
sorted_UK_heat.show(20, False)

US_first = US_Unis.filter(US_Unis.Uni.contains(US_unis[0])).cache()

US_first.select('timestamp')
US_max_date = US_first.agg({"timestamp": "max"}).collect()[0]
US_min_date = US_first.agg({"timestamp": "min"}).collect()[0]



split_col_US = F.split(US_first['timestamp'], '/')
split2_US = F.split(US_first['timestamp'], '\:')
US_df2 = US_first.withColumn('day', split_col.getItem(0)) \
       .withColumn('hour', split2.getItem(1))
US_df2.show(5, False) 

# Need to check this & get it working
heat_US_counts = US_df2.groupBy("day", "hour").count()


heat_cols = ["day", "hour"]

sorted_US_heat = heat_US_counts.orderBy(heat_cols, ascending=True)
sorted_US_heat.show(20, False)


Jp_first = Jp_Unis.filter(Jp_Unis.Uni.contains(Jp_unis[0])).cache()

Jp_first.select('timestamp')
Jp_max_date = Jp_first.agg({"timestamp": "max"}).collect()[0]
Jp_min_date = Jp_first.agg({"timestamp": "min"}).collect()[0]



split_col_Jp = F.split(Jp_first['timestamp'], '/')
split2_Jp = F.split(Jp_first['timestamp'], '\:')
Jp_df2 = Jp_first.withColumn('day', split_col.getItem(0)) \
       .withColumn('hour', split2.getItem(1))
Jp_df2.show(5, False) 

# Need to check this & get it working
heat_Jp_counts = Jp_df2.groupBy("day", "hour").count()


heat_cols = ["day", "hour"]

sorted_Jp_heat = heat_Jp_counts.orderBy(heat_cols, ascending=True)
sorted_Jp_heat.show(20, False)

# Extracting the highest number of requests in a timestamp for each country
Jp_max_c = sorted_Jp_heat.select("count").rdd.max()[0]
UK_max_c = sorted_UK_heat.select("count").rdd.max()[0]
US_max_c = sorted_US_heat.select("count").rdd.max()[0]
# Not sure how to account for the missing timestamps to convert it to a heatmap

