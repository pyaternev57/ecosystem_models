from ld_utils.spark_utils import create_spark_session, save2ps, sdf2cluster, delete_table_from_ps
from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import date, timedelta
from ld_utils.utils import get_last_day_of_current_month
from ld_utils.utils import dict_from_file, get_list_of_cities
import os

# spark = create_spark_session('sberprime task', n_executors=16, n_cores=8)

# os.system("hdfs dfs -rm -r -skipTrash lk_train")
# df = spark.table("sbx_t_team_mp_cmpn_ds.lk_train ")
# df2 = df.sampleBy("target", {1: 0.5, 0: 0.5})
# df2.groupBy("target").count().show()
# df2.coalesce(1).write.option('header', 'true').csv("hdfs://nsld3/user/pyaternev1-is_ca-sbrf-ru/lk_train")
# os.system("rm -rf  ../../notebooks/datasets/lk_train")
# os.system(" hdfs dfs -copyToLocal lk_train/ ../../notebooks/datasets/")


path2conf = "../conf"
# path2conf = "conf"
partner = 'lk'
# partner_name = dict_from_file(f"{path2conf}/partners.json")[partner]
first_dt = date(2021, 2, 1)
pivot_dt = date(2020, 12, 1)
last_dt = date(2021, 1, 1)
# TODO list with all cities?
# cities = get_list_of_cities(partner, path2conf=path2conf)
cities = ['Москва']
join_columns = ['epk_id']
target_source = "sbx_t_team_mp_cmpn_ds. local_kitchen_clients"

## CONNECT TO DB

spark = create_spark_session('sberprime task', n_executors=16, n_cores=8)


# TODO create features about ecosystem
##
## TARGET CREATION.

# sales = spark.table(target_source).where(F.col("partner_name") == partner_name)
# sales = sales.where((F.col("evt_dt") < last_dt) & (F.col("evt_dt") >= first_dt))
# sales = sales.withColumn("report_dt_part", F.expr("date_sub(evt_dt, dayOfMonth(evt_dt))"))
# sales = sales.select(*ощ).withColumn("target", F.lit(1))
# join_columns

sales = spark.table(target_source)
sales = sales.withColumn("target", F.lit(1))

# TODO dates as parameter

## ADD FEATURES

aggr = spark.table("sklod_dwh_sbx_retail_mp_ext.ft_client_aggr_mnth_epk")\
    .where(F.col("report_dt_part") ==  first_dt - timedelta(days=1))
# aggr = aggr.drop(["report_dt", "ctl_loading"])
kn = spark.table("sklod_dwh_sbx_retail_mp_ext.dm_client_knowledge_epk ")\
    .where(F.col("report_dt_part") ==  first_dt - timedelta(days=1))
if cities:
    kn = kn.where(F.col('client_city').isin(cities))
# kn = kn.drop("report_dt", "ctl_loading")
for column in ["report_dt", "ctl_loading", "report_dt_part"]:
    aggr = aggr.drop(column)
    kn = kn.drop(column)
dataset = aggr.join(kn, ["epk_id"], how='inner').join(sales, ["epk_id"], how='left')
dataset = dataset.fillna({"target": 0})

save2ps(dataset, "lkrscore")

spark.stop()