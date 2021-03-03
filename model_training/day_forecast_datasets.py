from ld_utils.spark_utils import create_spark_session, save2ps, ps2cluster, sdf2cluster, delete_folder_from_ps
from ld_utils.utils import dict_from_file, get_list_of_cities
import pyspark.sql.functions as F
from datetime import date, timedelta

partner = "delivery_club"
last_dt = date(2021, 1, 1)
pivot_dt = date(2020, 12, 1)
first_dt =  date(2020, 10, 1)
# TODO create function that construct path2conf through enviroment
path2conf = "../conf" # uncomment if run as py
# path2conf = "conf"

spark = create_spark_session('dataset_creation', n_executors=16, n_cores=8, executor_memory=32, driver_memory=64, )
partners = dict_from_file(f"{path2conf}/partners.json")
aggr = spark.table("sbx_t_team_mp_cmpn_ds.day_aggr")
aggr = aggr.where(F.col("report_dt") < last_dt)
aggr = aggr.where (F.col("client_city").isin(get_list_of_cities(partner, path2conf=path2conf)))
sales = spark.table("sbx_t_team_mp_cmpn_ds.dm_partner_sales")
sales = sales.where(F.col("partner_name") == partners[partner])
sales = sales.where((F.col("evt_dt") < last_dt) & (F.col("evt_dt") >= first_dt))
sales = sales.withColumn("target", F.lit(1))
sales = sales.withColumn("report_dt", F.date_sub('evt_dt', 3))
# sales = sales.withColumnRenamed("evt_dt", "report_dt")
sales = sales.select("epk_id", "report_dt", "target")

# TODO crete good algorithm to do this shit
dataset = aggr.join(sales, ['epk_id', "report_dt"], how="left")
print(dataset.select(F.sum("target").alias('amount_of')).show())
dataset = dataset.fillna({"target": 0})
dataset = dataset.where(F.col("report_dt").isNotNull())
train = dataset.where(F.col("report_dt") < pivot_dt).sampleBy("target", fractions={1: 0.05, 0: 0.95})
# sdf2cluster(train, f'{partner}_train')
spark.sql(f"drop table if exists sbx_t_team_mp_cmpn_ds.{partner}_train")
delete_folder_from_ps(f'{partner}_train')
save2ps(train, f'{partner}_train', partition="report_dt")
oot = dataset.where(F.col("report_dt") >= pivot_dt).sampleBy("target", fractions={1: 0.03, 0: 0.97})
# sdf2cluster(oot, f'{partner}_oot')
spark.sql(f"drop table if exists sbx_t_team_mp_cmpn_ds.{partner}_oot")
delete_folder_from_ps( f'{partner}_oot')
save2ps(oot, f'{partner}_oot', partition="report_dt")
spark.stop()