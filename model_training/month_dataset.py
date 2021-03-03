from ld_utils.spark_utils import create_spark_session, save2ps, sdf2cluster, delete_table_from_ps
from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import date, timedelta
from ld_utils.utils import get_last_day_of_current_month
from ld_utils.utils import dict_from_file, get_list_of_cities
import os

## ARGUMENTS

path2conf = "../conf"
# path2conf = "conf"
partner = 'delivery_club'
partner_name = dict_from_file(f"{path2conf}/partners.json")[partner]
first_dt = date(2020, 12, 1)
pivot_dt = date(2021, 1, 1)
last_dt = date(2021, 2, 1)
# TODO list with all cities?
cities = get_list_of_cities(partner, path2conf=path2conf)
join_columns = ['epk_id', "report_dt_part"]
target_source = "sbx_t_team_mp_cmpn_ds.dm_partner_sales"

## CONNECT TO DB

spark = create_spark_session('sberprime task', n_executors=16, n_cores=8)


# TODO create features about ecosystem
##
## TARGET CREATION
partners = [el for el in dict_from_file(f"{path2conf}/partners.json").keys() if el not in ['sberprime', 'level_kitchen']]
for partner in partners:
    print(f"<BEGIN> {partner}")
    try:
        cities = get_list_of_cities(partner, path2conf=path2conf)
    except:
        cities = False
    partner_name = dict_from_file(f"{path2conf}/partners.json")[partner]
    sales = spark.table(target_source).where(F.col("partner_name") == partner_name)
    sales = sales.groupBy("epk_id").agg({"evt_dt": "min"}).withColumnRenamed("min(evt_dt)", "evt_dt")
    sales = sales.where((F.col("evt_dt") < last_dt) & (F.col("evt_dt") >= first_dt))
    sales = sales.withColumn("report_dt_part", F.expr("date_sub(evt_dt, dayOfMonth(evt_dt))"))
    sales = sales.select("epk_id", "report_dt_part").withColumn("target", F.lit(1))
    # join_columns

    # TODO dates as parameter

    ## ADD FEATURES

    aggr = spark.table("sklod_dwh_sbx_retail_mp_ext.ft_client_aggr_mnth_epk")\
        .where((F.col("report_dt_part") ==  first_dt - timedelta(days=1))
               | (F.col("report_dt_part") ==  pivot_dt - timedelta(days=1)))
    # aggr = aggr.drop(["report_dt", "ctl_loading"])
    kn = spark.table("sklod_dwh_sbx_retail_mp_ext.dm_client_knowledge_epk ")\
    .where((F.col("report_dt_part") ==  first_dt - timedelta(days=1))
               | (F.col("report_dt_part") ==  pivot_dt - timedelta(days=1)))
    if cities:
        kn = kn.where(F.col('client_city').isin(cities))
    # kn = kn.drop("report_dt", "ctl_loading")
    for column in ["report_dt", "ctl_loading"]:
        aggr = aggr.drop(column)
        kn = kn.drop(column)
    dataset = aggr.join(kn, ["epk_id", "report_dt_part"], how='inner').join(sales, ["epk_id", "report_dt_part"], how='left')
    dataset = dataset.fillna({"target": 0})

    percents = dataset.groupBy("target").count().toPandas()
    ones = 100000 / percents["count"].tolist()[0]
    zeros = 1900000 / percents["count"].tolist()[1]
    print(dataset.groupBy("report_dt_part").count().show())
    print(ones, zeros)
    train = dataset.where(F.col("report_dt_part") == first_dt - timedelta(days=1))\
        .sampleBy("target", fractions={1: min(ones * 2, 1), 0: min(zeros * 2, 1)})
    print(train.groupBy("target").count().show())
    delete_table_from_ps(f'{partner}_train', spark=spark)
    save2ps(train, f'{partner}_train')
    oot = dataset.where(F.col("report_dt_part") == pivot_dt - timedelta(days=1))\
        .sampleBy("target", fractions={1: min(ones * 4, 1), 0: min(zeros * 1.5, 1)})
    print(train.groupBy("target").count().show())
    delete_table_from_ps(f'{partner}_oot', spark=spark)
    save2ps(train, f'{partner}_oot')
    print(f"<END>")
