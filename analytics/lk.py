from ld_utils.spark_utils import create_spark_session, save2ps, sdf2cluster, delete_table_from_ps
from pyspark.sql.types import *
import pyspark.sql.functions as F
from datetime import date, timedelta
from ld_utils.utils import get_last_day_of_current_month
from ld_utils.utils import dict_from_file, get_list_of_cities
import os

spark = create_spark_session('sberprime task', n_executors=16, n_cores=8)

target_source = "sbx_t_team_mp_cmpn_ds. local_kitchen_clients"

sales = spark.table(target_source)
sales = sales.withColumn("target", F.lit(1))

scores = spark.table("default.lk_scores")

scores = scores.join(sales, ["epk_id"], how='left').fillna(0, ["target"])