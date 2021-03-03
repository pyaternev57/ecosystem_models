from ld_utils.spark_utils import create_spark_session, save2ps
from tqdm import tqdm
from pyspark.sql.types import *
import pyspark.sql.functions as F
from itertools import chain
from datetime import date, timedelta

spark = create_spark_session(name="day_aggr", n_executors=8, n_cores=8)

start_dt = date(2020, 12, 31)
end_dt = date.today()
table_name = "five_day_aggr_with_epk"

def compress_and_add_epk(start_dt, name, delta):
    end_dt = start_dt + timedelta(days=delta)
    for _ in tqdm(range((date.today() - start_dt).days // delta)):
        table = spark.table("sbx_t_team_mp_cmpn_ds.day_aggr").where((F.col("report_dt") >= start_dt) &
                                                                    (F.col("report_dt") <= end_dt))
        table.columns =

