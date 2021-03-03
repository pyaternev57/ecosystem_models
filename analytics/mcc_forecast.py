from ld_utils.utils import dict_from_file
from ld_utils.spark_utils import create_spark_session, save2ps
import pyspark.sql.functions as func
from datetime import date

spark = create_spark_session('mcc', n_executors=8, n_cores=8)

for day in range(7, 14):
    current = date(2020, 12, day)
    transactions = spark.table("rozn_custom_rb_smartvista.card_transaction")
    transactions = transactions.select("epk_id", "merchant", "day_part")
    transactions = transactions.where(func.col("day_part") == current)
    transactions = transactions.where(func.col("epk_id").isNotNull())
    if day == 7:
        save2ps(transactions, "transactions4mcc_test", partition='day_part', mode="overwrite")
    else:
        save2ps(transactions, "transactions4mcc_test", partition='day_part', mode="append")

