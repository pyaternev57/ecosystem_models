from ld_utils.td_utils import create_connection, create_table
from ld_utils.utils import dict_from_file
from ld_utils.spark_utils import create_spark_session, custom_load, make_sql, save2ps

config = dict_from_file("../conf/logins.json")

spark = create_spark_session('partner_transfer', n_executors=8, n_cores=8)
sql = make_sql("sbx_retail_mp_de ", "dm_prime_report", columns=[
    "name",
    "packet_name",
    "packet_category",
    "epk_id",
    "state",
    "integral_state",
    "packet_offer_price",
    "expired_date",
    "recurrent",
    "trial",
    "trialend_date",
    "start_date",
    "activate_date",
    "close_date",
    "report_dt"
])
df = custom_load(spark, sql, config)
save2ps(df, 'dm_prime_report', partition="report_dt")