from ld_utils.td_utils import create_connection, create_table
from ld_utils.utils import dict_from_file
from ld_utils.spark_utils import create_spark_session, custom_load, make_sql, save2ps

config = dict_from_file("../conf/logins.json")

spark = create_spark_session('sberprime_transfer', n_executors=8, n_cores=8)
sql = make_sql("sbx_retail_mp_lm ", "dm_partner_sales")
df = custom_load(spark, sql, config)
save2ps(df, 'dm_partner_sales', partition="evt_dt")