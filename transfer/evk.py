from ld_utils.td_utils import create_connection, create_table, drop_table, grant
from ld_utils.utils import dict_from_file
from ld_utils.spark_utils import create_spark_session, custom_load, make_sql, save2ps
from datetime import date

config = dict_from_file("../conf/logins.json")
print(config)
session = create_connection(config=config)
drop_table(session, "ip_sberprime_cmpns")
sql = """select
            b.epk_id, b.start_dt, b.end_dt, b.cg_flg, b.delivery_ts, b.open_ts, b.contact_ts,
            a.product_name, a.campaign_id, a.channel_name from SBX_RETAIL_MP_DM.v_ref_union_campaign_dic a
inner join SBX_RETAIL_MP_DM.dm_union_campaign_history b on b.sk_id=a.sk_id and product_name = 'Сберпрайм'"""
create_table(session, 'sberprime_cmpns', sql)
grant("ip_sberprime_cmpns", "CA-Bashkirtsev-Dv-0527")
#
# spark = create_spark_session('td_transfer', n_executors=16)
# sql = make_sql("sbx_retail_mp_ds_cmpn", "ip_sberprime_cmpns")
# df = custom_load(spark, sql, config)
# df = df.fillna(str(date(2046, 1, 1)), subset=["contact_ts"])
# save2ps(df, 'sberprime_cmpns', partition="contact_ts")

