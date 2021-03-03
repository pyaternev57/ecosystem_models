from ld_utils.td_utils import create_connection, create_table
from ld_utils.utils import dict_from_file
from ld_utils.spark_utils import create_spark_session, custom_load, make_sql, save2ps

config = dict_from_file("../conf/logins.json")

spark = create_spark_session('local kitchen', n_executors=16,)
# sql = make_sql("sbx_retail_mp_lm ", "matched_local_kitchen_1202", columns=["epk_id"])
sql = '''select t1.*, t2.mcc_subgroup_name, t2.mcc_group_id from sbx_retail_mp_ca_vd.vsiv_autocj_next_mcc_scores_fnl_corr  t1
left join  sbx_retail_mp_dm.ref_mcc_subgroup t2
on  t1.mcc_subgroup_id = t2.mcc_subgroup_id'''
df = custom_load(spark, sql, config)
save2ps(df, 'knowledge_mcc_test')