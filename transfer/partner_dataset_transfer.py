from ld_utils.spark_utils import td2cluster
from datetime import date

partner = 'sberprime'
report_dt = date(2020, 11, 30)
dataset_type = 'oot'
file_format = 'csv'

name = f"{partner}_{dataset_type}_{report_dt}".replace("-", "_")
td2cluster(f"ip_{name}", "sbx_retail_mp_ds_cmpn", name)