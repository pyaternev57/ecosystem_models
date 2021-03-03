from ld_utils.spark_utils import create_spark_session, save2ps
from tqdm import tqdm
from pyspark.sql.types import *
import pyspark.sql.functions as F
from itertools import chain
from datetime import date, timedelta
from ld_utils.utils import get_last_day_of_current_month
import os

spark = create_spark_session(name="day_aggr", n_executors=8, n_cores=8)

chosen_types = [
    737,  # Транзакция   завершения покупки на POS-терминале
    787,  # Вторая   часть 2-этапной кредитной транзакции на POS- терминале
    890,  # Перевод   средств со счета на счет в пределах карты
    502,  # Покупка   за бонусные баллы
    501,  # Покупка   с частичным использованием бонусных баллов
    511,  # Покупка   по топливной карте
    782,  # Дебетовая   часть транзакции перевода с карты на карту
    680,  # Покупка   через ePOS –терминал
    678,  # Завершение   покупки через ePOS-терминал
    776,  # Транзакция   Покупка со сдачей на POS-терминале
    700,  # Выдача   наличных через банкомат
    699,  # Снятие   наличных с POS-терминала без карты
    774,  # покупка в POS

]

data = {'Здоровье и красота': 'health',
        'Одежда и аксессуары': 'clothes',
        'Образование': 'education',
        'Отдых и развлечения': 'entertainment',
        'Супермаркеты': 'supermarkets',
        'Транспорт': 'transport',
        'Путешествия': 'travel',
        'Все для дома': 'home',
        'Рестораны и кафе': 'restaurants',
        'Искусство': 'art',
        'Перевод с карты': 'transfer_from',
        'Перевод на карту': 'transfer_to',
        'Перевод со вклада': 'to_deposit',
        'Перевод на вклад': 'transfer_out',
        'Перевод во вне': 'cash_out',
        'Выдача наличных': 'cash_in',
        'Внесение наличных': 'comission',
        'Комиссия': 'loan_repayment',
        'Погашение кредитов': 'add_money',
        'Зачисления': 'automobile',
        'Автомобиль': 'utilities'}


# TODO create differnet functions for all merchant codes and categories
def create_aggr(start, finish, chosen_types=chosen_types, translation=data, id_name="epk_id"):
    spark.sql("refresh table rozn_custom_rb_smartvista.card_transaction")
    data = spark.table("rozn_custom_rb_smartvista.card_transaction")
    period = data.where((data["day_part"] >= start) & (data["day_part"] <= finish))
    period = period.withColumn('trans_type', period['trans_type'].cast(IntegerType()))
    period = period.withColumn('merchant', period['merchant'].cast(IntegerType()))
    period = period.withColumn('actamt', period['actamt'].cast(IntegerType()))
    codes = spark.table("rozn_custom_rb_smartvista.mcc_code")
    mapping = F.create_map([F.lit(x) for x in chain(*translation.items())])
    codes = codes.select("mcc_code", "mcc_group_name", "mcc_group_id", "is_debit", mapping[codes["mcc_group_name"]] \
                         .alias("column_name")).fillna('other', ['column_name'])
    codes = codes.select(F.col("mcc_code").alias("merchant"), "column_name")
    period = period.join(codes, 'merchant')
    transactions = period.where(period["trans_type"].isin(chosen_types)). \
        groupBy([id_name, "column_name"]). \
        agg(F.sum("actamt"), F.max("actamt"), F.min("actamt"), F.count("actamt"))
    return transactions


def create_pivot_table(transactions, id_name="epk_id"):
    pivot_table = transactions.groupBy(id_name).pivot("column_name").agg((F.mean("sum(actamt)") / 100).alias("sum_amt"),
                                                                        (F.mean("max(actamt)") / 100).alias("max_amt"),
                                                                        (F.mean("min(actamt)") / 100).alias("min_amt"))
    pivot_table = pivot_table.fillna(0)
    return pivot_table

# TODO create pipeline for different tasks
# def add_knowledge_info(spark, table, start_dt):
#     table = table.where(F.col("epk_id").isNotNull())
#     knowledge = spark.table("sklod_dwh_sbx_retail_mp_ext.dm_client_knowledge_epk").\
#         where(F.col("report_dt_part") == get_last_day_of_current_month(start_dt))
#     knowledge = knowledge.drop(F.col("report_dt"))
#     table = table.join(knowledge, ["epk_id"], how="left")
#     return table

start_dt = date(2020, 10, 1)
end_dt = date(2020, 12, 31)
table_name = "day_aggr"

def write_aggr2ps(start_dt, name, delta, id_name="epk_id"):
    end_dt = start_dt + timedelta(days=delta - 1)
    for _ in tqdm(range((date(2021, 12, 1) - start_dt).days // delta)):
        aggr_days = create_aggr(start_dt, end_dt)
        pv_aggr_days = create_pivot_table(aggr_days)
        pv_aggr_days = pv_aggr_days.withColumn("report_dt", F.lit(end_dt))
        pv_aggr_days = pv_aggr_days.where(F.col(id_name).isNotNull())
        features = add_knowledge_info(spark, pv_aggr_days, start_dt)
        save2ps(features, name, mode="append", partition="report_dt")
        start_dt = start_dt + timedelta(days=delta)
        end_dt = end_dt + timedelta(days=delta)


os.system(f"hdfs dfs -rm -r -skipTrash hdfs://clsklsbx/user/team/team_mp_cmpn_ds/hive/{table_name}")
write_aggr2ps(start_dt, table_name, 1)