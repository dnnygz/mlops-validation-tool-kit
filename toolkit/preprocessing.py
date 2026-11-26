"""Generación y procesamiento local con PySpark.

En local, el proceso Python es el *driver* y la JVM crea executors/hilos workers.
Las transformations (select/filter/groupBy) son lazy; una action (count/write)
dispara el DAG. Repartition, groupBy y join por claves suelen causar shuffle.
"""
from __future__ import annotations

from pathlib import Path
import os
import sys

from pyspark.sql import DataFrame, SparkSession, functions as F, types as T


def create_spark(app_name: str = "mlops-validation") -> SparkSession:
    """Crea Spark local: dos hilos simulan workers y UI desactivada para CI."""
    # SparkContext lee estas variables al arrancar el worker Python.
    os.environ["PYSPARK_PYTHON"] = sys.executable
    os.environ["PYSPARK_DRIVER_PYTHON"] = sys.executable
    return (
        SparkSession.builder.master("local[2]")
        .appName(app_name)
        .config("spark.ui.enabled", "false")
        .config("spark.sql.shuffle.partitions", "4")
        # Garantiza que driver y workers usen el mismo Python del entorno virtual.
        .config("spark.pyspark.python", sys.executable)
        .config("spark.pyspark.driver.python", sys.executable)
        .getOrCreate()
    )


def generate_transactions(spark: SparkSession, n_rows: int = 2_000, seed: int = 42) -> DataFrame:
    """Genera fraude sintético, reproducible y desbalanceado (~5-8%)."""
    base = spark.range(n_rows).withColumnRenamed("id", "transaction_id")
    amount = F.round(F.exp(F.randn(seed) * 1.0 + 3.5), 2)
    score = (
        F.when(amount > 100, 1.6).otherwise(0.0)
        + F.when(F.rand(seed + 1) < 0.12, 1.3).otherwise(0.0)
        + F.when(F.rand(seed + 2) < 0.10, 1.2).otherwise(0.0)
        + F.randn(seed + 3) * 0.7
    )
    return (
        base.withColumn("amount", amount)
        .withColumn("hour", (F.rand(seed + 4) * 24).cast("int"))
        .withColumn("customer_age", (18 + F.rand(seed + 5) * 62).cast("int"))
        .withColumn("account_age_days", (F.rand(seed + 6) * 3650).cast("int"))
        .withColumn("is_international", (F.rand(seed + 1) < 0.12).cast("int"))
        .withColumn(
            "channel",
            F.when(F.rand(seed + 7) < 0.55, "mobile")
            .when(F.rand(seed + 8) < 0.75, "web")
            .otherwise("atm"),
        )
        .withColumn("label", (score > 2.0).cast("int"))
    )


def build_medallion(spark: SparkSession, output_dir: Path, n_rows: int = 2_000) -> DataFrame:
    """Persiste Bronze/Silver/Gold en Parquet y devuelve Gold."""
    bronze_path = output_dir / "bronze"
    silver_path = output_dir / "silver"
    gold_path = output_dir / "gold"
    generate_transactions(spark, n_rows).write.mode("overwrite").parquet(str(bronze_path))  # action
    bronze = spark.read.parquet(str(bronze_path))
    silver = (
        bronze.dropDuplicates(["transaction_id"])
        .dropna(subset=["transaction_id", "amount", "label"])
        .filter((F.col("amount") >= 0) & F.col("label").isin(0, 1))
    )
    silver.write.mode("overwrite").parquet(str(silver_path))
    silver = spark.read.parquet(str(silver_path))
    gold = (
        silver.withColumn("log_amount", F.log1p("amount"))
        .withColumn("is_night", F.col("hour").between(0, 5).cast("int"))
        .withColumn("channel_index", F.when(F.col("channel") == "mobile", 0).when(F.col("channel") == "web", 1).otherwise(2))
    )
    gold.write.mode("overwrite").parquet(str(gold_path))
    return spark.read.parquet(str(gold_path))


def sql_examples(gold: DataFrame) -> DataFrame:
    """Muestra equivalencia SQL/DataFrame para una agregación con shuffle."""
    gold.createOrReplaceTempView("gold_transactions")
    sql_result = gold.sparkSession.sql(
        "SELECT channel, count(*) AS transactions, avg(amount) AS avg_amount "
        "FROM gold_transactions GROUP BY channel ORDER BY transactions DESC"
    )
    # Equivalente DataFrame: gold.groupBy("channel").agg(F.count("*").alias(...), F.avg("amount"))
    return sql_result


def join_example(gold: DataFrame) -> DataFrame:
    """Ejemplo de join; broadcast evita shuffle para una tabla pequeña."""
    risk = gold.sparkSession.createDataFrame([("mobile", 1.0), ("web", 1.1), ("atm", 0.8)], ["channel", "risk_weight"])
    return gold.join(F.broadcast(risk), "channel", "left")

