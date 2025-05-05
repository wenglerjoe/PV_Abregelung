# Databricks notebook source
# MAGIC %md
# MAGIC # Importing Packages

# COMMAND ----------

import pandas as pd
import os
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType

# COMMAND ----------

# MAGIC %md
# MAGIC ## Defining Paths

# COMMAND ----------

pronovo_data_path = "/Volumes/ewz_dap_netze_int/pv_analysis/input/"

#elevation_data_path = f"{pronovo_data_path}pronovo_selected_with_elevation.csv"
#matching_data_path = f"{pronovo_data_path}pronovo_pv_selected_ids_bearb.csv"
#test_data_path = f"{pronovo_data_path}LPEX_Export_2025_04_17_0954_2023_01_01-2025_01_01_42.txt"

table_values_path = 'ewz_dap_netze_int.pv_analysis.pronovo_pv_data'
table_stat_path = 'ewz_dap_netze_int.pv_analysis.pronovo_pv_stat'

path_matching = '/Volumes/ewz_dap_netze_int/pv_analysis/input_meta/pronovo_pv_selected_ids_bearb.csv'
path_elevation = '/Volumes/ewz_dap_netze_int/pv_analysis/input_meta/pronovo_selected_with_elevation.csv'

# COMMAND ----------

# MAGIC %md
# MAGIC ## Loading additional files

# COMMAND ----------

# File with elevation and canton per xtf_id
matching_sdf = (spark.read
                .option("sep", ";")
                .option("header", True)
                .csv(path_matching))
# File with matching between xtf_id and MeteringPointId
elevation_sdf = (spark.read
                .option("sep", ";")
                .option("header", True)
                .csv(path_elevation))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Preparing additional meta data

# COMMAND ----------

# Extract all MeteringPoints from -A
matching_sdf_am = matching_sdf.select("xtf_id", "Messpunkt Produktion (A-)").withColumnRenamed("Messpunkt Produktion (A-)", "ZPB")
# Extract all remaining MeteringPoints from +A which are not in -A
matching_sdf_ap = matching_sdf.filter((f.col("Messpunkt Produktion (A-)") != f.col("Messpunkt Hilfsspeisung (A+)")) & (f.col("Messpunkt Hilfsspeisung (A+)") != "0")).select("xtf_id", "Messpunkt Hilfsspeisung (A+)").withColumnRenamed("Messpunkt Hilfsspeisung (A+)", "ZPB")

# Keeping only elevation, Power and location data - no personalized data
elevation_sdf = elevation_sdf.select('xtf_id', 'Canton', 'TotalPower', 'elevation', '_x', '_y')

# Matching both tables to a final matching table
matching_tot_sdf = matching_sdf_am.union(matching_sdf_ap).join(elevation_sdf, 'xtf_id', how = 'left').select('ZPB', 'Canton', 'TotalPower', 'elevation', '_x', '_y')

# COMMAND ----------

# Defining the structure for Pronovo Data to be read
# First define the static columns
schema = StructType()\
    .add("Datum", StringType(), True)\
    .add("Zeit", StringType(), True)\
    .add("Kundennummer", IntegerType(), True)\
    .add("Kundenname", StringType(), True)\
    .add("eindeutigeKDNr", IntegerType(), True)\
    .add("GEId", StringType(), True)\
    .add("GEKANr", StringType(), True)\
    .add("KALINr", StringType(), True)\
    .add("Linie", StringType(), True)\
    .add("eindeutigeLINr", StringType(), True)\
    .add("ZPB", StringType(), True)\
    .add("Kennzahl", StringType(), True)\
    .add("Einheit", StringType(), True)\
    .add("Wandlerfaktor", IntegerType(), True)\
    .add("MPDauer", IntegerType(), True)
# Then define the data columns
# They are 96 values (24*4) each one followed by a unit
time_values = []
unit_values = []
for hour in range(0, 24):
    for min in range(0, 60, 15):
        schema = schema.add(f"{hour:02d}:{min:02d}", FloatType(), True)
        schema = schema.add(f"{hour:02d}:{min:02d}_Einheit", StringType(), True)
        time_values.append(f"{hour:02d}:{min:02d}")
        unit_values.append(f"{hour:02d}:{min:02d}_Einheit")
   

# COMMAND ----------

def process_raw_sdf(sdf):
    """
    Function to process the input data and return two dataframes.

    First, only the static data is extracted and all duplicates are dropped.
    Then the dynamic data is extracted. The data is then melted to have a long format table. This is done separately for the values and the units and finally joined together.
    """
    # Extract the static information
    sdf_stat = sdf.select('Kundennummer', 'Kundenname', 'eindeutigeKDNr', 'GEId', 'GEKANr', 'KALINr', 'Linie', 'eindeutigeLINr', 'ZPB', 'Kennzahl', 'Einheit', 'Wandlerfaktor', 'MPDauer').dropDuplicates()
    # Extract the values
    sdf_dyn_values = sdf.select(['Datum', 'Linie', 'ZPB'] + time_values)
    # Extract the units
    sdf_dyn_units = sdf.select(['Datum', 'Linie', 'ZPB'] + unit_values)
    # Melt the values dataframe and add a timestamp column
    sdf_dyn_values_melt = sdf_dyn_values.melt(ids = ['Datum', 'Linie', 'ZPB'], values = time_values, variableColumnName='Time', valueColumnName='Values').withColumn('TS', f.to_timestamp(f.concat(f.col('Datum'), f.lit(' '), f.col('Time')), 'dd.MM.yy HH:mm'))
    # Melt the units dataframe and add a timestamp column
    sdf_dyn_units_melt = sdf_dyn_units.melt(ids = ['Datum', 'Linie', 'ZPB'], values = unit_values, variableColumnName='Time', valueColumnName='Units').withColumn('Time', f.col('Time').substr(0, 5)).withColumn('TS', f.to_timestamp(f.concat(f.col('Datum'), f.lit(' '), f.col('Time')), 'dd.MM.yy HH:mm'))
    # Select the relevant columns from both melted dataframes and join on Timestamp, Linie and ZPB
    sdf_dyn_final = sdf_dyn_values_melt.select('TS', 'Linie', 'ZPB', 'Values').join(sdf_dyn_units_melt.select('TS', 'Linie', 'ZPB', 'Units'), ['TS', 'Linie', 'ZPB'], how='left')
    sdf_dyn_final = sdf_dyn_final.withColumn('Year', f.year(f.col('TS')))
    # Return both dataframes
    return sdf_stat, sdf_dyn_final

# COMMAND ----------

# MAGIC %md
# MAGIC # Load all data and store to table

# COMMAND ----------

# Load all input files
sdf = (spark.read
    .option("sep", ";")
    .schema(schema)
    .option("skipRows", 2)
    .csv(pronovo_data_path))
# Split into static and dynamic dataframes
sdf_stat, sdf_dyn_final = process_raw_sdf(sdf)
# Store both dataframes to tables
sdf_stat_matched = sdf_stat.join(matching_tot_sdf, 'ZPB', how = 'left')
sdf_stat_matched.write.mode("overwrite").saveAsTable(table_stat_path)
sdf_dyn_final.write\
    .mode("overwrite")\
    .saveAsTable(table_values_path)