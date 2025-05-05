# Databricks notebook source
# MAGIC %md
# MAGIC # Import Packages

# COMMAND ----------

import pandas as pd
import os
import pyspark.sql.functions as f
from pyspark.sql.types import StructType, StructField, StringType, IntegerType, FloatType
from pyspark.sql.functions import udf

import matplotlib.pyplot as plt
import seaborn as sns
sns.set()

# COMMAND ----------

# MAGIC %md
# MAGIC # Define Paths and Load Data

# COMMAND ----------

table_values_path = 'ewz_dap_netze_int.pv_analysis.pronovo_pv_data'
table_stat_path = 'ewz_dap_netze_int.pv_analysis.pronovo_pv_stat'

aggregated_data_path = 'ewz_dap_netze_int.pv_analysis.pronovo_pv_data_aggregated'


# COMMAND ----------

# Load Data
pv_stat_sdf = spark.read.table(table_stat_path)
pv_data_sdf = spark.read.table(table_values_path)

# COMMAND ----------

# MAGIC %md
# MAGIC # PV Analysis

# COMMAND ----------

# MAGIC %md
# MAGIC ## Variables

# COMMAND ----------

# Number of null values per year to be filtered out (35040 = 1 year)
min_nbr_non_null = 35000
# Maximum peak power in relation to installed power. If peak power more than this factor above installed power, the generator is filtered out
max_peak_power_factor = 1.1
# PV reduction factor
pv_reduction_factor = 0.7
# Elevation bin start
elevation_start = 500
# Elevation bin end
elevation_end = 2000
# Bin-size
bin_step = 100

# COMMAND ----------

# MAGIC %md
# MAGIC ## Calculation

# COMMAND ----------

# Filter data to contain only production data and join on static data to get TotalPower, Canton and elevation of each plant
pv_data_analysis = pv_data_sdf.filter(f.col('Linie') == '-A').join(pv_stat_sdf.filter(f.col('Linie') == '-A').select('ZPB', 'TotalPower', 'Canton', 'elevation'), 'ZPB', how = 'inner')

# For each plant calculate production limit based on total power and the resulting energy loss in kWh for 60%-reduction to 80%-reduction
pv_data_analysis = pv_data_analysis\
    .withColumn('Values_limited_60', 
                f.when(f.col('Values') > f.col('TotalPower')*0.6/4, f.col('TotalPower')*0.6/4).otherwise(f.col('Values')))\
    .withColumn('Values_limited_65', 
                f.when(f.col('Values') > f.col('TotalPower')*0.65/4, f.col('TotalPower')*0.65/4).otherwise(f.col('Values')))\
    .withColumn('Values_limited_70', 
                f.when(f.col('Values') > f.col('TotalPower')*0.7/4, f.col('TotalPower')*0.7/4).otherwise(f.col('Values')))\
    .withColumn('Values_limited_75', 
                f.when(f.col('Values') > f.col('TotalPower')*0.75/4, f.col('TotalPower')*0.75/4).otherwise(f.col('Values')))\
    .withColumn('Values_limited_80', 
                f.when(f.col('Values') > f.col('TotalPower')*0.8/4, f.col('TotalPower')*0.8/4).otherwise(f.col('Values')))\
    .withColumn('Energy_Loss_kWh_60', f.col('Values') - f.col('Values_limited_60'))\
    .withColumn('Energy_Loss_kWh_65', f.col('Values') - f.col('Values_limited_65'))\
    .withColumn('Energy_Loss_kWh_70', f.col('Values') - f.col('Values_limited_70'))\
    .withColumn('Energy_Loss_kWh_75', f.col('Values') - f.col('Values_limited_75'))\
    .withColumn('Energy_Loss_kWh_80', f.col('Values') - f.col('Values_limited_80'))

# Data is aggregated per year and ZPB
pv_data_analysis_agg = pv_data_analysis.groupBy('Year', 'ZPB', 'Canton', 'elevation').agg(
    f.sum(f.when(f.col('Values').isNull(), 1)).alias('null_count'),
    f.sum(f.when(f.col('Values').isNotNull(), 1)).alias('not_null_count'),
    f.sum('Values').alias('Total_Energy_kWh'),
    (f.max('Values') * 4).alias('PeakPower_kW'),
    f.max('TotalPower').alias('TotalPower'),
    f.sum('Energy_Loss_kWh_60').alias('Energy_Loss_kWh_60'),
    f.sum('Energy_Loss_kWh_65').alias('Energy_Loss_kWh_65'),
    f.sum('Energy_Loss_kWh_70').alias('Energy_Loss_kWh_70'),
    f.sum('Energy_Loss_kWh_75').alias('Energy_Loss_kWh_75'),
    f.sum('Energy_Loss_kWh_80').alias('Energy_Loss_kWh_80'),
    f.max('Values_limited_60').alias('PeakPowerLimited_kW_60'),
    f.max('Values_limited_65').alias('PeakPowerLimited_kW_65'),
    f.max('Values_limited_70').alias('PeakPowerLimited_kW_70'),
    f.max('Values_limited_75').alias('PeakPowerLimited_kW_75'),
    f.max('Values_limited_80').alias('PeakPowerLimited_kW_80')
)

# The energy loss fraction is calculated for 60% to 80% reduction
pv_data_analysis_agg = pv_data_analysis_agg\
    .withColumn('Energy_Loss_fraction_60', f.col('Energy_Loss_kWh_60')/f.col('Total_Energy_kWh'))\
    .withColumn('Energy_Loss_fraction_65', f.col('Energy_Loss_kWh_65')/f.col('Total_Energy_kWh'))\
    .withColumn('Energy_Loss_fraction_70', f.col('Energy_Loss_kWh_70')/f.col('Total_Energy_kWh'))\
    .withColumn('Energy_Loss_fraction_75', f.col('Energy_Loss_kWh_75')/f.col('Total_Energy_kWh'))\
    .withColumn('Energy_Loss_fraction_80', f.col('Energy_Loss_kWh_80')/f.col('Total_Energy_kWh'))\

# The data is being stored
pv_data_analysis_agg.write.mode('overwrite').saveAsTable(aggregated_data_path)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Analyses & Plots

# COMMAND ----------

# The data is reloaded
pv_data_analysis_agg_reload = spark.read.table(aggregated_data_path)

# Entries with missing datapoints are filtered out
missing_values_sdf = pv_data_analysis_agg_reload.filter(f.col('not_null_count') < min_nbr_non_null)

# Units with unreasonably high production
elevated_power_sdf = pv_data_analysis_agg_reload.filter(f.col('PeakPower_kW') > f.col('TotalPower')*max_peak_power_factor)

# Valid units - only units with missing datapoints above threshold are filtered out
valid_units_sdf = pv_data_analysis_agg_reload.filter((f.col('not_null_count') >= min_nbr_non_null))

#valid_units_sdf.display()

# COMMAND ----------

missing_values_sdf.select('Year', 'ZPB', 'null_count', 'not_null_count').display()

# COMMAND ----------

elevated_power_sdf.select('Year', 'ZPB', 'PeakPower_kW', 'TotalPower').display()

# COMMAND ----------

# Function to create elevation bins between start and end with stepsize of step
def create_elevation_bin(elevation, start=elevation_start, end=elevation_end, step=bin_step):
    for i in range(start, end + step, step):
        if int(elevation) <= i:
            return i
    return i+step

# define udf function
elevation_bin_udf = udf(create_elevation_bin, IntegerType())

# Keep only rows with elevation non null and assign elevation bin
ch_bined_sdf = valid_units_sdf.filter(f.col('elevation').isNotNull()).withColumn('elevation_bin', elevation_bin_udf(f.col('elevation')))
# aggregate bined data by elevation bin and calculate the number of units above and fraction of units below 3% energy loss
ch_aggregated_bined_sdf = ch_bined_sdf.groupBy('elevation_bin').agg(
    f.count('ZPB').alias('TotalCount'),
    f.sum(f.when(f.col('Energy_Loss_fraction_60') > 0.03,1).otherwise(0)).alias('CountAbove3Percent_60'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_60') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent_60'),
    f.sum(f.when(f.col('Energy_Loss_fraction_65') > 0.03,1).otherwise(0)).alias('CountAbove3Percent_65'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_65') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent_65'),
    f.sum(f.when(f.col('Energy_Loss_fraction_70') > 0.03,1).otherwise(0)).alias('CountAbove3Percent_70'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_70') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent_70'),
    f.sum(f.when(f.col('Energy_Loss_fraction_75') > 0.03,1).otherwise(0)).alias('CountAbove3Percent_75'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_75') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent_75'),
    f.sum(f.when(f.col('Energy_Loss_fraction_80') > 0.03,1).otherwise(0)).alias('CountAbove3Percent_80'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_80') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent_80')
)

# Create a dataframe with number of units above and fraction of units below 2% and 3% energy loss for power restrictions of 60% to 80%
ch_aggregated_tot_sdf = valid_units_sdf.agg(
    f.lit(60).alias('Reduktion'),
    f.count('ZPB').alias('TotalCount'),
    f.sum(f.when(f.col('Energy_Loss_fraction_60') > 0.02,1).otherwise(0)).alias('CountAbove2Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_60') <= 0.02,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow2Percent'),
    f.sum(f.when(f.col('Energy_Loss_fraction_60') > 0.03,1).otherwise(0)).alias('CountAbove3Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_60') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent')
).union(
    valid_units_sdf.agg(
    f.lit(65).alias('Reduktion'),
    f.count('ZPB').alias('TotalCount'),
    f.sum(f.when(f.col('Energy_Loss_fraction_65') > 0.02,1).otherwise(0)).alias('CountAbove2Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_65') <= 0.02,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow2Percent'),
    f.sum(f.when(f.col('Energy_Loss_fraction_65') > 0.03,1).otherwise(0)).alias('CountAbove3Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_65') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent')
)).union(
    valid_units_sdf.agg(
    f.lit(70).alias('Reduktion'),
    f.count('ZPB').alias('TotalCount'),
    f.sum(f.when(f.col('Energy_Loss_fraction_70') > 0.02,1).otherwise(0)).alias('CountAbove2Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_70') <= 0.02,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow2Percent'),
    f.sum(f.when(f.col('Energy_Loss_fraction_70') > 0.03,1).otherwise(0)).alias('CountAbove3Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_70') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent')
)).union(
    valid_units_sdf.agg(
    f.lit(75).alias('Reduktion'),
    f.count('ZPB').alias('TotalCount'),
    f.sum(f.when(f.col('Energy_Loss_fraction_75') > 0.02,1).otherwise(0)).alias('CountAbove2Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_75') <= 0.02,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow2Percent'),
    f.sum(f.when(f.col('Energy_Loss_fraction_75') > 0.03,1).otherwise(0)).alias('CountAbove3Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_75') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent')
)).union(
    valid_units_sdf.agg(
    f.lit(80).alias('Reduktion'),
    f.count('ZPB').alias('TotalCount'),
    f.sum(f.when(f.col('Energy_Loss_fraction_80') > 0.02,1).otherwise(0)).alias('CountAbove2Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_80') <= 0.02,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow2Percent'),
    f.sum(f.when(f.col('Energy_Loss_fraction_80') > 0.03,1).otherwise(0)).alias('CountAbove3Percent'),
    (f.sum(f.when(f.col('Energy_Loss_fraction_80') <= 0.03,1).otherwise(0))/f.count('ZPB')).alias('FractionBelow3Percent')
))


# COMMAND ----------

# Plot Fraction of units below 2% and 3% of yearly energy loss per power restriction level (from 60% until 80%)
ch_aggregated_tot_pdf = ch_aggregated_tot_sdf.toPandas()

plt.figure(figsize=(10, 6))
plt.plot(ch_aggregated_tot_pdf['Reduktion'], ch_aggregated_tot_pdf['FractionBelow2Percent'], label='Anteil Anlagen unter 2%-Jahresenergieverlust')
plt.plot(ch_aggregated_tot_pdf['Reduktion'], ch_aggregated_tot_pdf['FractionBelow3Percent'], label='Anteil Anlagen unter 3%-Jahresenergieverlust')
plt.xlabel('Fixe Einspeisereduktion [in %]')
plt.ylabel('Anteil Anlagen [-]')
plt.title('Einhaltung Jahresenergieverlust gegenüber Einspeisereduktion')
plt.legend()
plt.grid(True)
#plt.ylim(0.5, 1)
plt.show()

# COMMAND ----------

# Plot the fraction of units below 3% energy loss per elevation bin at a power restriction of 70%

ch_aggregated_bined_pdf = ch_aggregated_bined_sdf.orderBy('elevation_bin').toPandas()

plt.figure(figsize=(10, 6))
plt.plot(ch_aggregated_bined_pdf['elevation_bin'], ch_aggregated_bined_pdf['FractionBelow3Percent_70'], marker='o')
plt.xlabel('Höhe [m ü. M.]')
plt.ylabel('Anteil Anlagen unter 3%-Jahresenergieverlust')
plt.title('Anteil Anlagen unter 3%-Jahresenergieverlust pro Höhe')
plt.grid(True)

#for x, y in zip(ch_aggregated_bined_pdf['elevation_bin'], ch_aggregated_bined_pdf['FractionBelow3Percent_70']):
 #   plt.text(x, y, f'{y:.3f}', ha='center', va='bottom')

plt.show()

# COMMAND ----------

ch_aggregated_bined_pdf['CountBelow3Percent_70'] = ch_aggregated_bined_pdf['TotalCount'] - ch_aggregated_bined_pdf['CountAbove3Percent_70']
# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for TotalCount
ax1.bar(ch_aggregated_bined_pdf['elevation_bin'], ch_aggregated_bined_pdf['CountBelow3Percent_70'], color='b', width=80)
ax1.set_xlabel('Höhe [m ü. M.]')
ax1.set_ylabel('Anzahl Anlagen [-]')

# COMMAND ----------

ch_aggregated_bined_pdf[['elevation_bin', 'FractionBelow3Percent_70']].display()

# COMMAND ----------

# Using a bar plot to show the number of units in each elevation bin as well as the cumulative count

# Calculate cumulative count
ch_aggregated_bined_pdf['CumulativeCount'] = ch_aggregated_bined_pdf['TotalCount'].cumsum()

# Plot
fig, ax1 = plt.subplots(figsize=(10, 6))

# Bar plot for TotalCount
ax1.bar(ch_aggregated_bined_pdf['elevation_bin'], ch_aggregated_bined_pdf['TotalCount'], color='b', width=80)
ax1.set_xlabel('Höhe [m ü. M.]')
ax1.set_ylabel('Anzahl Anlagen [-]')

# Line plot for CumulativeCount on secondary y-axis
ax2 = ax1.twinx()
ax2.plot(ch_aggregated_bined_pdf['elevation_bin'], ch_aggregated_bined_pdf['CumulativeCount'], color='r', marker='o')
ax2.set_ylabel('Kummulierte Anzahl Anlagen [-]')

plt.title('Einfache und kummulierte Anzahl an Anlagen pro Höhe')
plt.grid(True)
plt.show()

# COMMAND ----------

