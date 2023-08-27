import pandas as pd
# Contoh csv
df = pd.read_csv('titanic_dataset.csv')

# Contoh excel
df = pd.read_excel('titanic_dataset.xlsx')

# Contoh sqlite
import sqlite3
conn = sqlite3.connect('titanic_dataset.sqlite3')
df = pd.read_sql_query("SELECT * FROM titanic", conn)