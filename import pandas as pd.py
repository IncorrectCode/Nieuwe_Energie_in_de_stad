import pandas as pd
import psycopg2
import plotly.express as px

# connectie naar de JUISTE database
conn = psycopg2.connect(
    dbname="RetrofitEuropeGroup",
    user="DATABIM2526AB",
    password=r"pv6QGtNJ#rs\#<xoc#'n",
    host="145.38.184.32",
    port="5432"
)

# haal energielabels op uit schema futurefactory
query = """
SELECT energielabel
FROM futurefactory.utrecht_opgemaakt;
"""
df = pd.read_sql(query, conn)

# tel hoeveel woningen per energielabel
counts = (
    df['energielabel']
    .value_counts(dropna=False)
    .rename_axis("energielabel")
    .reset_index(name="aantal")
)

# vervang lege waarden door "Niet ingeleverd"
counts['energielabel'] = counts['energielabel'].fillna("Niet ingeleverd")

# maak plot
fig = px.bar(
    counts,
    x="energielabel",
    y="aantal",
    title="Verdeling energielabels (incl. niet ingeleverd)",
    text="aantal"
)
fig.update_traces(textposition="outside")
fig.show()

conn.close()
