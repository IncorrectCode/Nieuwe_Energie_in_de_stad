import pandas as pd
import matplotlib.pyplot as plt

# Excelbestand inladen
bestand = "databim ruwe data.xlsx"
corporaties = pd.read_excel(bestand, sheet_name="DATABIM_ruwe_data — corporaties")

# Tellen hoeveel panden per corporatie
corporatie_kolommen = ['bo_ex', 'portaal', 'ssh', 'habion', 'overig']
aantal_per_corporatie = corporaties[corporatie_kolommen].sum().sort_values(ascending=False)

# Visualisatie
plt.figure(figsize=(8, 5))
aantal_per_corporatie.plot(kind='bar', color='skyblue', edgecolor='black')

plt.title("Aantal panden per woningcorporatie", fontsize=14)
plt.xlabel("Corporatie", fontsize=12)
plt.ylabel("Aantal panden", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
