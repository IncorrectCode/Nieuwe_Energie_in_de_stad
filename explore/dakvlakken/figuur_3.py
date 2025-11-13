import pandas as pd
import matplotlib.pyplot as plt

# Excelbestand inladen
bestand = "databim ruwe data dakvlakken.xlsx"
dakvlakken = pd.read_excel(bestand)

# Hoogteverschillen
plt.figure(figsize=(8, 5))
plt.hist(dakvlakken["b3_h_max"], bins=40, color="lightcoral", edgecolor="black")

plt.title("Hoogteverschillen tussen dakvlakken", fontsize=14)
plt.xlabel("Hoogte (meter)", fontsize=12)
plt.ylabel("Aantal dakvlakken", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()