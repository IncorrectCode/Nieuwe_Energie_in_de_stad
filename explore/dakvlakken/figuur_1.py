import pandas as pd
import matplotlib.pyplot as plt

# Excelbestand inladen
bestand = "databim ruwe data dakvlakken.xlsx"
dakvlakken = pd.read_excel(bestand)

# Oriëntatie van daken
plt.figure(figsize=(8, 5))
plt.hist(dakvlakken["b3_azimut"], bins=36, color="skyblue", edgecolor="black")

plt.title("Verdeling van dakoriëntaties in Utrecht", fontsize=14)
plt.xlabel("Oriëntatie (graden)", fontsize=12)
plt.ylabel("Aantal dakvlakken", fontsize=12)
plt.grid(axis="y", linestyle="--", alpha=0.7)
plt.tight_layout()
plt.show()
