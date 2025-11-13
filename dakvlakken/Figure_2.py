import pandas as pd
import matplotlib.pyplot as plt

# Excel-bestand inlezen
file_path = "databim ruwe data.xlsx"
dakvlakken_df = pd.read_excel(file_path)

plt.figure(figsize=(8, 5))
plt.hist(dakvlakken_df["b3_azimut"], bins=36, color='lightgreen', edgecolor='black')
plt.title("Verdeling van dakoriëntatie (azimut)", fontsize=14, pad=10)
plt.xlabel("Oriëntatie (° vanaf noorden)", fontsize=12)
plt.ylabel("Aantal dakvlakken", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
