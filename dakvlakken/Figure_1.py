import pandas as pd
import matplotlib.pyplot as plt

# Excel-bestand inlezen
file_path = "databim ruwe data.xlsx"
dakvlakken_df = pd.read_excel(file_path)

plt.figure(figsize=(8, 5))
plt.hist(dakvlakken_df["b3_hellingshoek"], bins=30, color='lightblue', edgecolor='black')
plt.title("Verdeling van dakhellingshoeken", fontsize=14, pad=10)
plt.xlabel("Hellingshoek (°)", fontsize=12)
plt.ylabel("Aantal dakvlakken", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
