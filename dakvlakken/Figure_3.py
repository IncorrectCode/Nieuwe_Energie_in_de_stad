import pandas as pd
import matplotlib.pyplot as plt

# Excel-bestand inlezen
file_path = "databim ruwe data.xlsx"
dakvlakken_df = pd.read_excel(file_path)

plt.figure(figsize=(6, 5))
counts = dakvlakken_df["hoofddakconstructie"].value_counts()
bars = counts.plot(kind='bar', color=['steelblue', 'lightgray'], edgecolor='black')

for bar in bars.patches:
    height = bar.get_height()
    bars.annotate(f'{height}',
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 5),
                  textcoords='offset points',
                  ha='center', va='bottom', fontsize=10)

plt.title("Verdeling hoofddaken vs nevendaken", fontsize=14, pad=10)
plt.xlabel("Type dak", fontsize=12)
plt.ylabel("Aantal dakvlakken", fontsize=12)
plt.xticks([0, 1], ["Hoofddak", "Nevendak"])
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
