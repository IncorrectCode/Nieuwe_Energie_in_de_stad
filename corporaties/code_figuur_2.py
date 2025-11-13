import pandas as pd
import matplotlib.pyplot as plt

# Excel-bestand inlezen
file_path = "databim ruwe data.xlsx"
corporaties_df = pd.read_excel(file_path)

# Totale oppervlakte per corporatie berekenen
total_area_per_corp = corporaties_df.groupby("corporatie")["shape__area"].sum().sort_values(ascending=False)

# Balkgrafiek maken
plt.figure(figsize=(12, 6))
bars = total_area_per_corp.plot(kind='bar', color='skyblue', edgecolor='black')

# Waarden boven de balken zetten
for i, value in enumerate(total_area_per_corp):
    bars.annotate(f'{value:.0f}', xy=(i, value), xytext=(0, 5), textcoords='offset points', 
                  ha='center', va='bottom', fontsize=10, fontweight='bold')

plt.title("Totaal pandoppervlak per corporatie", fontsize=16, pad=15)
plt.xlabel("Corporatie", fontsize=14)
plt.ylabel("Totaal oppervlak (m²)", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
