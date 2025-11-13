import pandas as pd
import matplotlib.pyplot as plt

# Excel-bestand inlezen
file_path = "databim ruwe data.xlsx"
corporaties_df = pd.read_excel(file_path)

# Aantal panden per corporatie tellen
corporatie_counts = corporaties_df["corporatie"].value_counts()

# Grafiek maken
plt.figure(figsize=(12, 6))  # Grotere figuur
bars = corporatie_counts.plot(kind='bar', color='skyblue', edgecolor='black')

# Waarden boven de balken tonen
for bar in bars.patches:
    height = bar.get_height()
    bars.annotate(f'{height}', 
                  xy=(bar.get_x() + bar.get_width() / 2, height),
                  xytext=(0, 5),  # kleine offset boven de balk
                  textcoords='offset points',
                  ha='center', va='bottom', fontsize=10)

plt.title("Aantal panden per corporatie", fontsize=16, pad=15)
plt.xlabel("Corporatie", fontsize=14)
plt.ylabel("Aantal panden", fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()