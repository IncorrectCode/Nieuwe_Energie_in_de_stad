import pandas as pd
import matplotlib.pyplot as plt

# Excel-bestand inlezen
file_path = "databim ruwe data.xlsx"
corporaties_df = pd.read_excel(file_path)

plt.figure(figsize=(8, 5))
plt.hist(corporaties_df["shape__area"], bins=30, color='lightgreen', edgecolor='black')
plt.yscale('log')  # log-schaal
plt.title("Verdeling van pandoppervlakte (log-schaal)", fontsize=14, pad=10)
plt.xlabel("Oppervlakte (m²)", fontsize=12)
plt.ylabel("Aantal panden", fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
