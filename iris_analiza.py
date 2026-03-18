import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

iris=sns.load_dataset('iris')
sns.set_theme(style='darkgrid', palette='pastel')

fig=plt.figure(figsize=(18,12))
fig.suptitle('Iris Datset- Kompletna Analiza',fontsize=24, fontweight='bold', y=1.02)

ax1=fig.add_subplot(2,3,1)
sns.histplot(data=iris, x='petal_length', hue='species', kde=True, ax=ax1)
ax1.set_title('Distribucija duzine latica')
ax1.set_xlabel('Duzina latice(cm)')
ax1.set_ylabel('Broj cvetova')

ax2=fig.add_subplot(2,3,2)
sns.kdeplot(data=iris, x='sepal_length',hue='species', fill=True, alpha=0.5, ax=ax2)
ax2.set_title('KDE - Duzina casice')
ax2.set_xlabel('Duzina casice (cm)')
ax2.set_ylabel('Gustina')

ax3 = fig.add_subplot(2, 3, 3)
sns.boxplot(data=iris, x='species', y='petal_length', palette='Set2', ax=ax3)
ax3.set_title('Boxplot - Duzina latica po vrsti')
ax3.set_xlabel('Vrsta')
ax3.set_ylabel('Duzina latice (cm)')

ax4 = fig.add_subplot(2, 3, 4)
sns.violinplot(data=iris, x='species', y='sepal_width', palette='Set2', ax=ax4)
ax4.set_title('Violinplot - Sirina casice po vrsti')
ax4.set_xlabel('Vrsta')
ax4.set_ylabel('Sirina casice (cm)')

ax5 = fig.add_subplot(2, 3, 5)
sns.scatterplot(data=iris, x='petal_length', y='petal_width', hue='species', style='species', s=80, ax=ax5)
ax5.set_title('Duzina vs Sirina latice')
ax5.set_xlabel('Duzina latice (cm)')
ax5.set_ylabel('Sirina latice (cm)')

ax6 = fig.add_subplot(2, 3, 6)
corr = iris.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', linewidths=0.5, mask=mask, ax=ax6, vmin=-1, vmax=1)
ax6.set_title('Korelaciona matrica')

plt.tight_layout(rect=[0, 0, 1, 0.96], h_pad=4.0, w_pad=3.0)
plt.savefig('iris_analiza.png', dpi=200, bbox_inches='tight')
plt.show()
print('Grafik sacuvan kao iris_analiza.png')