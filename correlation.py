import pandas as pd
#df = pd.read_csv("sc2vector.csv")
#print(df.shape)
df = pd.read_csv("ts2vector.csv")
print(df.shape)
print(df.head(3))
#df = pd.read_csv("sc2tfidf.csv")
#print(df.shape)
#df = pd.read_csv("ts2tfidf.csv")
#print(df.shape)

import seaborn as sns
import matplotlib.pyplot as plt

#df = df.iloc[:, 0:11]
cp = df.corr()
#plt.figure(figsize=(12,8))
#sns.heatmap(cp, cmap="RdBu_r",annot=True)

#plt.title('Correlation between Numeric Variables')
#plt.show()

import numpy as np

nmax = -np.inf  # Dimulai dengan nilai terendah yang mungkin
nmin= np.inf  # Dimulai dengan nilai tertinggi yang mungkin
maxim = set()
minim = set()
rmax = 0
rmin = 0
for i, row in enumerate(cp.index):
    col = "label"
    if (row!=col):
        nilai_korelasi = cp.loc[row, col]
        if nilai_korelasi > nmax:
            nmax = nilai_korelasi
            maxim = {(row, col)}
            rmax = row
        elif nilai_korelasi == nmax:
            maxim.add((row, col))
    
        if nilai_korelasi < nmin:
            nmin = nilai_korelasi
            minim = {(row, col)}
            rmin = row
        elif nilai_korelasi == nmin:
            minim.add((row, col))

# Cetak hasil
print(f"Nilai korelasi tertinggi: {nmax}, Pasangan: {maxim}")
print(f"Nilai korelasi terendah: {nmin}, Pasangan: {minim}")
#print(df["tv_intliteral"])
