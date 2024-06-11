import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from sklearn.svm import SVR
from sklearn_rvm import EMRVR

# Generate sample data
np.random.seed(8)
rng = np.random.RandomState(0)
X = 4 * np.pi * np.random.random(100) - 2 * np.pi
y = np.sinc(X)
y += 0.25 * (0.5 - rng.rand(X.shape[0]))  # add noise

#df = pd.read_csv("ts2vector.csv")
#X = df['tv_intliteral']
#y = df['label']


X = X[:, None]
print(X.shape)
print(y.shape)

# Fit SVR
svr = SVR(kernel="rbf", gamma="auto")
svr.fit(X, y)

# Fit RVR
rvr = EMRVR(kernel="rbf")
rvr.fit(X, y)
#X_plot = np.linspace(0, 50, 10000)[:, None]
X_plot = np.linspace(-2 * np.pi, 2 * np.pi, 10000)[:, None]

# Prediction
y_svr = svr.predict(X_plot)
y_rvr = rvr.predict(X_plot, return_std=False)

# Plot results
fig = plt.figure(figsize=(10, 5))
lw = 2
fig.suptitle("RVR versus SVR", fontsize=16)

plt.subplot(121)
plt.scatter(X, y, marker=".", c="k", label="data")
plt.plot(X_plot, np.sinc(X_plot), color="navy", lw=lw, label="True")

plt.plot(X_plot, y_svr, color="turquoise", lw=lw, label="SVR")
support_vectors_idx = svr.support_
plt.scatter(X[support_vectors_idx], y[support_vectors_idx], s=80, facecolors="none", edgecolors="r",
            label="support vectors")
plt.ylabel("target")
plt.xlabel("data")
plt.legend(loc="best", scatterpoints=1, prop={"size": 8})

plt.subplot(122)
plt.scatter(X, y, marker=".", c="k", label="data")
plt.plot(X_plot, np.sinc(X_plot), color="navy", lw=lw, label="True")

plt.plot(X_plot, y_rvr, color="darkorange", lw=lw, label="RVR")
relevance_vectors_idx = rvr.relevance_
plt.scatter(X[relevance_vectors_idx], y[relevance_vectors_idx], s=80, facecolors="none", edgecolors="r",
            label="relevance vectors")

plt.xlabel("data")
plt.legend(loc="best", scatterpoints=1, prop={"size": 8})
plt.show()

print(support_vectors_idx.shape)
print(relevance_vectors_idx.shape)