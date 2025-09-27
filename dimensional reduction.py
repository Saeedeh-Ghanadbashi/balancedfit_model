# Original Data: You have 100 measurements for each person: height, weight, shoe size, salary, IQ, favorite number, etc. This is messy and has too many details (high dimensionality).

# Finding Variance (The "Differences"): PCA looks at all this data and asks: "What is the single most important factor that makes people different from each other?"

# It might find that "Overall Body Size" is the biggest source of difference. This is PC1. It's a combination of height, weight, and shoe size. If you only knew this one fact about a person, you could already tell them apart from others pretty well. PC1 captures the most variance—it explains the biggest spread in the data.

# Capturing the Next Biggest Difference: After accounting for body size, PCA looks for the next most important difference, which must be unrelated to the first.

# It might find that "Wealth vs. Intelligence" is the next biggest differentiator. This is PC2. It might be a combination of salary (positive) and IQ (negative). This direction captures the second highest amount of variance.

import pandas as pd
from sklearn.datasets import load_digits

ds = load_digits()
df = pd.DataFrame(ds.data, columns=ds.feature_names)
df.head(3)
df['target'] = ds.target


ds.data[0].reshape(8,8)

from matplotlib import pyplot as plt
plt.gray()
plt.matshow(ds.data[0].reshape(8,8))

X=df

y = ds.target

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

from sklearn.model_selection import train_test_split

train_x, test_x, train_y, test_y = train_test_split(X_scaled, y, test_size = 0.3, random_state=42, stratify=y)

from sklearn.linear_model import LogisticRegression

lr_model = LogisticRegression()

lr_model.fit(train_x, train_y)

lr_model.score(test_x, test_y)

from sklearn.decomposition import PCA

pca = PCA(0.95)
X_pca = pca.fit_transform(X)

train_x, test_x, train_y, test_y = train_test_split(X_pca, y, test_size = 0.3, random_state = 32)

lr_model = LogisticRegression(max_iter=1000)

lr_model.fit(train_x, train_y)
lr_model.score(test_x, test_y)
