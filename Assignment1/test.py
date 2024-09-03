import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px

# Load the data from your file
data = np.loadtxt('ellipsoids.txt')  # Make sure to replace with your actual file path

# Perform PCA for 2D projection
pca_2d = PCA(n_components=2)
projected_2d = pca_2d.fit_transform(data)

# Perform PCA for 1D projection
pca_1d = PCA(n_components=1)
projected_1d = pca_1d.fit_transform(data)

# Prepare data for Plotly
df_2d = pd.DataFrame(projected_2d, columns=['PC1', 'PC2'])
df_1d = pd.DataFrame(projected_1d, columns=['PC1'])

# Create an interactive 2D plot using Plotly
fig_2d = px.scatter(df_2d, x='PC1', y='PC2', title='2D PCA Projection')
fig_2d.show()

# Create an interactive 1D plot using Plotly
fig_1d = px.scatter(df_1d, x='PC1', y=[0]*len(df_1d), title='1D PCA Projection')
fig_1d.update_yaxes(visible=False)
fig_1d.show()