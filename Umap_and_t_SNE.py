import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from sklearn.manifold import TSNE
from umap import UMAP
from sklearn.datasets import load_iris

def load_iris_data():
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = pd.Categorical.from_codes(iris.target, iris.target_names)
    return df

def normal_db():
    df = load_iris_data()
    sns.pairplot(df, hue='species')
    plt.show()

def tsne_2d():
    df = load_iris_data()
    features = df.drop('species', axis=1)

    tsne = TSNE(n_components=2, random_state=0)
    projections = tsne.fit_transform(features)

    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=projections[:, 0], y=projections[:, 1], hue=df.species, palette='deep')
    plt.title('t-SNE 2D projection')
    plt.show()

def tsne_3d():
    df = load_iris_data()
    features = df.drop('species', axis=1)

    tsne = TSNE(n_components=3, random_state=0)
    projections = tsne.fit_transform(features)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(projections[:, 0], projections[:, 1], projections[:, 2], c=pd.factorize(df.species)[0], cmap='viridis')
    ax.set_title('t-SNE 3D projection')
    plt.colorbar(scatter)
    plt.show()

def umap_plot():
    df = load_iris_data()
    features = df.drop('species', axis=1)

    umap_2d = UMAP(n_components=2, random_state=0)
    umap_3d = UMAP(n_components=3, random_state=0)

    proj_2d = umap_2d.fit_transform(features)
    proj_3d = umap_3d.fit_transform(features)

    # 2D UMAP
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x=proj_2d[:, 0], y=proj_2d[:, 1], hue=df.species, palette='deep')
    plt.title('UMAP 2D projection')
    plt.show()

    # 3D UMAP
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(proj_3d[:, 0], proj_3d[:, 1], proj_3d[:, 2], c=pd.factorize(df.species)[0], cmap='viridis')
    ax.set_title('UMAP 3D projection')
    plt.colorbar(scatter)
    plt.show()




#normal_db()
#tsne_2d()
#tsne_3d()
umap_plot()