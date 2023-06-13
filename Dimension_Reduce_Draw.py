### 基本套件 ###
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
# %matplotlib inline

### 標準化 ###
from sklearn.preprocessing import StandardScaler

### 降維工具 ###
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP

### PCA 函式 ###
def Draw_PCA(filename):
    
    ### 資料處理 ###
    data = pd.read_csv(filename)
    data['Index'] = list(range(data.shape[0]))
    data.set_index('Index', inplace = True)

    n_columns = data.shape[1] - 1 
    feature = data.iloc[:, 1:n_columns]
    target = data.iloc[:, n_columns]
    
    filepath = filename
    basename = os.path.basename(filepath) 
    filename = os.path.splitext(basename)[0]

    ### 資料標準化 ###
    scaled_data = StandardScaler().fit(feature).transform(feature)
    
    ### PCA ###
    pca = PCA(n_components = 3).fit(scaled_data).transform(scaled_data)

    Xax = pca[:, 0]
    Yax = pca[:, 1]
    Zax = pca[:, 2]
    
    color = {0:'red', 1:'skyblue'}
    label = {0:'before', 1:'after'}
    marker = {0:'*', 1:'o'}
    alpha = {0:.3, 1:.5}

    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title(filename, fontsize = 15)

    fig.patch.set_facecolor('white')
    for i in np.unique(target):
        ix = np.where(target == i)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c = color[i], s = 40, label = label[i], marker = marker[i], alpha = alpha[i])
    
    ax.grid()
    plt.show()

    ### 互動式 PCA ###
    df = pd.DataFrame(pca, columns = ('x', 'y', 'z'))
    df["class"] = target

    pca_interactive = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'class', title = filename)
    pca_interactive.update_traces(marker_size = 5)
    pca_interactive.show()
    

def Draw_PCA_Multi(filename):
    
    ### 資料處理 ###
    data = pd.read_csv(filename)
    data['Index'] = list(range(data.shape[0]))
    data.set_index('Index', inplace = True)

    n_columns = data.shape[1] - 1 
    feature = data.iloc[:, 1:n_columns]
    target = data.iloc[:, n_columns]
    
    filepath = filename
    basename = os.path.basename(filepath) 
    filename = os.path.splitext(basename)[0]

    ### 資料標準化 ###
    scaled_data = StandardScaler().fit(feature).transform(feature)
    
    ### PCA ###
    pca = PCA(n_components = 3).fit(scaled_data).transform(scaled_data)

    Xax = pca[:, 0]
    Yax = pca[:, 1]
    Zax = pca[:, 2]
    
    color = {0:'red', 1:'skyblue', 2:'green', 3:'purple'}
    label = {0:'before', 1:'one', 2:'two', 3:'three'}
    marker = {0:'*', 1:'o', 2:'*', 3:'o'}
    alpha = {0:.3, 1:.5, 2:.3, 3:.5}

    fig = plt.figure(figsize = (12, 8))
    ax = fig.add_subplot(111, projection = '3d')
    ax.set_xlabel('Principal Component 1', fontsize = 15)
    ax.set_ylabel('Principal Component 2', fontsize = 15)
    ax.set_zlabel('Principal Component 3', fontsize = 15)
    ax.set_title(filename, fontsize = 15)

    fig.patch.set_facecolor('white')
    for i in np.unique(target):
        ix = np.where(target == i)
        ax.scatter(Xax[ix], Yax[ix], Zax[ix], c = color[i], s = 40, label = label[i], marker = marker[i], alpha = alpha[i])
    
    ax.grid()
    plt.show()

    ### 互動式 PCA ###
    df = pd.DataFrame(pca, columns = ('x', 'y', 'z'))
    df["class"] = target
    df["class"] = df["class"].replace({0:'before', 1:'one', 2:'two', 3:'three'})

    pca_interactive = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'class', title = filename)
    pca_interactive.update_traces(marker_size = 5)
    pca_interactive.show()
    
### t-SNE 函式 ###
def Draw_t_SNE(filename):
    
    ### 資料處理 ###
    data = pd.read_csv(filename)
    data['Index'] = list(range(data.shape[0]))
    data.set_index('Index', inplace = True)

    n_columns = data.shape[1] - 1 
    feature = data.iloc[:, 1:n_columns]
    target = data.iloc[:, n_columns]
    
    filepath = filename
    basename = os.path.basename(filepath) 
    filename = os.path.splitext(basename)[0]

    ### 資料標準化 ###
    scaled_data = StandardScaler().fit(feature).transform(feature)
    
    ### t_SNE ###
    t_SNE = TSNE(n_components = 2, init = 'random', random_state = 5, verbose = 1).fit_transform(scaled_data)

    x_min, x_max = t_SNE.min(0), t_SNE.max(0)
    X_norm = (t_SNE - x_min) / (x_max - x_min)

    plt.figure(figsize = (5, 5))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(target[i]), color = plt.cm.Set1(target[i]), 
                 fontdict = {'weight': 'bold', 'size': 9})
    plt.title(filename, fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    ### 互動式 t_SNE ###
    t_SNE = TSNE(n_components = 3, init = 'random', random_state = 5, verbose = 1).fit_transform(scaled_data)

    df = pd.DataFrame(t_SNE, columns = ('x', 'y', 'z'))
    df["class"] = target
    
    t_SNE_interactive = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'class', title = filename)
    t_SNE_interactive.update_traces(marker_size = 5)
    t_SNE_interactive.show()

    
def Draw_t_SNE_Multi(filename):
    
    ### 資料處理 ###
    data = pd.read_csv(filename)
    data['Index'] = list(range(data.shape[0]))
    data.set_index('Index', inplace = True)

    n_columns = data.shape[1] - 1 
    feature = data.iloc[:, 1:n_columns]
    target = data.iloc[:, n_columns]
    
    filepath = filename
    basename = os.path.basename(filepath) 
    filename = os.path.splitext(basename)[0]

    ### 資料標準化 ###
    scaled_data = StandardScaler().fit(feature).transform(feature)
    
    ### t_SNE ###
    t_SNE = TSNE(n_components = 2, init = 'random', random_state = 5, verbose = 1).fit_transform(scaled_data)

    x_min, x_max = t_SNE.min(0), t_SNE.max(0)
    X_norm = (t_SNE - x_min) / (x_max - x_min)

    plt.figure(figsize = (5, 5))
    for i in range(X_norm.shape[0]):
        plt.text(X_norm[i, 0], X_norm[i, 1], str(target[i]), color = plt.cm.Set1(target[i]), 
                 fontdict = {'weight': 'bold', 'size': 9})
    plt.title(filename, fontsize = 15)
    plt.xticks([])
    plt.yticks([])
    plt.show()

    ### 互動式 t_SNE ###
    t_SNE = TSNE(n_components = 3, init = 'random', random_state = 5, verbose = 1).fit_transform(scaled_data)

    df = pd.DataFrame(t_SNE, columns = ('x', 'y', 'z'))
    df["class"] = target
    df["class"] = df["class"].replace({0:'before', 1:'one', 2:'two', 3:'three'})
    
    t_SNE_interactive = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'class', title = filename)
    t_SNE_interactive.update_traces(marker_size = 5)
    t_SNE_interactive.show()
    

### UMAP 函式 ###
def Draw_UMAP(filename):
    
    ### 資料處理 ###
    data = pd.read_csv(filename)
    data['Index'] = list(range(data.shape[0]))
    data.set_index('Index', inplace = True)

    n_columns = data.shape[1] - 1 
    feature = data.iloc[:, 1:n_columns]
    target = data.iloc[:, n_columns]
    
    filepath = filename
    basename = os.path.basename(filepath) 
    filename = os.path.splitext(basename)[0]

    ### 資料標準化 ###
    scaled_data = StandardScaler().fit(feature).transform(feature)
    
    ### UMAP ###
    embedding = UMAP(random_state = 42).fit_transform(scaled_data)

    df = pd.DataFrame(embedding, columns = ('x', 'y'))
    df["class"] = target
    df["class"] = df["class"].replace({0:'before', 1:'after'})

    sns.set_style("whitegrid", {'axes.grid' : True})

    ax = sns.pairplot(x_vars = ["x"], y_vars = ["y"], data = df, 
                      hue = "class", size = 5, plot_kws = {"s": 20})

    ax.fig.suptitle(filename, fontsize = 15)
    plt.show()

    ### 互動式 UMAP ###
    proj_3d = UMAP(n_components = 3, init = 'random', random_state = 0).fit_transform(scaled_data)

    df = pd.DataFrame(proj_3d, columns = ('x', 'y', 'z'))
    df["class"] = target
    df["class"] = df["class"].replace({0:'before', 1:'after'})

    UMAP_interactive = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'class', title = filename)
    UMAP_interactive.update_traces(marker_size = 5)
    UMAP_interactive.show()
    
def Draw_UMAP_Multi(filename):
    
    ### 資料處理 ###
    data = pd.read_csv(filename)
    data['Index'] = list(range(data.shape[0]))
    data.set_index('Index', inplace = True)

    n_columns = data.shape[1] - 1 
    feature = data.iloc[:, 1:n_columns]
    target = data.iloc[:, n_columns]
    
    filepath = filename
    basename = os.path.basename(filepath) 
    filename = os.path.splitext(basename)[0]

    ### 資料標準化 ###
    scaled_data = StandardScaler().fit(feature).transform(feature)
    
    ### UMAP ###
    embedding = UMAP(random_state = 42).fit_transform(scaled_data)

    df = pd.DataFrame(embedding, columns = ('x', 'y'))
    df["class"] = target
    df["class"] = df["class"].replace({0:'before', 1:'one', 2:'two', 3:'three'})

    sns.set_style("whitegrid", {'axes.grid' : True})

    ax = sns.pairplot(x_vars = ["x"], y_vars = ["y"], data = df, 
                      hue = "class", size = 5, plot_kws = {"s": 20})

    ax.fig.suptitle(filename, fontsize = 15)
    plt.show()

    ### 互動式 UMAP ###
    proj_3d = UMAP(n_components = 3, init = 'random', random_state = 0).fit_transform(scaled_data)

    df = pd.DataFrame(proj_3d, columns = ('x', 'y', 'z'))
    df["class"] = target
    df["class"] = df["class"].replace({0:'before', 1:'one', 2:'two', 3:'three'})

    UMAP_interactive = px.scatter_3d(df, x = 'x', y = 'y', z = 'z', color = 'class', title = filename)
    UMAP_interactive.update_traces(marker_size = 5)
    UMAP_interactive.show()