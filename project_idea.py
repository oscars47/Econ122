import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.cluster import AgglomerativeClustering
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from IPython.display import SVG
from sknetwork.clustering import Louvain, get_modularity
from sknetwork.visualization import svg_graph, svg_bigraph
from scipy.spatial.distance import euclidean

# generate random data to test
def random_data(num_individuals):
    '''for each individual, generate random data: gender, education, income'''
    np.random.seed(47)  # for reproducibility

    # gender is either 1 or 0
    gender = np.random.randint(0, 2, num_individuals)
    # education is a random number between 6 and 20
    education = np.random.randint(6, 21, num_individuals)
    # income is a random number between 20 and 100
    income = np.random.randint(20, 101, num_individuals)

    # append to dataframe
    df = pd.DataFrame()
    df['gender'] = gender
    df['education'] = education
    df['income'] = income

    # plot gender, education, income
    # fig = plt.figure(figsize=(10, 7))
    # ax = fig.add_subplot(111, projection='3d')

    # ax.scatter(df['gender'], df['education'], df['income'], c='r', marker='o')
    # plt.show()


    return df

def perform_clustering(df):
    # Your existing setup, computation of similarity_matrix, and initialization of Louvain
    scaler = StandardScaler()
    df[['education', 'income']] = scaler.fit_transform(df[['education', 'income']])

    # Calculate pairwise similarities
    n = len(df)
    similarity_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                # Calculate similarity for gender (1 if same, 0 if different)
                gender_similarity = .09 if df.iloc[i]['gender'] == df.iloc[j]['gender'] else .01
                # Calculate inverse Euclidean distance for numerical attributes
                edu_income_similarity = 1 / (1 + euclidean(df.iloc[i][['education', 'income']], df.iloc[j][['education', 'income']]))

                # Combine similarities (average)
                combined_similarity = (gender_similarity + edu_income_similarity) / 2
                similarity_matrix[i, j] = combined_similarity

    print("Similarity matrix:", similarity_matrix)

    louvain = Louvain()
    labels_sparse = louvain.fit_predict(similarity_matrix)
    
    # # Convert sparse matrix to a dense array
    # labels_dense = labels_sparse.toarray().ravel()  # Flatten the dense array to 1D

    # Analyze the distribution of nodes across communities
    labels_unique, counts = np.unique(labels_sparse, return_counts=True)
    print("Community labels:", labels_unique)
    print("Counts per community:", counts)
    print("Labels dense:", labels_sparse)

    print("Modularity", get_modularity(similarity_matrix, labels_sparse))
    
    return labels_sparse

def plot_results(labels_dense, df):
    '''plots results based on Louvain community detection'''
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111, projection='3d')

    # Map each unique label to a color
    unique_labels = np.unique(labels_dense)
    colors = plt.cm.jet(np.linspace(0, 1, len(unique_labels)))

    for i, row in df.iterrows():
        label = labels_dense[i]
        color = colors[np.where(unique_labels == label)[0][0]]
        ax.scatter(row['education'], row['gender'], row['income'], color=color, s=50)

    ax.set_xlabel('Education (Normalized)')
    ax.set_ylabel('Gender')
    ax.set_zlabel('Income (Normalized)')
    plt.title('3D Plot of Individuals with Louvain Community Detection')
    plt.show()

def run_idea(num_individuals):
    # generate random data
    df = random_data(num_individuals)
    # perform clustering
    clusters = perform_clustering(df)
    # plot results
    plot_results(clusters, df)

if __name__ == "__main__":
    run_idea(100)