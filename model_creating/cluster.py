import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.cluster.hierarchy import linkage
from scipy.cluster.hierarchy import fcluster


def group_neighborhoods(neighborhood_df):
    # Get list of neighborhood names
    labels = neighborhood_df['name'].drop_duplicates().to_list()
    words = {}
    for label in labels:
        ls = neighborhood_df[neighborhood_df['name'] == label].description.to_list()
        string = ' '.join(ls)
        words[label] = string
    return pd.DataFrame.from_dict(words, orient='index')[0]


def create_similarity_matrix(grouped_neighborhoods):
    # First we create tfidf vectors to find similarities in neighborhood keywords that are not in otherneighborhoods
    tv = TfidfVectorizer(min_df=0., max_df=1., use_idf=True)
    tv_matrix = tv.fit_transform(grouped_neighborhoods)
    tv_matrix = tv_matrix.toarray()

    # now we create a similarity matrix using the tfidf vectors
    similarity_matrix = cosine_similarity(tv_matrix)
    return pd.DataFrame(similarity_matrix)


def make_cluster_labels(similarity_matrix, max_dist):
    z = linkage(similarity_matrix, 'ward')
    cluster_labels = fcluster(z, max_dist, criterion='distance')
    cluster_labels = pd.DataFrame(cluster_labels, columns=['ClusterLabel'])
    return cluster_labels
