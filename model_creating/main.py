import data_wrangling
import cluster
import neighborhood_model
import pandas as pd


# read data from csv and create dataframe and clean data
df = data_wrangling.read_listings('listings.csv')
data_wrangling.data_cleaning(df)

# group all neighborhood descriptions by neighborhood into one entry for clusteri
cleaned_df = pd.read_csv('filtered_neighborhood.csv')
neighborhood_groups = cluster.group_neighborhoods(cleaned_df)
similarity_matrix = cluster.create_similarity_matrix(neighborhood_groups)

# create cluster labels for each neighborhood
max_dist = 1.4
cluster_labels = cluster.make_cluster_labels(similarity_matrix, max_dist)
clustered_df = pd.concat([neighborhood_groups.reset_index(), cluster_labels], axis=1)
# use new cluster df to assign cluster label in original clean neighborhood dataframe
neighborhood_df = cleaned_df.join(clustered_df.set_index('index').drop(0, axis=1), on='name')

# create new df and save to file
neighborhood_df.to_csv('neighborhood_df.csv', index=False)
neighborhood_model.create_model(neighborhood_df)
