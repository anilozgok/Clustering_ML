import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# creating and reading dataset
dataset = pd.read_csv("Live.csv")
# print(dataset.head())
# print(dataset.tail())

silhouettes = []
ks = list(range(2, 12))

# finding best n_cluster value with sill_coeff
for n in ks:
    kmeans = KMeans(n_clusters=n).fit(dataset)
    label = kmeans.labels_
    sill_coeffs = silhouette_score(dataset, label, metric="euclidean")
    # print("For n={}, The Silhouette Coefficient is {}".format(n,sill_coeffs))
    silhouettes.append(sill_coeffs)

# getting n_cluster
n_cluster = silhouettes.index(max(silhouettes)) + 2
# print(n_cluster)

# dividing model to n_cluster cluster group
model = KMeans(n_clusters=n_cluster)
model.fit(dataset)
labels = model.predict(dataset)

# displaying groups of cluster
# np.unique(labels, return_counts=True)

# adding labels to dataset
dataset["labels"] = labels

# finding average comments for each cluster group
group_zero = dataset[dataset["labels"] == 0]["num_comments"].mean()
group_one = dataset[dataset["labels"] == 1]["num_comments"].mean()
group_two = dataset[dataset["labels"] == 2]["num_comments"].mean()
group_three = dataset[dataset["labels"] == 3]["num_comments"].mean()

# finding average shares for each cluster group
group_zero = dataset[dataset["labels"] == 0]["num_shares"].mean()
group_one = dataset[dataset["labels"] == 1]["num_shares"].mean()
group_two = dataset[dataset["labels"] == 2]["num_shares"].mean()
group_three = dataset[dataset["labels"] == 3]["num_shares"].mean()


# final results
status_type = dataset[["status_type_photo", "status_type_video", "status_type_status"]].idxmax(axis=1)
dataset = pd.concat([dataset["labels"], status_type.rename("status_type")], axis=1)

print(dataset.groupby(["labels", "status_type"])["status_type"].count())


