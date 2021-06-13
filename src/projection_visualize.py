from utils import load, load_weight_matrix
from sklearn.decomposition import PCA
from collections import Counter
import os
import matplotlib.pyplot as plt
from utils import set_random_seed, Condition_Setter


PATH_TO_DIR = os.getcwd().replace("Hol-CCG/src", "")
condition = Condition_Setter(PATH_TO_DIR)

set_random_seed(0)

print("loading_weight_matrix...")
initial_weight_matrix = load_weight_matrix(
    condition.path_to_pretrained_weight_matrix)
trained_weight_matrix = load_weight_matrix(
    PATH_TO_DIR + "Hol-CCG/result/data/weight_matrix_with_projection_learning.csv")

pca = PCA()
print("PCA working...")
fig = plt.figure(figsize=(20, 10))
ax1 = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
embedded = pca.fit_transform(initial_weight_matrix)
ax1.scatter(embedded[0], embedded[1])
embedded = pca.fit_transform(trained_weight_matrix)
ax2.scatter(embedded[0], embedded[1])
plt.show()
