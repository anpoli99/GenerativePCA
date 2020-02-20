import numpy as np
from sklearn.decomposition import PCA

"""
sample principle components from saved pre-encodings
"""

denseRep = np.load("denseArrHd.npy") 
norm_dense_rep = denseRep-np.mean(denseRep, axis = 0)

pca = PCA(n_components=norm_dense_rep.shape[1])
pca.fit(norm_dense_rep)
values = np.sqrt(pca.explained_variance_)
vectors = pca.components_
np.save("eigenvalueshd.npy",values)
np.save("eigenvectorshd.npy",vectors)
