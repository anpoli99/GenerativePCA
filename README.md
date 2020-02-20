# GenerativePCA

Using Principle Component Analysis, we can determine the eigenvectors and eigenvalues of a dataset, and by randomly sampling these eigenvectors, we generate images that are normal along the linearly independent axes. These generated images share characteristics with the original dataset. However, the dataset is too large to do Principle Component Analysis on independently, so we use an autoencoder to condense each of the images. This project is an implementation of that using the TensorFlow and Scikit-learn APIs. 

## Results

The implementation with two datasets: an anime face dataset and a human face dataset. 
