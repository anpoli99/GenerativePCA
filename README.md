# GenerativePCA

Using Principle Component Analysis, we can determine the eigenvectors and eigenvalues of a dataset, and by randomly sampling these eigenvectors, we generate images that are normal along the linearly independent axes. These generated images share characteristics with the original dataset. However, the dataset is too large to do Principle Component Analysis on independently, so we use an autoencoder to condense each of the images. This project is an implementation of that using the TensorFlow and Scikit-learn APIs. 

## Results

The implementation with two datasets: an anime face dataset and a human face dataset. Although the generated images tend to be "loud" (with a lot of variation in color), the main features defining a face are still clearly visible, including eyes, hair and a nose. 
![Faces generated](/results/results5.jpg)
![Faces generated](/results/results7.jpg)

In addition, the components learned often strongly correlate to a highly human-readable feature. Below, the right shows the original face, and the left 4 images show the result of modifying one principle component in the encoded state. 
![Modify one prinicple component](/results/results8.jpg)

From the anime face dataset, one highly readable feature was hair-color, which just by using four principle components, is almost completely customizable.
![](/results/results3.jpg)
