# Comparison between PCA and ZCA whitening on a dummy set of 2-d points

test_zca.py generates the following figure:

![PCA ZCA Image](https://github.com/topel/demo-zca/blob/master/comparison_pca_zca.png)

- Top: original 2-d points
- Middle: after PCA whitening
- Bottom: after ZCA whitening

Points with x1 > 5 were colored in red to see the impact of PCA and ZCA on them. One can see that ZCA does not change the data point orientation contrary to PCA. 

# ZCA applied on a 10 second speech file (13 MFCCs + delta + delta-delta)

- Top: original 39-d MFCCs on 10 seconds of speech
- Bottom: these MFCCs after ZCA whitening (eps=0.01)

The dynamic range has been set to about [-3.0, +3.0] for all the feature dimensions.

![ZCA speech](https://github.com/topel/demo-zca/blob/master/mfc_zca_french_10s_643.png)

# References
- <http://ufldl.stanford.edu/wiki/index.php/Whitening>

- A. Coates, A.Y. Ng, "Learning Feature Representations with K-means" <https://www-cs.stanford.edu/~acoates/papers/coatesng_nntot2012.pdf>

