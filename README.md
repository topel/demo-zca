# Comparison between PCA and ZCA whitening on a dummy set of 2-d points

test_zca.py generates the following figure:

![PCA ZCA Image](https://github.com/topel/demo-zca/blob/master/comparison_pca_zca.png)

- Top: original 2-d points
- Middle: after PCA whitening
- Bottom: after ZCA whitening

Points with x1 > 5 were colored in red to see the impact of PCA and ZCA on them. One can see that ZCA does not change the data point orientation contrary to PCA. 
