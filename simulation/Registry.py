# Have a registered function for all values in the ontology

# Curry routine from internet site
# Massimiliano Tomassoli, 2012.

# Curry


def genCur(func, unique=True, minArgs=None):
    """ Generates a 'curried' version of a function. """

    def g(*myArgs, **myKwArgs):
        def f(*args, **kwArgs):
            if args or kwArgs:  # some more args!
                # Allocates data to assign to the next 'f'.
                newArgs = myArgs + args
                newKwArgs = dict.copy(myKwArgs)

                # If unique is True, we don't want repeated keyword arguments.
                if unique and not kwArgs.keys().isdisjoint(newKwArgs):
                    raise ValueError("Repeated kw arg while unique = True")

                # Adds/updates keyword arguments.
                newKwArgs.update(kwArgs)

                # Checks whether it's time to evaluate func.
                if minArgs is not None and minArgs <= len(newArgs) + len(newKwArgs):
                    return func(*newArgs, **newKwArgs)  # time to evaluate func
                else:
                    return g(*newArgs, **newKwArgs)  # returns a new 'f'
            else:  # the evaluation was forced
                return func(*myArgs, **myKwArgs)

        return f

    return g


def cur(f, minArgs=None):
    return genCur(f, True, minArgs)


def curr(f, minArgs=None):
    return genCur(f, False, minArgs)


# Simple Function.
def func(a, b, c, d, e, f, g=100):
    print(a, b, c, d, e, f, g)

# Tests
def test_clusterer_silhouette(XY):
    # "_args": [{  "type": "numpy.ndarray",  "dtype": "float32"},
    #           {"type": "numpy.ndarray", "dtype": "int32"}],
    # "_return": [{"type": "float"}]

    # we only want to test cosine metric for this example, but it could be a parameter in other cases

    from sklearn import metrics

    min_score = -1.0
    max_score = 1.0
    # normalize from zero to one, from the min and max that this test gives.

    X = XY[0]
    Y = XY[1]
    print('test_clusterer_silhouette')
    silhouette = metrics.silhouette_score(X, Y, metric='cosine')
    silhouette = (silhouette - min_score) / (max_score - min_score)
    return silhouette


def test_clusterer_calinskiHarabaz(XY):
    X = XY[0]
    Y = XY[1]

    # "_args": [{  "type": "numpy.ndarray",  "dtype": "float32"},
    #           {"type": "numpy.ndarray", "dtype": "int32"}],
    # "_return": [{"type": "float"}]

    # we only want to test cosine metric for this example, but it could be a parameter in other cases

    from sklearn import metrics

    print('test_clusterer_calinskiHarabaz')
    min_score = 0
    max_score = 200

    calinski_harabaz = metrics.calinski_harabaz_score(X, Y)

    calinski_harabaz = (calinski_harabaz - min_score) / (max_score - min_score)
    return calinski_harabaz


# NLP routines
def vectorSpace_gensim_doc2vec(X, size, iterations, minfreq):
    #   "_args": [{"type": "list","firstElement":"gensim.models.doc2vec.TaggedDocument" }],
    #   "_return": [{"type": "numpy.ndarray","dtype": "float32" }

    import gensim
    import numpy as np
    import sklearn.preprocessing
    from sklearn.preprocessing import StandardScaler

    print('vectorSpace_gensim_doc2vec')

    model = gensim.models.doc2vec.Doc2Vec(size=size, min_count=minfreq, iter=iterations, dm=0)
    model.build_vocab(X)
    model.train(X, total_examples=model.corpus_count, epochs=model.iter)
    cmtVectors = [model.infer_vector(X[i].words) for i in range(len(X))]
    cmtVectors = [inferred_vector for inferred_vector in cmtVectors
                  if not np.isnan(inferred_vector).any()
                  and not np.isinf(inferred_vector).any()]

    return StandardScaler().fit_transform(cmtVectors)


def preprocessor_freetext_tag(X):
    # convert a list of strings to a tagged document
    # if it is a list of a list of strings broadcast to a list of tagged documents

    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "list","gensim.models.doc2vec.TaggedDocument" }]

    import gensim
    print('preprocessor_freetext_tag')

    tag = lambda x, y: gensim.models.doc2vec.TaggedDocument(x, [y])

    if type(X) is str:
        tagged = tag(X, X)
    else:
        tagged = [tag(x, y) for y, x in enumerate(X)]
    return tagged


def preprocessor_freetext_lemmatization(X):
    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "list","firstElement":"list" }]

    # converts string documents into list of tokens
    # if given a list, broadcasts
    import gensim

    print('preprocessor_freetext_lemmatization')
    stopfile = 'stopwords.txt'
    lemmatized = []
    with open(stopfile, 'r') as f:
        stopwords = {word.lower().strip() for word in f.readlines()}
        lemma = lambda x: [b.decode('utf-8') for b in gensim.utils.lemmatize(str(x), stopwords=frozenset(stopwords))]

        if type(X) is str:
            lemmatized = lemma(X)
        else:
            lemmatized = [lemma(x) for x in X]

    return lemmatized


def preprocessor_freetext_strip(X):
    # strips addresses and emojis. if you get string strip, if you get list broadcast

    #   "_args": [{"type": "list","firstElement":"string" }],
    #   "_return": [{"type": "list","firstElement":"string" }]

    import re

    print("preprocessor_freetext_strip")
    code = 'utf-8'
    strip = lambda x: re.sub(r"\s?http\S*", "", x).encode(code).decode(code)

    # strip = lambda  x: re.sub(r"\s?http\S*", "", x).decode(code)
    # strip = lambda  x: re.sub(r"\s?http\S*", "", x.decode(code))
    # strip = lambda  x: re.sub(r"\s?http\S*", "", x)

    if type(X) is str:
        decoded = strip(X)
    else:
        decoded = [strip(x) for x in X]
    return decoded


def preprocessor_freetext_shuffle(X):
    #   "_args": [{"type": "list" }],
    #   "_return": [{"type": "list" }]
    import random

    print("preprocessor_freetext_shuffle")
    random.shuffle(X)
    return X

# Data
def data_freetext_csvColumn(relative_path, col='text'):
    #  returns a list of documents that are strings
    #   "_return": [{"type": "list","firstElement":"string" }]

    import pandas as pd
    import os

    # script.py
    # current_file = os.path.abspath(os.path.dirname(__file__))   # older/folder2/scripts_folder

    # csv_filename
    # csv_filename = os.path.join(current_file, relative_path)

    cwd = os.getcwd()
    path = cwd + relative_path

    print('data_freetext_csvColumn at ' + path)
    raw_data = pd.read_csv(path, encoding="ISO-8859-1")
    docList = [raw_data.loc[i, col] for i in range(len(raw_data)) if raw_data.loc[i, col]]
    return docList


def data_vector_varied(n_samples=1500):
    import sklearn

    print('data_vector_varied')
    from sklearn.datasets import make_blobs
    X, Y = make_blobs(n_samples=n_samples,
                      cluster_std=[1.0, 2.5, 0.5],
                      random_state=170)
    return X


def data_vector_ansio(n_samples=1500):
    import numpy as np
    from sklearn.datasets import make_blobs
    print('data_vector_ansio')
    X, y = make_blobs(n_samples=n_samples, random_state=170)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)

    return X_aniso


def data_vector_circles(n_samples=1500):
    from sklearn.datasets import make_circles
    print('data_vector_circles')
    X, Y = make_circles(n_samples=n_samples, factor=.5,
                        noise=.05)
    return X


def data_vector_moons(n_samples=1500):
    from sklearn.datasets import make_moons
    print('data_vector_moons')
    X, Y = make_moons(n_samples=n_samples, noise=.05)
    return X


def data_vector_blobs(n_samples=1500):
    from sklearn.datasets import make_blobs
    print('data_vector_blobs')
    X, Y = make_blobs(n_samples=n_samples, random_state=8)
    return X

# Clusterers
params = {'quantile': .3,
          'eps': .3,
          'damping': .9,
          'preference': -200,
          'n_neighbors': 10,
          'n_clusters': 20}


def clusterer_sklearn_kmeans(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn.cluster import MiniBatchKMeans

    print('clusterer_sklearn_kmeans')
    clusterAlgSKN = MiniBatchKMeans(n_clusters).fit(X)
    clusterAlgLabelAssignmentsSKN = clusterAlgSKN.predict(X)
    XY = (X, clusterAlgLabelAssignmentsSKN)
    return XY


def clusterer_sklearn_agglomerative(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter

    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    import numpy as np
    print('clusterer_sklearn_agglomerative')

    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)

    average_linkage = AgglomerativeClustering(linkage="average",
                                              affinity="cosine", n_clusters=params['n_clusters'],
                                              connectivity=connectivity).fit(X)
    clusterAlgLabelAssignmentsSAG = average_linkage.labels_.astype(np.int)

    XY = (X, clusterAlgLabelAssignmentsSAG)
    return XY


def clusterer_sklearn_affinityPropagation(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn.cluster import AffinityPropagation
    print('clusterer_sklearn_affinityPropagation')

    affinity_propagation = AffinityPropagation(damping=params['damping'], preference=params['preference']).fit(
        X)
    clusterAlgLabelAssignmentsSAP = affinity_propagation.predict(X)

    XY = (X, clusterAlgLabelAssignmentsSAP)
    return XY


def clusterer_sklearn_meanShift(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter

    import sklearn
    from sklearn.cluster import MeanShift
    print('clusterer_sklearn_meanShift')

    bandwidth = sklearn.cluster.estimate_bandwidth(X, quantile=params['quantile'])

    ms = MeanShift(bandwidth=bandwidth, bin_seeding=True).fit(X)
    clusterAlgLabelAssignmentsSM = ms.predict(X)

    XY = (X, clusterAlgLabelAssignmentsSM)
    return XY


def clusterer_sklearn_spectral(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter

    from sklearn.cluster import SpectralClustering
    import numpy as np
    print('clusterer_sklearn_spectral')

    spectral = SpectralClustering(
        n_clusters=params['n_clusters'], eigen_solver='arpack',
        affinity="cosine")
    try:
        clusterAlgLabelAssignmentsSS = None
        spectral = spectral.fit(X)
    except ValueError as e:
        pass
    else:
        clusterAlgLabelAssignmentsSS = spectral.labels_.astype(np.int)

    XY = (X, clusterAlgLabelAssignmentsSS)
    return XY


def clusterer_sklearn_ward(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn.cluster import AgglomerativeClustering
    from sklearn.neighbors import kneighbors_graph
    import numpy as np
    print('clusterer_sklearn_ward')

    connectivity = kneighbors_graph(
        X, n_neighbors=params['n_neighbors'], include_self=False)
    # make connectivity symmetric
    connectivity = 0.5 * (connectivity + connectivity.T)
    ward = AgglomerativeClustering(n_clusters=params['n_clusters'], linkage='ward',
                                   connectivity=connectivity).fit(X)
    clusterAlgLabelAssignmentsSW = ward.labels_.astype(np.int)

    XY = (X, clusterAlgLabelAssignmentsSW)
    return XY


def clusterer_sklearn_dbscan(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn.cluster import DBSCAN
    import numpy as np
    print('clusterer_sklearn_dbscan')

    dbscan = DBSCAN(eps=params['eps']).fit(X)
    clusterAlgLabelAssignmentsSD = dbscan.labels_.astype(np.int)

    XY = (X, clusterAlgLabelAssignmentsSD)
    return XY


def clusterer_sklearn_birch(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn.cluster import Birch
    print('clusterer_sklearn_birch')

    birch = Birch(n_clusters=params['n_clusters']).fit(X)
    clusterAlgLabelAssignmentsSB = birch.predict(X)

    XY = (X, clusterAlgLabelAssignmentsSB)
    return XY


def clusterer_sklearn_gaussian(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    from sklearn import mixture
    print('clusterer_sklearn_gaussian')

    clusterAlgSGN = mixture.GaussianMixture(n_components=params['n_clusters'], covariance_type='full').fit(X)
    clusterAlgLabelAssignmentsSGN = clusterAlgSGN.predict(X)

    XY = (X, clusterAlgLabelAssignmentsSGN)
    return XY


def clusterer_nltk_kmeans(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    import nltk
    import numpy as np
    from nltk.cluster.kmeans import KMeansClusterer
    print('clusterer_nltk_kmeans')

    clusterAlgLabelAssignmentsNK = None
    # X = XY[0]
    cmtVectors = X  # XY[1]
    if type(cmtVectors) is np.ndarray and len(cmtVectors) > 0:
        # dt = np.dtype(cmtVectors)
        dt = cmtVectors.dtype
        if dt.type is np.float32 or dt.type is np.float64:
            clusterAlgNK = KMeansClusterer(params['n_clusters'], distance=nltk.cluster.util.cosine_distance, repeats=25,
                                           avoid_empty_clusters=True)
            clusterAlgLabelAssignmentsNK = clusterAlgNK.cluster(cmtVectors, assign_clusters=True)

    XY = (X, clusterAlgLabelAssignmentsNK)
    return XY


def clusterer_nltk_agglomerative(X, n_clusters):
    # "_args": [{"type": "numpy.ndarray","dtype": "float32"} ],
    #   "_return": [{ "type": "numpy.ndarray","dtype": "int32"}

    # in this case we want to try different numbers of clusters, so it is a parameter
    import numpy as np
    from nltk.cluster.gaac import GAAClusterer
    print('clusterer_nltk_agglomerative')

    clusterAlgLabelAssignmentsNG = None

    # X = XY[0]
    cmtVectors = X  # XY[1]

    if type(cmtVectors) is np.ndarray and len(cmtVectors) > 0:
        # dt = np.dtype(cmtVectors)
        dt = cmtVectors.dtype
        if dt.type is np.float32 or dt.type is np.float64:
            clusterAlgNG = GAAClusterer(num_clusters=params['n_clusters'], normalise=True, svd_dimensions=None)
            clusterAlgLabelAssignmentsNG = clusterAlgNG.cluster(cmtVectors, assign_clusters=True)

    return X, clusterAlgLabelAssignmentsNG


# Fill the initial function
# In this case we dont want other programs to fill in the parameters, so we dont put original functions in the registry
curries = {'data_freetext_csvColumn': curr(data_freetext_csvColumn),
           # 'data_vector_blobs': curr(data_vector_blobs,
           'vectorSpace_gensim_doc2vec': curr(vectorSpace_gensim_doc2vec),
           'clusterer_sklearn_kmeans': curr(clusterer_sklearn_kmeans),
           'clusterer_sklearn_agglomerative': curr(clusterer_sklearn_agglomerative),
           'clusterer_sklearn_affinityPropagation': curr(clusterer_sklearn_affinityPropagation),
           'clusterer_sklearn_meanShift': curr(clusterer_sklearn_meanShift),
           'clusterer_sklearn_spectral': curr(clusterer_sklearn_spectral),
           'clusterer_sklearn_ward': curr(clusterer_sklearn_ward),
           'clusterer_sklearn_dbscan': curr(clusterer_sklearn_dbscan),
           'clusterer_sklearn_birch': curr(clusterer_sklearn_birch),
           'clusterer_sklearn_gaussian': curr(clusterer_sklearn_gaussian),
           'clusterer_nltk_agglomerative': curr(clusterer_nltk_agglomerative),
           'clusterer_nltk_kmeans': curr(clusterer_nltk_kmeans)}

# Create the constructions that would be machine learned, using a shortened dataset
registry = {'data_vector_blobs': curr(data_vector_blobs), 'data_vector_moons': curr(data_vector_moons),
            'data_vector_circles': curr(data_vector_circles), 'data_vector_ansio': curr(data_vector_ansio),
            'data_vector_varied': curr(data_vector_varied),
            'test_clusterer_silhouette': curr(test_clusterer_silhouette),
            'test_clusterer_calinskiHarabaz': curr(test_clusterer_calinskiHarabaz),
            'preprocessor_freetext_shuffle': curr(preprocessor_freetext_shuffle),
            'preprocessor_freetext_strip': curr(preprocessor_freetext_strip),
            'preprocessor_freetext_lemmatization': curr(preprocessor_freetext_lemmatization),
            'preprocessor_freetext_tag': curr(preprocessor_freetext_tag),
            'data_freetext_politicalTweets': curries['data_freetext_csvColumn']('/data/pol_tweets.csv'),
            'data_freetext_internetResearchAgencyFacebook': curries['data_freetext_csvColumn'](
                '/data/facebook.csv'),
            'data_freetext_trumpTweets': curries['data_freetext_csvColumn']('/data/trump_tweets.csv'),
            'data_freetext_politicalAds': curries['data_freetext_csvColumn']('/data/political_ads.csv'),
            'data_freetext_BSdetector': curries['data_freetext_csvColumn']('/data/fake.csv'),
            'data_freetext_internetResearchAgencyTweets': curries['data_freetext_csvColumn'](
                '/data/tweets.csv'), 'data_freetext_short': curries['data_freetext_csvColumn']('/data/short.csv'),
            'clusterer_sklearn_kmeans_5clusters': curries['clusterer_sklearn_kmeans'](n_clusters=5),
            'clusterer_sklearn_kmeans_10clusters': curries['clusterer_sklearn_kmeans'](n_clusters=10),
            'clusterer_sklearn_kmeans_20clusters': curries['clusterer_sklearn_kmeans'](n_clusters=20),
            'clusterer_sklearn_agglomerative_5clusters': curries['clusterer_sklearn_agglomerative'](n_clusters=5),
            'clusterer_sklearn_agglomerative_10clusters': curries['clusterer_sklearn_agglomerative'](n_clusters=10),
            'clusterer_sklearn_agglomerative_20clusters': curries['clusterer_sklearn_agglomerative'](n_clusters=20),
            'clusterer_sklearn_affinityPropagation_5clusters': curries['clusterer_sklearn_affinityPropagation'](
                n_clusters=5),
            'clusterer_sklearn_affinityPropagation_10clusters': curries['clusterer_sklearn_affinityPropagation'](
                n_clusters=10),
            'clusterer_sklearn_affinityPropagation_20clusters': curries['clusterer_sklearn_affinityPropagation'](
                n_clusters=20),
            'clusterer_sklearn_meanShift_5clusters': curries['clusterer_sklearn_meanShift'](n_clusters=5),
            'clusterer_sklearn_meanShift_10clusters': curries['clusterer_sklearn_meanShift'](n_clusters=10),
            'clusterer_sklearn_meanShift_20clusters': curries['clusterer_sklearn_meanShift'](n_clusters=20),
            'clusterer_sklearn_spectral_5clusters': curries['clusterer_sklearn_spectral'](n_clusters=5),
            'clusterer_sklearn_spectral_10clusters': curries['clusterer_sklearn_spectral'](n_clusters=10),
            'clusterer_sklearn_spectral_20clusters': curries['clusterer_sklearn_spectral'](n_clusters=20),
            'clusterer_sklearn_ward_5clusters': curries['clusterer_sklearn_ward'](n_clusters=5),
            'clusterer_sklearn_ward_10clusters': curries['clusterer_sklearn_ward'](n_clusters=10),
            'clusterer_sklearn_ward_20clusters': curries['clusterer_sklearn_ward'](n_clusters=20),
            'clusterer_sklearn_dbscan_5clusters': curries['clusterer_sklearn_dbscan'](n_clusters=5),
            'clusterer_sklearn_dbscan_10clusters': curries['clusterer_sklearn_dbscan'](n_clusters=10),
            'clusterer_sklearn_dbscan_20clusters': curries['clusterer_sklearn_dbscan'](n_clusters=20),
            'clusterer_sklearn_birch_5clusters': curries['clusterer_sklearn_birch'](n_clusters=5),
            'clusterer_sklearn_birch_10clusters': curries['clusterer_sklearn_birch'](n_clusters=10),
            'clusterer_sklearn_birch_20clusters': curries['clusterer_sklearn_birch'](n_clusters=20),
            'clusterer_sklearn_gaussian_5clusters': curries['clusterer_sklearn_gaussian'](n_clusters=5),
            'clusterer_sklearn_gaussian_10clusters': curries['clusterer_sklearn_gaussian'](n_clusters=10),
            'clusterer_sklearn_gaussian_20clusters': curries['clusterer_sklearn_gaussian'](n_clusters=20),
            'clusterer_nltk_kmeans_5clusters': curries['clusterer_nltk_kmeans'](n_clusters=5),
            'clusterer_nltk_kmeans_10clusters': curries['clusterer_nltk_kmeans'](n_clusters=10),
            'clusterer_nltk_kmeans_20clusters': curries['clusterer_nltk_kmeans'](n_clusters=20),
            'clusterer_nltk_agglomerative_5clusters': curries['clusterer_nltk_agglomerative'](n_clusters=5),
            'clusterer_nltk_agglomerative_10clusters': curries['clusterer_nltk_agglomerative'](n_clusters=10),
            'clusterer_nltk_agglomerative_20clusters': curries['clusterer_nltk_agglomerative'](n_clusters=20),
            'vectorSpace_gensim_doc2vec_50size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_50size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_50size_200iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=200)(minfreq=2),
            'vectorSpace_gensim_doc2vec_50size_200iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=200)(minfreq=5),
            'vectorSpace_gensim_doc2vec_50size_1000iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=1000)(minfreq=2),
            'vectorSpace_gensim_doc2vec_50size_1000iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=50)(
                iterations=1000)(minfreq=5),
            'vectorSpace_gensim_doc2vec_100size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=100)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_100size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=100)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_100size_200iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=200)(minfreq=2),
            'vectorSpace_gensim_doc2vec_100size_200iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=200)(minfreq=5),
            'vectorSpace_gensim_doc2vec_100size_1000iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=1000)(minfreq=2),
            'vectorSpace_gensim_doc2vec_100size_1000iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=100)(iterations=1000)(minfreq=5),
            'vectorSpace_gensim_doc2vec_200size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=200)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_200size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=200)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_200size_200iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=200)(minfreq=2),
            'vectorSpace_gensim_doc2vec_200size_200iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=200)(minfreq=5),
            'vectorSpace_gensim_doc2vec_200size_1000iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=1000)(minfreq=2),
            'vectorSpace_gensim_doc2vec_200size_1000iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=200)(iterations=1000)(minfreq=5),
            'vectorSpace_gensim_doc2vec_300size_20iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](size=300)(
                iterations=20)(minfreq=2),
            'vectorSpace_gensim_doc2vec_300size_20iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](size=300)(
                iterations=20)(minfreq=5),
            'vectorSpace_gensim_doc2vec_300size_200iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=200)(minfreq=2),
            'vectorSpace_gensim_doc2vec_300size_200iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=200)(minfreq=5),
            'vectorSpace_gensim_doc2vec_300size_1000iterations_2minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=1000)(minfreq=2),
            'vectorSpace_gensim_doc2vec_300size_1000iterations_5minFreq': curries['vectorSpace_gensim_doc2vec'](
                size=300)(iterations=1000)(minfreq=5)}

