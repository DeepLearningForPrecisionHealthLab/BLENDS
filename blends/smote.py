'''
Re-implementation of fMRI augmentation method developed by 
Taban Eslami, Fahad Saeed, Vahid Mirjalili, Alvis Fong and Angela Laird (2019) 
ASD-DiagNet: A hybrid learning approach for detection of Autism Spectrum Disorder 
using fMRI data, Frontiers in Neuroinformatics, 13 (2019): 70.

which combines the SMOTE linear interpolation method with the EROS (extended 
Frobenius Norm) similarity metric for multivarate timeseries. 

Based on the authors' code at https://github.com/pcdslab/ASD-DiagNet
'''

import numpy as np
import pandas as pd
import tqdm

def compute_eigs(dictTimeseries):
    """Compute eigenvalues, eigenvectors, and normed eigenvalues for multivariate timeseries

    Args:
        dictTimeseries (dict): dict of multivariate timeseries (timepoints x regions). Keys are sample IDs

    Returns:
        dictEigData (dict): dict of dicts containing eigenvalues, normed eigenvalues, and eigenvectors
        arrNormWeights (array): array of size (eigenvalues) containing weights for EROS
    """    
    dictEigData = {}
    tupDimensions = (0, 0)
    progbar = tqdm.tqdm(total=len(dictTimeseries))
    for i, (strID, arrTimeseries) in enumerate(dictTimeseries.items()):
        # Check that all timeseries have the same dimensions
        if i == 0:
            tupDimensions = arrTimeseries.shape
        else:
            if tupDimensions != arrTimeseries.shape:
                raise ValueError(f'Timeseries {i} has dimensions {arrTimeseries.shape} which does not match {tupDimensions}')
        arrCorr = np.nan_to_num(np.corrcoef(arrTimeseries.T))
        arrEigVals, arrEigVecs = np.linalg.eig(arrCorr)

        # Check that each eigenvector has unit norm
        for arrEV in arrEigVecs.T:
            np.testing.assert_array_almost_equal(1.0, np.linalg.norm(arrEV))

        fSumEigVals = np.sum(np.abs(arrEigVals))
        nEigVals = len(arrEigVals)
        # Create list of (eigenvalue, eigenvector, normed eigenvalue) tuples
        lsEigTuples = [(np.abs(arrEigVals[k]), arrEigVecs[:, k], np.abs(arrEigVals[k]) / fSumEigVals) for k in range(nEigVals)]
        # Sort from highest to lowest eigenvalue
        lsEigTuples.sort(key=lambda x: x[0], reverse=True)

        dictEigData[strID] = {'eigvals': np.array([t[0] for t in lsEigTuples]),
                              'norm-eigvals': np.array([t[2] for t in lsEigTuples]),
                              'eigvecs': np.array([t[1] for t in lsEigTuples])}
        progbar.update()                       
    progbar.close()
    # Compute the weight vector by summing up the normed eigenvalues for all
    # samples. The paper uses the mean over the samples but the authors' code
    # uses the sum. Let's go with the sum for now.
    ##TODO: compare sum with mean for aggregating eigenvalues into weight vector
    arrNormWeights = np.zeros((nEigVals))
    for dictEig in dictEigData.values():
        arrNormWeights += dictEig['norm-eigvals']

    return dictEigData, arrNormWeights

def eros_similarity(arr1, arr2, arrWeights, nEigs=None):
    """Computes EROS similarity between two multivariate timeseries

    Args:
        arr1 (array): sample 1
        arr2 (array): sample 2
        arrWeights (array): weights based on normalized eigenvalues, computed by compute_eigs
        nEigs (int, optional): Number of eigenvalues to use. Defaults to None to use all eigenvalues.
    Returns:
        res (float): EROS similarity
    """    

    res = 0.0
    if nEigs is None:
        arrWeights2 = arrWeights.copy()
    else:
        arrWeights2 = arrWeights[:nEigs].copy()
        arrWeights2 /= np.sum(arrWeights2)
    for i, weight in enumerate(arrWeights2):
        res += weight * np.inner(arr1[i], arr2[i])
    return res

class SmoteEros:
    def __init__(self, data, timeseries, labels, regression=False, n_neighbors=5):    
        self.data = data
        self.timeseries = timeseries
        self.labels = labels
        self.regression = regression
        self.n_neighbors = n_neighbors

        if ~regression:
            self.classes = np.unique(labels)
        else:
            self.classes = None
        
        nSamplesOrig = len(data)
        self.neighbors = {}
        print('Computing eigen decomposition', flush=True)
        dictEig, arrWeights = compute_eigs(timeseries)

        print(f'Selecting {n_neighbors} neighbors for each sample', flush=True)
        progbar = tqdm.tqdm(total=nSamplesOrig)
        for id in labels.index:
            label = self.labels.loc[id]
            candidates = self.labels.loc[self.labels == label].index
            if candidates.nlevels > 1:
                # If using BLENDS-augmented data with MultiIndex levels (sub_id,
                # aug_number), ensure that no augmented samples from the same
                # original sample are in the neighbor candidates
                candidates = candidates[candidates.get_level_values(0) != id[0]]    
            else:
                candidates = candidates[candidates != id]
            lsCandidates = candidates.tolist()
            lsCandidates = set(lsCandidates)

            arrEigVecs = dictEig[id]['eigvecs']
            lsSimilarity = []
            for strCand in lsCandidates:
                arrEigVecsCand = dictEig[strCand]['eigvecs']
                sim = eros_similarity(arrEigVecs, arrEigVecsCand, arrWeights)
                lsSimilarity += [(sim, strCand)]
            # Sort from highest to lowest similarity
            lsSimilarity.sort(key=lambda x: x[0], reverse=True)
            self.neighbors[id] = [t[1] for t in lsSimilarity[:self.n_neighbors]]
            progbar.update()
        progbar.close()
    
    def get_aug_data(self, factor=1):
        if factor == 1:
            return self.data, self.labels
        else:
            if self.data.index.nlevels == 1:
                indexAug = pd.MultiIndex.from_product((self.data.index, range(factor)), names=['ID', 'SMOTE'])
            else:
                lsIndexAug = []
                for tupOrig in self.data.index:
                    lsIndexAug += [tupOrig + (a,) for a in range(factor)]
                indexAug = pd.MultiIndex.from_tuples(lsIndexAug, names=['ID', 'BLENDS', 'SMOTE'])
            dataAug = pd.DataFrame(columns=self.data.columns, index=indexAug)
            labelAug = pd.Series(index=indexAug)
            progbar = tqdm.tqdm(total=self.data.shape[0])
            for strID in self.data.index:
                arrDataOrig = self.data.loc[strID].values
                labelOrig = self.labels.loc[strID]

                for iAug in range(factor):
                    # np.random.choice doesn't like lists of tuples, so need to
                    # indirectly choose a neighbor by selecting the index
                    lsNeighbors = self.neighbors[strID]
                    strIDNeighbor = lsNeighbors[np.random.choice(range(len(lsNeighbors)))]
                    arrDataNeighbor = self.data.loc[strIDNeighbor]
                    labelNeighbor = self.labels.loc[strIDNeighbor]

                    assert labelOrig == labelNeighbor
                    # Select random interpolation distance
                    r = np.random.uniform(low=0, high=1)
                    arrDataAug = r * arrDataOrig + (1 - r) * arrDataNeighbor
                    if self.data.index.nlevels == 1:
                        dataAug.loc[(strID,) + (iAug,), :] = arrDataAug
                        labelAug.loc[(strID,) + (iAug,), :] = labelOrig
                    else: 
                        dataAug.loc[strID + (iAug,), :] = arrDataAug
                        labelAug.loc[strID + (iAug,), :] = labelOrig
                progbar.update()
            progbar.close()
            return dataAug, labelAug