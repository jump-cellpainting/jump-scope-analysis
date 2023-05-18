#adapted with appreciation from https://github.com/jump-cellpainting/pilot-cpjump1-analysis

from multiprocessing.sharedctypes import Value
import os
import random
import textwrap

import pandas as pd
import numpy as np
import scipy
from sklearn.utils.validation import check_is_fitted
from sklearn.utils import check_array, as_float_array
from sklearn.base import TransformerMixin, BaseEstimator
import kneed
import matplotlib.pyplot as plt
import seaborn
import ast

random.seed(9000)

def get_metacols(df):
    """return a list of metadata columns"""
    return [c for c in df.columns if c.startswith("Metadata_")]

def get_featurecols(df):
    """returna  list of featuredata columns"""
    return [c for c in df.columns if not c.startswith("Metadata")]

def get_metadata(df):
    """return dataframe of just metadata columns"""
    return df[get_metacols(df)]

def get_featuredata(df):
    """return dataframe of just featuredata columns"""
    return df[get_featurecols(df)]

def remove_negcon_empty_wells(df):
    """return dataframe of non-negative-control wells"""
    df = (
        df.query('Metadata_control_type!="negcon"')
        .dropna(subset=['Metadata_broad_sample'])
        .reset_index(drop=True)
    )
    return df

def percent_score(null_dist, corr_dist, how='right'):
    """
    Calculates the Percent replicating
    :param null_dist: Null distribution
    :param corr_dist: Correlation distribution
    :param how: "left", "right" or "both" for using the 5th percentile, 95th percentile or both thresholds
    :return: proportion of correlation distribution beyond the threshold
    """
    if how == 'right':
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        return 100 * np.mean(above_threshold.astype(float)), perc_95
    if how == 'left':
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return 100 * np.mean(below_threshold.astype(float)), perc_5
    if how == 'both':
        perc_95 = np.nanpercentile(null_dist, 95)
        above_threshold = corr_dist > perc_95
        perc_5 = np.nanpercentile(null_dist, 5)
        below_threshold = corr_dist < perc_5
        return 100 * np.mean(above_threshold.astype(float)) + np.mean(below_threshold.astype(float)), perc_95, perc_5
    
def corr_between_replicates(df, group_by_feature):
    """
    Correlation between replicates
    :param df: pd.DataFrame
    :param group_by_feature: Feature name to group the data frame by
    :return: list-like of correlation values
    """
    replicate_corr = []
    replicate_grouped = df.groupby(group_by_feature)
    for name, group in replicate_grouped:
        group_features = get_featuredata(group)
        corr = np.corrcoef(group_features)
        if len(group_features) == 1:  # If there is only one replicate on a plate
            replicate_corr.append(np.nan)
        else:
            np.fill_diagonal(corr, np.nan)
            replicate_corr.append(np.nanmedian(corr))  # median replicate correlation
    return replicate_corr

def corr_between_non_replicates(df, n_samples, n_replicates, metadata_compound_name):
    """
    Null distribution between random "replicates".
    :param df: pandas.DataFrame
    :param n_samples: int
    :param n_replicates: int
    :param metadata_compound_name: Compound name feature
    :return: list-like of correlation values, with a  length of `n_samples`
    """
    df.reset_index(drop=True, inplace=True)
    null_corr = []
    while len(null_corr) < n_samples:
        compounds = random.choices([_ for _ in range(len(df))], k=n_replicates)
        sample = df.loc[compounds].copy()
        if len(sample[metadata_compound_name].unique()) == n_replicates:
            sample_features = get_featuredata(sample)
            corr = np.corrcoef(sample_features)
            np.fill_diagonal(corr, np.nan)
            null_corr.append(np.nanmedian(corr))  # median replicate correlation
    return null_corr

def corr_between_replicates_across_plates(df, reference_df, pertcol = 'Metadata_pert_iname'):
    items = list(df[pertcol].unique())
    common_columns = [x for x in df.columns if x in reference_df.columns]
    df = df[common_columns]
    reference_df = reference_df[common_columns]
    replicate_corr = []
    for item in items:
        compound_df = df.query(pertcol+' == @item')
        compound_reference_df = reference_df.query(pertcol+' == @item')

        compound_df_profiles = get_featuredata(compound_df).values
        compound_reference_df_profiles = get_featuredata(compound_reference_df).values
        try:
            corr = np.corrcoef(compound_df_profiles, compound_reference_df_profiles)
            corr = corr[0:len(compound_df_profiles), len(compound_df_profiles):]

            corr_median_value = np.nanmedian(corr, axis=1)
            corr_median_value = np.nanmedian(corr_median_value)

            replicate_corr.append(corr_median_value)
        except:
            print(item,corr)
    return replicate_corr

def corr_between_non_replicates_across_plates(df, reference_df, n_samples, pertcol = 'Metadata_pert_iname'):
    np.random.seed(9000)
    common_columns = [x for x in df.columns if x in reference_df.columns]
    df = df[common_columns]
    reference_df = reference_df[common_columns]
    null_corr = []
    compounds = list(df[pertcol].unique())  
    while len(null_corr) < n_samples:
        both_compounds = np.random.choice(compounds, size=2, replace=False)
        compound1 = both_compounds[0]
        compound2 = both_compounds[1]

        compound1_df = df.query(pertcol+' == @compound1')
        compound2_df = reference_df.query(pertcol+' == @compound2')

        compound1_df_profiles = get_featuredata(compound1_df).values
        compound2_df_profiles = get_featuredata(compound2_df).values
        try:
            corr = np.corrcoef(compound1_df_profiles, compound2_df_profiles)
            corr = corr[0:len(compound1_df_profiles), len(compound1_df_profiles):]

            corr_median_value = np.nanmedian(corr, axis=1)
            corr_median_value = np.nanmedian(corr_median_value)

            null_corr.append(corr_median_value)
        except:
            pass

    return null_corr

def corr_between_compound_moa(df, metadata_moa, metadata_compound_name):
    """
        Correlation between compounds with the same MOA
        Parameters:
        -----------
        df: pd.DataFrame
        metadata_moa: MOA feature
        metadata_compound_name: Compound name feature
        Returns:
        --------
        list-like of correlation values
     """
    replicate_corr = []

    profile_df = (
        get_metadata(df)
        .assign(profiles=list(get_featuredata(df).values))
    )

    replicate_grouped = (
        profile_df.groupby([metadata_moa, metadata_compound_name]).profiles
            .apply(list)
            .reset_index()
    )

    moa_grouped = (
        replicate_grouped.groupby([metadata_moa]).profiles
            .apply(list)
            .reset_index()
    )

    for i in range(len(moa_grouped)):
        if len(moa_grouped.iloc[i].profiles) > 1:
            compound1_profiles = moa_grouped.iloc[i].profiles[0]
            compound2_profiles = moa_grouped.iloc[i].profiles[1]

            corr = np.corrcoef(compound1_profiles, compound2_profiles)
            corr = corr[0:len(moa_grouped.iloc[i].profiles[0]), len(moa_grouped.iloc[i].profiles[0]):]
            # np.fill_diagonal(corr, np.nan)
            replicate_corr.append(np.nanmedian(corr))

    return replicate_corr

def null_corr_between_compound_moa(df, n_samples, metadata_moa, metadata_compound_name):
    """
        Null distribution between random pairs of compounds.
        Parameters:
        ------------
        df: pandas.DataFrame
        n_samples: int
        metadata_moa: MOA feature
        metadata_compound_name: Compound name feature
        Returns:
        --------
        list-like of correlation values, with a  length of `n_samples`
    """
    df.reset_index(drop=True, inplace=True)
    null_corr = []

    profile_df = (
        get_metadata(df)
        .assign(profiles=list(get_featuredata(df).values))
    )

    replicate_grouped = (
        profile_df.groupby([metadata_moa, metadata_compound_name]).profiles
            .apply(list)
            .reset_index()
    )

    while len(null_corr) < n_samples:
        compounds = random.choices([_ for _ in range(len(replicate_grouped))], k=2)
        compound1_moa = replicate_grouped.iloc[compounds[0]].Metadata_moa
        compound2_moa = replicate_grouped.iloc[compounds[1]].Metadata_moa
        if compound1_moa != compound2_moa:
            compound1_profiles = replicate_grouped.iloc[compounds[0]].profiles
            compound2_profiles = replicate_grouped.iloc[compounds[1]].profiles
            corr = np.corrcoef(compound1_profiles, compound2_profiles)
            corr = corr[0:len(replicate_grouped.iloc[0].profiles), len(replicate_grouped.iloc[0].profiles):]
            # np.fill_diagonal(corr, np.nan)
            null_corr.append(np.nanmedian(corr))  # median replicate correlation
    return null_corr

def correlation_between_modalities(modality_1_df, modality_2_df, modality_1, modality_2, metadata_common, metadata_perturbation):
    """
    Compute the correlation between two different modalities.
    :param modality_1_df: Profiles of the first modality
    :param modality_2_df: Profiles of the second modality
    :param modality_1: feature that identifies perturbation pairs
    :param modality_2: perturbation name feature
    :param metadata_common: perturbation name feature
    :param metadata_perturbation: perturbation name feature
    :return: list-like of correlation values
    """
    list_common_perturbation_groups = list(np.intersect1d(list(modality_1_df[metadata_common]), list(modality_2_df[metadata_common])))

    merged_df = pd.concat([modality_1_df, modality_2_df], ignore_index=False, join='inner')

    modality_1_df = merged_df.query('Metadata_modality==@modality_1')
    modality_2_df = merged_df.query('Metadata_modality==@modality_2')

    corr_modalities = []

    for group in list_common_perturbation_groups:
        modality_1_perturbation_df = modality_1_df.loc[modality_1_df[metadata_common] == group]
        modality_2_perturbation_df = modality_2_df.loc[modality_2_df[metadata_common] == group]

        for sample_1 in modality_1_perturbation_df[metadata_perturbation].unique():
            for sample_2 in modality_2_perturbation_df[metadata_perturbation].unique():
                modality_1_perturbation_sample_df = modality_1_perturbation_df.loc[modality_1_perturbation_df[metadata_perturbation] == sample_1]
                modality_2_perturbation_sample_df = modality_2_perturbation_df.loc[modality_2_perturbation_df[metadata_perturbation] == sample_2]

                modality_1_perturbation_profiles = get_featuredata(modality_1_perturbation_sample_df)
                modality_2_perturbation_profiles = get_featuredata(modality_2_perturbation_sample_df)

                corr = np.corrcoef(modality_1_perturbation_profiles, modality_2_perturbation_profiles)
                corr = corr[0:len(modality_1_perturbation_profiles), len(modality_1_perturbation_profiles):]
                corr_modalities.append(np.nanmedian(corr))  # median replicate correlation

    return corr_modalities

def null_correlation_between_modalities(modality_1_df, modality_2_df, modality_1, modality_2, metadata_common, metadata_perturbation, n_samples):
    """
    Compute the correlation between two different modalities.
    :param modality_1_df: Profiles of the first modality
    :param modality_2_df: Profiles of the second modality
    :param modality_1: "Compound", "ORF" or "CRISPR"
    :param modality_2: "Compound", "ORF" or "CRISPR"
    :param metadata_common: feature that identifies perturbation pairs
    :param metadata_perturbation: perturbation name feature
    :param n_samples: int
    :return:
    """
    list_common_perturbation_groups = list(np.intersect1d(list(modality_1_df[metadata_common]), list(modality_2_df[metadata_common])))

    merged_df = pd.concat([modality_1_df, modality_2_df], ignore_index=False, join='inner')

    modality_1_df = merged_df.query('Metadata_modality==@modality_1')
    modality_2_df = merged_df.query('Metadata_modality==@modality_2')

    null_modalities = []

    count = 0

    while count < n_samples:
        perturbations = random.choices(list_common_perturbation_groups, k=2)
        modality_1_perturbation_df = modality_1_df.loc[modality_1_df[metadata_common] == perturbations[0]]
        modality_2_perturbation_df = modality_2_df.loc[modality_2_df[metadata_common] == perturbations[1]]

        for sample_1 in modality_1_perturbation_df[metadata_perturbation].unique():
            for sample_2 in modality_2_perturbation_df[metadata_perturbation].unique():
                modality_1_perturbation_sample_df = modality_1_perturbation_df.loc[modality_1_perturbation_df[metadata_perturbation] == sample_1]
                modality_2_perturbation_sample_df = modality_2_perturbation_df.loc[modality_2_perturbation_df[metadata_perturbation] == sample_2]

                modality_1_perturbation_profiles = get_featuredata(modality_1_perturbation_sample_df)
                modality_2_perturbation_profiles = get_featuredata(modality_2_perturbation_sample_df)

                corr = np.corrcoef(modality_1_perturbation_profiles, modality_2_perturbation_profiles)
                corr = corr[0:len(modality_1_perturbation_profiles), len(modality_1_perturbation_profiles):]
                null_modalities.append(np.nanmedian(corr))  # median replicate correlation
        count += 1

    return null_modalities

class ZCA_corr(BaseEstimator, TransformerMixin):
    def __init__(self, copy=False):
        self.copy = copy

    def estimate_regularization(self, eigenvalue):
        x = [_ for _ in range(len(eigenvalue))]
        kneedle = kneed.KneeLocator(x, eigenvalue, S=1.0, curve='convex', direction='decreasing')
        reg = eigenvalue[kneedle.elbow]/10.0
        return reg # The complex part of the eigenvalue is ignored

    def fit(self, X, y=None):
        """
        Compute the mean, sphereing and desphereing matrices.
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data used to compute the mean, sphereing and desphereing
            matrices.
        """
        X = check_array(X, accept_sparse=False, copy=self.copy, ensure_2d=True)
        X = as_float_array(X, copy=self.copy)
        self.mean_ = X.mean(axis=0)
        X_ = X - self.mean_
        cov = np.dot(X_.T, X_) / (X_.shape[0] - 1)
        V = np.diag(cov)
        df = pd.DataFrame(X_)
        corr = np.nan_to_num(df.corr()) # replacing nan with 0 and inf with large values
        G, T, _ = scipy.linalg.svd(corr)
        regularization = self.estimate_regularization(T.real)
        t = np.sqrt(T.clip(regularization))
        t_inv = np.diag(1.0 / t)
        v_inv = np.diag(1.0/np.sqrt(V.clip(1e-3)))
        self.sphere_ = np.dot(np.dot(np.dot(G, t_inv), G.T), v_inv)
        return self

    def transform(self, X, y=None, copy=None):
        """
        Parameters
        ----------
        X : array-like with shape [n_samples, n_features]
            The data to sphere along the features axis.
        """
        check_is_fitted(self, "mean_")
        X = as_float_array(X, copy=self.copy)
        return np.dot(X - self.mean_, self.sphere_.T)

def sphere_plate_zca_corr(plate):
    """
    sphere each plate to the DMSO negative control values
    Parameters:
    -----------
    plate: pandas.DataFrame
        dataframe of a single plate's featuredata and metadata
    Returns:
    -------
    pandas.DataFrame of the same shape as `plate`
    """
    # sphere featuredata to DMSO sphereing matrix
    sphereer = ZCA_corr()
    dmso_df = plate.loc[plate.Metadata_control_type=="negcon"]
    # dmso_df = plate.query("Metadata_pert_type == 'control'")
    dmso_vals = get_featuredata(dmso_df).to_numpy()
    all_vals = get_featuredata(plate).to_numpy()
    sphereer.fit(dmso_vals)
    sphereed_vals = sphereer.transform(all_vals)
    # concat with metadata columns
    feature_df = pd.DataFrame(
        sphereed_vals, columns=get_featurecols(plate), index=plate.index
    )
    metadata = get_metadata(plate)
    combined = pd.concat([feature_df, metadata], axis=1)
    assert combined.shape == plate.shape
    return combined

def calculate_percent_replicating_MOA(batch_path,plate,data_df=None):
    """
    For plates treated with the JUMP-MOA source plates, at least 
    4 copies of each perturbation are present on each plate.
    Percent replicating is therefore calculated per plate.
    """
    metadata_compound_name = 'Metadata_pert_iname'
    n_samples_strong = 10000
    if type(data_df)!=pd.DataFrame:
        data_df = pd.read_csv(os.path.join(batch_path, plate,
                                            plate+'_normalized_feature_select_negcon_batch.csv.gz'))

    data_df = sphere_plate_zca_corr(data_df)

    data_df = remove_negcon_empty_wells(data_df)

    replicate_corr = list(corr_between_replicates(data_df, metadata_compound_name))
    null_corr = list(corr_between_non_replicates(data_df, n_samples=n_samples_strong, n_replicates=4, metadata_compound_name = metadata_compound_name))

    prop_95, value_95_replicating = percent_score(null_corr, replicate_corr)

    # return(prop_95)
    return replicate_corr, null_corr, prop_95, value_95_replicating

def calculate_percent_replicating_across_plates_MOA(batch_path1,plate1,batch_path2,plate2 ):
    """
    For plates treated with the JUMP-MOA source plates, at least 
    4 copies of each perturbation are present on each plate.
    Percent replicating is therefore calculated per plate.
    """
    metadata_compound_name = 'Metadata_pert_iname'
    n_samples_strong = 10000

    data_df1 = pd.read_csv(os.path.join(batch_path1, plate1,
                                           plate1+'_normalized_feature_select_negcon.csv.gz'))
    data_df1 = sphere_plate_zca_corr(data_df1)
    data_df1 = remove_negcon_empty_wells(data_df1)

    data_df2 = pd.read_csv(os.path.join(batch_path2, plate2,
                                           plate2+'_normalized_feature_select_negcon.csv.gz'))
    data_df2 = sphere_plate_zca_corr(data_df2)
    data_df2 = remove_negcon_empty_wells(data_df2)

    replicate_corr = corr_between_replicates_across_plates(data_df1, data_df2)
    null_corr = corr_between_non_replicates_across_plates(data_df1, data_df2, n_samples=n_samples_strong)

    prop_95, _ = percent_score(null_corr, replicate_corr)

    return(prop_95)

def calculate_percent_matching_MOA(batch_path,plate, data_df=None):
    """
    For plates treated with the JUMP-MOA source plates, at least 
    4 copies of each perturbation are present on each plate.
    Percent replicating is therefore calculated per plate.
    """
    metadata_moa_name = 'Metadata_moa'
    metadata_compound_name = 'Metadata_pert_iname'
    n_samples_strong = 10000

    if type(data_df)!=pd.DataFrame:
        data_df = pd.read_csv(os.path.join(batch_path, plate,
                                            plate+'_normalized_feature_select_negcon.csv.gz'))

    data_df = sphere_plate_zca_corr(data_df)

    data_df = remove_negcon_empty_wells(data_df)

    replicate_corr = list(corr_between_compound_moa(data_df, metadata_moa_name, metadata_compound_name))
    null_corr = list(null_corr_between_compound_moa(data_df, n_samples_strong, metadata_moa_name, metadata_compound_name))

    prop_95, value_95_matching = percent_score(null_corr, replicate_corr)

    return replicate_corr, null_corr, prop_95, value_95_matching

def calculate_percent_replicating_Target(batch_path,platelist,sphere=None,
suffix = '_normalized_feature_select_negcon.csv.gz',n_replicates=4):
    """
    For plates treated with the JUMP-Target source plates, most 
    perturbations are only present in one or two 2 copies per plate. 
    Percent replicating is therefore calculated per group of replicate plates.

    Since feature selection happens on a per-plate level, an inner join
    is performed across all plates in the replicate, meaning only common
    features are used in calculation of percent replicating.

    It doesn't look like sphering was done consistently in previous 
    analysis of these plates, therefore it is configurable here; either 
    not done, done at the plate level by passing 'sphere=plate', or 
    done at the batch level by passing 'sphere=batch'.
    """
    metadata_compound_name = 'Metadata_broad_sample'
    n_samples_strong = 10000

    data_dict = {}

    for plate in platelist:
        plate_df = pd.read_csv(os.path.join(batch_path, plate,
                                            plate+suffix))
        
        if sphere == 'plate':
            plate_df = sphere_plate_zca_corr(plate_df)

        data_dict[plate] = plate_df
    
    data_df = pd.concat(data_dict, join='inner', ignore_index=True)

    if sphere == 'batch':
        data_df = sphere_plate_zca_corr(data_df)

    data_df = remove_negcon_empty_wells(data_df)

    replicate_corr = list(corr_between_replicates(data_df, metadata_compound_name))
    null_corr = list(corr_between_non_replicates(data_df, n_samples=n_samples_strong, n_replicates=n_replicates, metadata_compound_name = metadata_compound_name))

    prop_95, _ = percent_score(null_corr, replicate_corr)

    return(prop_95)

def calculate_percent_replicating_across_plates_Target(batch_path1,plate1,batch_path2,plate2 ):
    """
    For Target 1 vs Target 2
    """
    metadata_compound_name = 'Metadata_broad_sample'
    n_samples_strong = 10000

    data_df1 = pd.read_csv(os.path.join(batch_path1, plate1,
                                           plate1+'_normalized_feature_select_negcon.csv.gz'))
    data_df1 = remove_negcon_empty_wells(data_df1)

    data_df2 = pd.read_csv(os.path.join(batch_path2, plate2,
                                           plate2+'_normalized_feature_select_negcon.csv.gz'))
    data_df2 = remove_negcon_empty_wells(data_df2)

    replicate_corr = corr_between_replicates_across_plates(data_df1, data_df2,pertcol=metadata_compound_name)
    null_corr = corr_between_non_replicates_across_plates(data_df1, data_df2, n_samples=n_samples_strong,pertcol=metadata_compound_name)

    prop_95, _ = percent_score(null_corr, replicate_corr)

    return(prop_95)

def calculate_percent_matching_Target(batch_path_1,platelist_1,modality_1, batch_path_2,platelist_2, modality_2,
sphere=None,suffix = '_normalized_feature_select_negcon.csv.gz'):
    """

    It doesn't look like sphering was done consistently in previous 
    analysis of these plates, therefore it is configurable here; either 
    not done, done at the plate level by passing 'sphere=plate', or 
    done at the batch level by passing 'sphere=batch'.
    """
    n_samples = 10000

    data_dict_1 = {}
    for plate in platelist_1:
        plate_df = pd.read_csv(os.path.join(batch_path_1, plate,
                                            plate+suffix))
        if sphere == 'plate':
            plate_df = sphere_plate_zca_corr(plate_df)

        data_dict_1[plate] = plate_df   
    data_df_1 = pd.concat(data_dict_1, join='inner', ignore_index=True)
    if modality_1 =='Compounds':
        data_df_1.rename(columns={'Metadata_target':'Metadata_genes'},inplace=True)
    data_df_1['Metadata_modality'] = modality_1
    if sphere == 'batch':
        data_df_1 = sphere_plate_zca_corr(data_df_1)
    data_df_1 = remove_negcon_empty_wells(data_df_1)

    data_dict_2 = {}
    for plate in platelist_2:
        plate_df = pd.read_csv(os.path.join(batch_path_2, plate,
                                            plate+suffix))
        if sphere == 'plate':
            plate_df = sphere_plate_zca_corr(plate_df)

        data_dict_2[plate] = plate_df   
    data_df_2 = pd.concat(data_dict_2, join='inner', ignore_index=True)
    if modality_2 =='Compounds':
        data_df_2.rename(columns={'Metadata_target':'Metadata_genes'},inplace=True)
    data_df_2['Metadata_modality'] = modality_2
    if sphere == 'batch':
        data_df_2 = sphere_plate_zca_corr(data_df_2)
    data_df_2 = remove_negcon_empty_wells(data_df_2)

    replicate_corr = list(correlation_between_modalities(data_df_1, data_df_2, modality_1, modality_2, 'Metadata_genes', 'Metadata_broad_sample'))
    null_corr = list(null_correlation_between_modalities(data_df_1, data_df_2, modality_1, modality_2, 'Metadata_genes', 'Metadata_broad_sample', n_samples))

    prop_95, _, _ = percent_score(null_corr, replicate_corr, how='both')

    return(prop_95)

def calculate_percent_matching_with_feature_dropout_Target(batch_path_1,platelist_1,modality_1, batch_path_2,platelist_2, modality_2,
sphere=None,suffix = '_normalized_negcon.csv.gz',drop='AGP'):
    """

    It doesn't look like sphering was done consistently in previous 
    analysis of these plates, therefore it is configurable here; either 
    not done, done at the plate level by passing 'sphere=plate', or 
    done at the batch level by passing 'sphere=batch'.
    """
    import pycytominer
    n_samples = 10000

    data_dict_1 = {}
    for plate in platelist_1:
        plate_df = pd.read_csv(os.path.join(batch_path_1, plate,
                                            plate+suffix))
        cols_to_drop = [x for x in plate_df.columns if drop in x]
        plate_df.drop(columns=cols_to_drop,inplace=True)
        feature_select_features = pycytominer.cyto_utils.infer_cp_features(
        plate_df
        )
        plate_df = pycytominer.feature_select(
        profiles=plate_df,
        features=feature_select_features,
        operation=['variance_threshold','correlation_threshold',
        'drop_na_columns','blocklist']
        )
        if sphere == 'plate':
            plate_df = sphere_plate_zca_corr(plate_df)

        data_dict_1[plate] = plate_df   
    data_df_1 = pd.concat(data_dict_1, join='inner', ignore_index=True)
    if modality_1 =='Compounds':
        data_df_1.rename(columns={'Metadata_target':'Metadata_genes'},inplace=True)
    data_df_1['Metadata_modality'] = modality_1
    if sphere == 'batch':
        data_df_1 = sphere_plate_zca_corr(data_df_1)
    data_df_1 = remove_negcon_empty_wells(data_df_1)

    data_dict_2 = {}
    for plate in platelist_2:
        plate_df = pd.read_csv(os.path.join(batch_path_1, plate,
                                            plate+suffix))
        cols_to_drop = [x for x in plate_df.columns if drop in x]
        plate_df.drop(columns=cols_to_drop,inplace=True)
        feature_select_features = pycytominer.cyto_utils.infer_cp_features(
        plate_df
        )
        plate_df = pycytominer.feature_select(
        profiles=plate_df,
        features=feature_select_features,
        operation=['variance_threshold','correlation_threshold',
        'drop_na_columns','blocklist']
        )
        if sphere == 'plate':
            plate_df = sphere_plate_zca_corr(plate_df)

        data_dict_2[plate] = plate_df   
    data_df_2 = pd.concat(data_dict_2, join='inner', ignore_index=True)
    if modality_2 =='Compounds':
        data_df_2.rename(columns={'Metadata_target':'Metadata_genes'},inplace=True)
    data_df_2['Metadata_modality'] = modality_2
    if sphere == 'batch':
        data_df_2 = sphere_plate_zca_corr(data_df_2)
    data_df_2 = remove_negcon_empty_wells(data_df_2)

    replicate_corr = list(correlation_between_modalities(data_df_1, data_df_2, modality_1, modality_2, 'Metadata_genes', 'Metadata_broad_sample'))
    null_corr = list(null_correlation_between_modalities(data_df_1, data_df_2, modality_1, modality_2, 'Metadata_genes', 'Metadata_broad_sample', n_samples))

    prop_95, _, _ = percent_score(null_corr, replicate_corr, how='both')

    return(prop_95)

def plot_simple_comparison(df,x,hue,y='Percent Replicating',order=None,hue_order=None,
col=None, col_order=None, col_wrap=None,row=None,row_order=None,jitter=0.25,dodge=True,plotname=None,
ylim=None, title=None,aspect=1,sharex=True,facet_kws={}):
    plt.rcParams["legend.markerscale"] =1.5
    sns.set_style("ticks")
    sns.set_context("paper",font_scale=1.5)
    g = sns.catplot(data=df, x = x ,y = y, order=order,
    hue=hue, hue_order=hue_order, col=col, col_order = col_order, col_wrap=col_wrap,row=row,
    row_order = row_order, palette='Set1',s=12,linewidth=1,jitter=jitter,
    alpha=0.8,dodge=dodge,aspect=aspect,sharex=sharex,facet_kws=facet_kws)
    if sharex:
        labels = []
        if not order:
            orig_labels = list(dict.fromkeys(df[x].values).keys())
        else:
            orig_labels = order
        for label in orig_labels:
            if type(label)!= str:
                label = str(int(label))
            labels.append(textwrap.fill(label, width=45/len(orig_labels),break_long_words=False))
        g.set_xticklabels(labels=labels,rotation=0)
    if ylim:
        ymin,ymax=ylim
    else:
        ymin = 50
        ymax = 80 
    if df[y].min()<ymin:
        ymin = df[y].min()-2
    if df[y].max()>ymax:
        ymax = df[y].max()+2
    g.set(ylim=([ymin,ymax]))    
    if plotname:
        plotname = f"../figures/{plotname}"
    else:
        plotname = f"../figures/{x}-{y}-{hue}-{col}-{row}.png"
    if not col:
        if not row:
            if title:
                g.set(title=title)
            else:
                g.set(title=f"{x}-{y}")
    g.savefig(plotname,dpi=300)
    print(f'Saved to {plotname}')

def plot_two_comparisons(df,x='Percent Replicating',y='Percent Matching',hue = None, hue_order=None,
col=None, col_order=None,col_wrap=None,row=None,row_order=None,style=None,xlim=None,ylim=None,title=None,
title_variable = None, facet_kws={'sharex':True}):
    plt.rcParams["legend.markerscale"] =1.5
    sns.set_style("ticks")
    sns.set_context("paper",font_scale=1.5)
    g = sns.relplot(data=df, x = x ,y= y, hue=hue, hue_order=hue_order, col=col, col_order = col_order, 
    col_wrap=col_wrap, row=row, row_order = row_order, style = style, palette='Set1',edgecolor='k',alpha=0.8,s=80,
    facet_kws=facet_kws)
    if xlim:
        xmin,xmax=xlim
    else:
        xmin = 50
        xmax = 80
    if df[x].min()<xmin:
        xmin = df[x].min()-2
    if df[x].max()>xmax:
        xmax = df[x].max()+2
    g.set(xlim=([xmin,xmax]))  
    if ylim:
        ymin,ymax=ylim
    else:
        ymin = 5
        ymax = 40 
    if df[y].min()<ymin:
        ymin = df[y].min()-2
    if df[y].max()>ymax:
        ymax = df[y].max()+2
    g.set(ylim=([ymin,ymax]))
    if title:
        g.set(title=title)
    elif title_variable:
        g.set(title=title_variable)
    plotname = f"../figures/{x}-{y}-{hue}-{col}-{row}-{style}.png"
    g.savefig(plotname,dpi=300)
    print(f'Saved to {plotname}')

def enforce_modality_match_order(modality1,modality2):
    modality_dict_forward = {'Compounds':1,'ORF':2,'CRISPR':3}
    modality_dict_reverse = {v:k for k,v in modality_dict_forward.items()}
    modality_list = [modality_dict_forward[modality1],modality_dict_forward[modality2]]
    modality_list.sort()
    return f"{modality_dict_reverse[modality_list[0]]} - {modality_dict_reverse[modality_list[1]]}"

def enforce_timepoint_order(timepoint1,timepoint2):
    timepoint_list = [int(timepoint1),int(timepoint2)]
    timepoint_list.sort()
    return f"{timepoint_list[0]}-{timepoint_list[1]}"

def enforce_timepoint_order_in_plot(timepointlist):
    timepointlist=list(set(timepointlist))
    intlist = []
    timepoint_dict = {}
    for x in timepointlist:
        first,second=x.split('-')
        intlist.append([int(first),int(second)])
    intlist.sort()
    outlist = []
    for x in intlist:
        outlist.append(f"{x[0]}-{x[1]}")
    return outlist

def enforce_modality_match_order_in_plot(modalitylist):
    modalitylist=list(set(modalitylist))
    modality_dict_forward = {'Compounds':1,'ORF':2,'CRISPR':3}
    modality_dict_reverse = {v:k for k,v in modality_dict_forward.items()}
    master_modalitylist = []
    for eachpair in modalitylist:
        modality1,modality2 = eachpair.split(' - ')
        sublist = [modality_dict_forward[modality1],modality_dict_forward[modality2]]
        sublist.sort()
        master_modalitylist.append(sublist)
    master_modalitylist.sort()
    outlist = [f"{modality_dict_reverse[x[0]]} - {modality_dict_reverse[x[1]]}" for x in master_modalitylist]    
    return outlist

def safe_literal_eval(node):
    """
    If eval doesn't work, make it a np.nan
    For JUMP-Scope, this is the only other string
    found in lists
    """
    try:
        return ast.literal_eval(node)
    except SyntaxError:
        return np.nan 
    except ValueError:
        return np.nan

def group_plot(
    df, 
    x, 
    y, 
    group, 
    error_x=None,
    error_y=None, 
    fig=None, 
    ax_=None, 
    legend=False, 
    legend_title=None,
    legend_location=None,
    infer_custom_legend=False,
    s=None,
    use_markers=False,
    label=None, 
    alpha=None,
    x_lim=None,
    y_lim=None,
    plot_title=False,
    xlabel=None,
    ylabel=None
    ):
    cmap = plt.cm.tab10
    colour_palette = list()
    markers = "sxo^D2"
    for i in range(cmap.N):
        colour_palette.append(cmap(i))
    
    if fig is None and ax_ is None:
        fig, ax = plt.subplots()
    else:
        ax = ax_
    for i, (group_label, group_df) in enumerate(df.groupby(group)):
        if infer_custom_legend:
            if len(group_df) > 1:
                raise ValueError("Cannot infer custom legend from grouped DF of more than 1")
            # Columns to drop that are not to be used for paired comparison
            infer_legend = df.apply(pd.Series.duplicated).any().drop([
                "Batch", 
                "Assay_Plate_Barcode", 
                "Percent_Replicating", 
                "Percent_Matching", 
                "euclidean_distance", 
                "cell_count"])
            infer_legend = infer_legend[~infer_legend].index
        if error_x is not None or error_y is not None:
            ax.errorbar(
                group_df.loc[:, x], 
                group_df.loc[:, y],
                xerr=group_df[error_x] if error_x else None,
                yerr=group_df[error_y] if error_y else None,
                ecolor=colour_palette[i],
                # label=group_label,
                # fmt="None",
                # zorder=1,
                lw=2
            )
        else:
            ax.scatter(
                group_df.loc[:, x],
                group_df.loc[:, y],
                color=colour_palette[i], 
                # We use .iloc[0] here becasue we assume that every group has a len(df) of 1
                # Labels cannot be resolved for groups larger than 1
                label=group_label if not infer_custom_legend else group_df[infer_legend].iloc[0].to_dict(),
                alpha=alpha,
                marker=markers[i] if use_markers else None,
                s=s
                # zorder=2
            )
        
        if x_lim:
            ax.set_xlim(x_lim)

        if y_lim:
            ax.set_ylim(y_lim)

        if legend:
            ax.legend(loc=legend_location if legend_location else None, title=legend_title)

        if label:
            for col, rows in label.items():
                for item in rows:
                    annotate_x = group_df[group_df[col] == item][x].values
                    annotate_y = group_df[group_df[col] == item][y].values
                    ax.annotate(item, annotate_x, annotate_y)

    if plot_title:
        ax.set_title(group if not plot_title else plot_title, size=15)
    ax.set_xlabel(x if not xlabel else xlabel, fontsize=15)
    ax.set_ylabel(y if not ylabel else ylabel, fontsize=15)
    plt.tight_layout()
    fig.set_facecolor("white")
    plt.subplots_adjust( 
                    wspace=0.2,
                    hspace=0.2)

def find_group_avg_df(_df, group, **kwargs):
    df = _df.copy()
    # Define operations to perform on columns during aggregation
    agg_dict = {
        'Percent_Replicating' : lambda x: list(x),
        'Mean_Percent_Replicating' : lambda y: np.mean(y),
        'SD_Percent_Replicating' : lambda z: float('%.3f'%np.std(z)),
        'Percent_Matching' : lambda x: list(x),
        'Mean_Percent_Matching' : lambda y: np.mean(y),
        'SD_Percent_Matching' : lambda z: float('%.3f'%np.std(z)),
    }
    # Make new columns
    df['Mean_Percent_Replicating'] = list(df['Percent_Replicating'])
    df['SD_Percent_Replicating'] = list(df['Percent_Replicating'])
    df['Mean_Percent_Matching'] = list(df['Percent_Matching'])
    df['SD_Percent_Matching'] = list(df['Percent_Matching'])
    # Implement desired cols and define operations
    if "add_cols" in kwargs:
        for new_col in kwargs["add_cols"].values():
            df[new_col[1]] = list(df[new_col[0]])
            # new_col[2] is the function to apply
            agg_dict.update({new_col[1]: new_col[2]})
    # Perform aggregation
    group_df = df.groupby(group,as_index=False).agg(agg_dict)
    return group_df

def profile_finder(df, profile_path, profile_type):
    """Returns a list of DataFrames from the provided df"""
    # Store profile types
    type_dict = {
        "unnormalized": ".csv.gz",
        "norm_feature_selected": "_normalized_feature_select_negcon_plate.csv.gz"
        }
    
    # Select desired profile type
    suffix = type_dict[profile_type]

    output = list()
    plate_info = list()

    for ind, row in df.iterrows():
        df_path = os.path.join(profile_path, row["Batch"], row["Assay_Plate_Barcode"], f"{row['Assay_Plate_Barcode']}{suffix}")
        print(f"loading: {df_path}")
        output.append(pd.read_csv(df_path))
        plate_info.append({"Batch": row["Batch"], "Assay_Plate_Barcode": row["Assay_Plate_Barcode"]})
    
    return output, plate_info

def compare_paired_correlations(
    df: pd.DataFrame, 
    profile_path: str,  
    correlation_metric: str, 
    grouping_metric: str,
    fig=None,
    ax=None,
    plot_title=None,
    xlabel=None,
    ylabel=None,
    ):
    """
    Between two profiles, return a dataframe that contains only the desired feature
    """
    # Load requested profiles
    profiles, plate_info = profile_finder(df, profile_path, "unnormalized")

    # Keep only the correlation columns
    for i, i_df in enumerate(profiles):
        mask_cols = [] # Store the columns to mask, find for each profile independently
        for cols in i_df.columns:
            if correlation_metric in cols and "BrightField" not in cols:
                mask_cols.append(cols)
        # Mask columns
        profiles[i] = i_df[mask_cols]

        unique_value = df[
            (df["Batch"].str.contains(plate_info[i]["Batch"])) &
            (df["Assay_Plate_Barcode"].str.contains(plate_info[i]["Assay_Plate_Barcode"]))
            ][grouping_metric].values
    
        profiles[i][grouping_metric] = unique_value[0]
    
    profiles = pd.concat(profiles)

    for j in profiles.groupby(grouping_metric):
        print(j[1].shape[0])

    profiles = profiles.melt(id_vars=grouping_metric)

    fig.set_facecolor("white")

    seaborn.violinplot(
        ax=ax,
        data=profiles,
        y="variable",
        x="value",
        orient="h",
        hue=grouping_metric,
        split=True,
        cut=0, 
        figsize=(20, 10),
        palette="muted"
        )

    ax.set_title(None if not plot_title else plot_title, size=15)
    ax.set_xlabel(None if not xlabel else xlabel, fontsize=15)
    ax.set_ylabel(None if not ylabel else ylabel, fontsize=15)
    plt.tight_layout()


def compare_single_correlations(
    df: pd.DataFrame, 
    profile_path: str, 
    correlation_metric: str, 
    fig=None,
    ax=None,
    plot_title=None,
    xlabel=None,
    ylabel=None,
    ):
    """
    Between two profiles, return a dataframe that contains only the desired feature
    """
    # Load requested profiles
    profiles, _ = profile_finder(df, profile_path, "unnormalized")
    # Keep only the correlation columns
    for i, i_df in enumerate(profiles):
        mask_cols = [] # Store the columns to mask, find for each profile independently
        for cols in i_df.columns:
            if correlation_metric in cols and "BrightField" not in cols:
                mask_cols.append(cols)
        # Mask columns
        profiles[i] = i_df[mask_cols]

    profiles = pd.concat(profiles)

    profiles = profiles.melt()

    fig.set_facecolor("white")

    seaborn.boxenplot(
        ax=ax,
        data=profiles,
        y="variable",
        x="value",
        orient="h",
        showfliers=False
        )


    ax.set_title(None if not plot_title else plot_title, size=15)
    ax.set_xlabel(None if not xlabel else xlabel, fontsize=15)
    ax.set_ylabel(None if not ylabel else ylabel, fontsize=15)
    plt.tight_layout()

def format_column_names(column_name):
    """
    For a given string, hyphens and underscores are replaced with spaces and 
    first letters are capitalised.
    """
    return column_name.title().replace("-", " ").replace("_", " ")

def aggregate_duplicates(df, non_group_cols):
    """
    Group data on setting columns, calculate the mean for grouped rows
    (which are therefore duplicates).

    The returned dataframe will not contain Assay_Plate_Barcode, since 
    the unique barcodes cannot be reconciled into an aggregated mean, 
    so are therefore dropped.
    """
    # Columns to not be used for grouping
    # non_group_cols = [
    #     "Assay_Plate_Barcode",
    #     "Batch",
    #     "Vendor",
    #     "value_95_replicating",
    #     "Percent_Replicating",
    #     "Size_MB",
    #     "Size_MB_std",
    #     "Percent_Matching",
    #     "value_95_matching",
    #     "cell_count",
    #     "Sites-SubSampled",
    #     "BF_Zplanes", # Ignore since only one BF zplane is used
    #     "brightfield_z_plane_used",
    #     # Remove channel names due to some profiles having "AGP" features 
    #     # and others having "WGPhalloidin" instead
    #     "feature_channels_found",
    #     "channel_names"
    # ]
    # Find the columns that are not in non_group_cols
    diff = list(set(df.columns) - set(non_group_cols))
    # Group df by setting columns, find the mean, then reset the index
    df = df.groupby(diff, dropna=False, as_index=False).mean()
    return df

def make_leaderboard(df, columns, non_group_cols, average_duplicates=True):
    """
    Process match_rep_df into a nice leaderboard
    """
    if average_duplicates:
        df = aggregate_duplicates(df, non_group_cols)

    # Create an aggregation score for replicating/matching and normalize to the max value
    df["Percent_Score"] = df[["Percent_Replicating", "Percent_Matching"]].mean(axis=1)
    df["Percent_Score"] = (df["Percent_Score"] / df["Percent_Score"].max()) * 100
    df = df.round({"Percent_Score": 1})

    df = df.round({"Percent_Replicating": 1})
    df = df.round({"Percent_Matching": 1})

    # Sort based on the Percent_Score and then add Place (ie 1st, 2nd, 3rd etc.)
    df = df.sort_values("Percent_Score", ascending=False)[columns]
    df["Place"] = df.reset_index(drop=True).index+1
    
    # Move place column to first position
    df.insert(0, "Place", df.pop("Place"))

    # Format column_names
    df.rename(columns=format_column_names, inplace=True)
    df.rename(columns={"Z Plane": "Z Planes"}, inplace=True)
    df.rename(columns={"Aperture": "NA"}, inplace=True)
    df.rename(columns={"Dry Immersion": "Immersion"}, inplace=True)
    return df

def aggregate_comparison_pvalues(df: pd.DataFrame, remove_cols: list, aggregate_rows: bool, aggregate_similar_features: bool):
    # Remove columns that are in remove cols and also only keep columns that are pvalues
    df = df.loc[:, (~df.columns.str.startswith(tuple(remove_cols))) & (df.columns.str.contains("_pvalue"))]
    # Drop Location, Number, Parent and Neighbour features
    df = df.loc[:, ~(df.columns.str.contains("|".join(["Location", "Number", "Parent", "Neighbors", "Children"])))]
    if aggregate_similar_features:
        compartments = ["Nuclei", "Cytoplasm", "Cells"]
        result_df = pd.DataFrame()

        for compart in compartments:
            subset_cols = [col for col in df.columns if compart in col]
            sub_df = df[subset_cols]
            # print(sub_df.columns)

            # Find the first 3 words of column names that will be used for grouping
            group_words = [col.split("_")[:3] for col in subset_cols]
            # Join them back
            group_words = ["_".join(w) for w in group_words]

            # Find the mean for the feature groups
            # result_df = pd.concat([result_df, sub_df.groupby(group_words, axis=1).mean().reset_index()])
            result_df = result_df.join(sub_df.groupby(group_words, axis=1).mean(), how="right")

        df = result_df
    # Convert DF to long format
    if aggregate_rows:
        df = df.mean(axis=0).to_frame().T.melt(var_name="feature", value_name="pvalue")
    else:
        df = df.melt(var_name="feature", value_name="pvalue")
    df["feature_type"] = df["feature"].str.split("_").str[1]
    return df

def combine_evalzoo_metrics(match_rep_df: pd.DataFrame, evalzoo_dir: str = "results/results"):
    # Add Batch and Metadata_Plate since we need to check the presence of a
    # string in them later, and you can't check them if they don't exist
    output_df = pd.DataFrame({"Batch": [], "Metadata_Plate": []})
    for ind, row in match_rep_df.iterrows():
        # Need to do some check that this row has not already been processed (so
        # we don't add it twice)
        if not (row["Batch"] in output_df["Batch"].values and row["Assay_Plate_Barcode"] in output_df["Metadata_Plate"].values):
            ref_metric_path = os.path.join(evalzoo_dir, row["Batch"]+"_ref", f"metrics_level_1_ref.parquet")
            non_rep_metric_path = os.path.join(evalzoo_dir, row["Batch"]+"_non_rep", f"metrics_level_1_non_rep.parquet")

            ref_metric_df = pd.read_parquet(ref_metric_path)
            non_rep_metric_df = pd.read_parquet(non_rep_metric_path)

            # Since we are going to use the metric_type to differentiate between ref
            # and non_rep, remove this information from the feature names
            # ref_metric_df = ref_metric_df.rename(lambda x: x + "_ref" if "sim_" in x else x, axis=1)
            # non_rep_metric_df = non_rep_metric_df.rename(lambda x: x + "_non_rep" if "sim_" in x else x, axis=1)

            ref_metric_df = ref_metric_df.rename(lambda x: x.replace("_ref", "") if "sim_" in x else x, axis=1)
            non_rep_metric_df = non_rep_metric_df.rename(lambda x: x.replace("_non_rep", "") if "sim_" in x else x, axis=1)

            # Add cols
            (
                ref_metric_df["Vendor"], 
                ref_metric_df["Batch"], 
                ref_metric_df["Magnification"], 
                ref_metric_df["Binning"], 
                ref_metric_df["z_plane"], 
                ref_metric_df["metric_type"]
                ) = (
                row["Vendor"], 
                row["Batch"], 
                row["Magnification"], 
                row["Binning"], 
                row["z_plane"], 
                "ref"
            )

            (
                non_rep_metric_df["Vendor"], 
                non_rep_metric_df["Batch"],
                non_rep_metric_df["Magnification"], 
                non_rep_metric_df["Binning"], 
                non_rep_metric_df["z_plane"], 
                non_rep_metric_df["metric_type"]
                ) = (
                row["Vendor"], 
                row["Batch"], 
                row["Magnification"], 
                row["Binning"], 
                row["z_plane"], 
                "non_rep"
                )
            
            # Cast some columns to int
            ref_metric_df["Magnification"] = ref_metric_df["Magnification"].astype("Int64")
            ref_metric_df["Binning"] = ref_metric_df["Binning"].astype("Int64")
            ref_metric_df["z_plane"] = ref_metric_df["z_plane"].astype("Int64")

            non_rep_metric_df["Magnification"] = non_rep_metric_df["Magnification"].astype("Int64")
            non_rep_metric_df["Binning"] = non_rep_metric_df["Binning"].astype("Int64")
            non_rep_metric_df["z_plane"] = non_rep_metric_df["z_plane"].astype("Int64")


            output_df = pd.concat([output_df, ref_metric_df, non_rep_metric_df], axis=0)

    return output_df