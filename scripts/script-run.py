import pandas as pd
import utilssphering
import utils
import os
import random
import numpy as np
import itertools
import pycytominer
import matplotlib.pyplot as plt
import ast
import logging
logging.basicConfig(filename="./log")

random.seed(9000)

# Read new experiment df
experiment_df = pd.read_csv("output/all-profile-metadata.csv")
# Add profile path
profile_path = "/home/ubuntu/ebs_tmp/jump-scope/profiles"

def create_moa_dataframe(experiment_metadata, profile_parent_dir, batch_col="Batch", match_or_rep_or_both="replicating", enable_sphering="both"):
    """
    batch_col is the name of the column to distinguish the profile parent folder. Eg. "Scope1_MolDev_10X" or "1siteSubSample_Scope1_MolDev_10X"
    Output df will also use this batch_col name
    """
    n_samples = 10000
    n_replicates = 4  # number of sample replicates within each plate 
    metadata_common = 'Metadata_moa'
    metadata_perturbation = 'Metadata_broad_sample'
    group_by_feature = 'Metadata_pert_iname'

    corr_replicating_list = list()
    corr_matching_list = list()

    for ind, a_vendor in enumerate(experiment_metadata["Vendor"].unique()):
        print(f"Processing {a_vendor}")
        vendor_data = experiment_metadata.loc[experiment_metadata["Vendor"] == a_vendor]
        for a_batch in vendor_data[batch_col].unique():
            batch_data = vendor_data.loc[vendor_data[batch_col] == a_batch]
            for a_plate in batch_data["Assay_Plate_Barcode"].unique():
                # plate_data = batch_data.loc[batch_data["Assay_Plate_Barcode"] == a_plate]
                data_path = os.path.join(profile_parent_dir, a_batch, a_plate, a_plate+"_normalized_feature_select_negcon_plate.csv.gz")
                load_data = pd.read_csv(data_path)
                print(f"Processing: {a_vendor}/{a_batch}/{a_plate}")
                try:
                    if match_or_rep_or_both.casefold() == "replicating" or match_or_rep_or_both.casefold() == "both":
                        if enable_sphering.casefold() == "yes" or enable_sphering.casefold() == "both":
                            sphere_bool = True
                            replicate_corr_sphere, null_replicating_sphere, prop_95_replicating_sphere, value_95_replicating_sphere = utilssphering.calculate_percent_replicating_MOA("", "", data_df=load_data)
                            corr_replicating_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                        batch_col: a_batch,
                                                                        'Assay_Plate_Barcode': a_plate,
                                                                        'Replicating':[replicate_corr_sphere],
                                                                        'Null_Replicating':[null_replicating_sphere],
                                                                        'Percent_Replicating':prop_95_replicating_sphere,
                                                                        'Value_95':value_95_replicating_sphere,
                                                                        'sphering': sphere_bool}, index=[ind]))

                        if enable_sphering.casefold() == "no" or enable_sphering.casefold() == "both": 
                            sphere_bool = False
                            plate_df = utils.remove_negcon_empty_wells(load_data)
                            replicate_corr = list(utils.corr_between_replicates(plate_df, group_by_feature))
                            null_replicating = list(utils.corr_between_non_replicates(plate_df, n_samples=n_samples, n_replicates=n_replicates, metadata_compound_name=group_by_feature))
                            prop_95_replicating, value_95_replicating = utils.percent_score(null_replicating, replicate_corr, how='right')
                            corr_replicating_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                        batch_col: a_batch,
                                                                        'Assay_Plate_Barcode': a_plate,
                                                                        'Replicating':[replicate_corr],
                                                                        'Null_Replicating':[null_replicating],
                                                                        'Percent_Replicating':prop_95_replicating,
                                                                        'Value_95':value_95_replicating,
                                                                        'sphering': sphere_bool}, index=[ind]))

                    if match_or_rep_or_both.casefold() == "matching" or match_or_rep_or_both.casefold() == "both":
                        if enable_sphering.casefold() == "yes" or enable_sphering.casefold() == "both":
                            sphere_bool = True
                            matching_corr_sphere, null_matching_sphere, prop_95_matching_sphere, value_95_matching_sphere = utilssphering.calculate_percent_matching_MOA("", "", data_df=load_data)
                            corr_matching_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                    batch_col: a_batch,
                                                                    'Assay_Plate_Barcode': a_plate,
                                                                    'Matching':[matching_corr_sphere],
                                                                    'Null_Matching':[null_matching_sphere],
                                                                    'Percent_Matching':prop_95_matching_sphere,
                                                                    'Value_95':value_95_matching_sphere,
                                                                    'sphering': sphere_bool}, index=[ind]))
                        
                        if enable_sphering.casefold() == "no" or enable_sphering.casefold() == "both": 
                            sphere_bool = False
                            plate_df = utils.remove_negcon_empty_wells(load_data)
                            matching_corr = list(utils.corr_between_perturbation_pairs(plate_df, 'Metadata_moa', 'Metadata_broad_sample'))
                            null_matching = list(utils.corr_between_perturbation_non_pairs(plate_df, n_samples=n_samples, metadata_common=metadata_common, metadata_perturbation=metadata_perturbation))
                            prop_95_matching, value_95_matching = utils.percent_score(null_matching, matching_corr, how='right')
                            corr_matching_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                    batch_col: a_batch,
                                                                    'Assay_Plate_Barcode': a_plate,
                                                                    'Matching':[matching_corr],
                                                                    'Null_Matching':[null_matching],
                                                                    'Percent_Matching':prop_95_matching,
                                                                    'Value_95':value_95_matching,
                                                                    'sphering': sphere_bool}, index=[ind]))
                except Exception as e:
                    logging.error(f"Passed: {data_path}", exc_info=e)
    # Concatenate the data
    if match_or_rep_or_both.casefold() == "replicating" or match_or_rep_or_both.casefold() == "both":
        corr_replicating_df = pd.concat(corr_replicating_list, ignore_index=True)
    if match_or_rep_or_both.casefold() == "matching" or match_or_rep_or_both.casefold() == "both":
        corr_matching_df = pd.concat(corr_matching_list, ignore_index=True)
                
    # Merge metadata with output dataframes
    merge_columns = ['Vendor', batch_col, 'Assay_Plate_Barcode']
    if match_or_rep_or_both.casefold() == "both":
        corr_replicating_df = experiment_metadata.merge(corr_replicating_df, how="inner", on=merge_columns)
        corr_matching_df = experiment_metadata.merge(corr_matching_df, how="inner", on=merge_columns)
        return corr_replicating_df, corr_matching_df
    if match_or_rep_or_both.casefold() == "replicating":
        return experiment_metadata.merge(corr_replicating_df, how="inner", on=merge_columns)
    elif match_or_rep_or_both.casefold() == "matching":
        return experiment_metadata.merge(corr_matching_df, how="inner", on=merge_columns)

df_replicating, df_matching = create_moa_dataframe(experiment_df, profile_path, match_or_rep_or_both="both", enable_sphering="both")

def add_total_cell_counts(df, profile_path):
    out_df = df.copy()
    out_df["cell_count"] = ""
    for i in df.index:
        batch = df.loc[i, "Batch"]
        barcode = df.loc[i, "Assay_Plate_Barcode"]
        load_path = os.path.join(profile_path, batch, barcode, f"{barcode}_normalized_negcon.csv.gz")
        load_df = pd.read_csv(load_path)
        try:
            sum_cells = sum(load_df.loc[:,"Metadata_Count_Cells"])
        except:
            # In case a profile is missing cell count data
            sum_cells = np.nan
        out_df.loc[i, "cell_count"] = sum_cells
    return out_df

df_replicating = add_total_cell_counts(df_replicating, profile_path)
df_matching = add_total_cell_counts(df_matching, profile_path)

## Checkpoint save
if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")
df_replicating.to_csv("checkpoints/moa-replicating.csv", index_label='index', index=False)
df_matching.to_csv("checkpoints/moa-matching.csv", index_label='index', index=False)