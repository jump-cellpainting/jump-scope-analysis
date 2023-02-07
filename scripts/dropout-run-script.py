import pandas as pd
import utilssphering
import itertools
import pycytominer
import os
import logging

logging.basicConfig(filename="./log-RNA-dropout")

experiment_metadata = pd.read_csv("output/all-profile-metadata.csv")

# Only need to process brightfield images
experiment_metadata = experiment_metadata[experiment_metadata["channel_names"].str.contains("BrightField")]

profile_parent_dir = "../jump-scope/profiles"

def do_feature_select(plate_df):
    """
    Find the column names that are CellProfiler features. Eg. column names
    that start with "Nuclei" or "Cytoplasm"
    """
    feature_select_features = pycytominer.cyto_utils.infer_cp_features(
        plate_df
    )
    # For all of the cellprofiler features, perform these operations on them
    return pycytominer.feature_select(
        profiles=plate_df,
        features=feature_select_features,
        operation=['variance_threshold','correlation_threshold',
        'drop_na_columns','blocklist']
    )

def create_moa_dataframe(experiment_metadata, profile_parent_dir, batch_col="Batch", match_or_rep_or_both="replicating", enable_sphering="both", dropout_cols=None):
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
    passed_data = list()

    for ind, a_vendor in enumerate(experiment_metadata["Vendor"].unique()):
        print(f"Processing {a_vendor}")
        vendor_data = experiment_metadata.loc[experiment_metadata["Vendor"] == a_vendor]
        for a_batch in vendor_data[batch_col].unique():
            batch_data = vendor_data.loc[vendor_data[batch_col] == a_batch]
            for a_plate in batch_data["Assay_Plate_Barcode"].unique():
                # plate_data = batch_data.loc[batch_data["Assay_Plate_Barcode"] == a_plate]
                data_path = os.path.join(profile_parent_dir, a_batch, a_plate, a_plate+"_normalized_negcon.csv.gz")
                load_data = pd.read_csv(data_path)
                if dropout_cols is not None:
                    input_columns = load_data.columns
                    possible_combinations = [y for x in range(len(dropout_cols)+1) for y in list(set(itertools.combinations(dropout_cols,x)))]
                    if len(possible_combinations) == 0:
                        # Either no combinations available or they're already present in the outfile
                        print("All combinations already computed")
                        return
                    # Iterate through the combination to dropout
                    for dropout_group in possible_combinations: 
                        col_list = input_columns
                        # Within the droupout group, find the column to actually drop
                        for each_item in dropout_group:
                            # Only keep columns that don't contain the dropout
                            col_list = [x for x in col_list if each_item not in x]
                        print(f"---- Dropping: {dropout_group} ----")
                        try:
                            dropped_df = pd.DataFrame(load_data[col_list])
                            feature_selected_df = do_feature_select(dropped_df)
                        except Exception as e:
                            print(f"Error: {e}")
                            feature_select = 0
                        print(data_path)
                        try:
                            if match_or_rep_or_both.casefold() == "replicating" or match_or_rep_or_both.casefold() == "both":
                                if enable_sphering.casefold() == "yes" or enable_sphering.casefold() == "both":
                                    sphere_bool = True
                                    replicate_corr_sphere, null_replicating_sphere, prop_95_replicating_sphere, value_95_replicating_sphere = utilssphering.calculate_percent_replicating_MOA("", "", data_df=feature_selected_df)
                                    corr_replicating_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                                batch_col: a_batch,
                                                                                'Assay_Plate_Barcode': a_plate,
                                                                                "num_features": len(feature_selected_df.columns),
                                                                                "dropout": str(dropout_group),
                                                                                "n_columns": len(col_list),
                                                                                'Replicating':[replicate_corr_sphere],
                                                                                'Null_Replicating':[null_replicating_sphere],
                                                                                'Percent_Replicating':prop_95_replicating_sphere,
                                                                                'Value_95':value_95_replicating_sphere,
                                                                                'sphering': sphere_bool}, index=[ind]))

                                if enable_sphering.casefold() == "no" or enable_sphering.casefold() == "both": 
                                    sphere_bool = False
                                    plate_df = utils.remove_negcon_empty_wells(feature_selected_df)
                                    replicate_corr = list(utils.corr_between_replicates(plate_df, group_by_feature))
                                    null_replicating = list(utils.corr_between_non_replicates(plate_df, n_samples=n_samples, n_replicates=n_replicates, metadata_compound_name=group_by_feature))
                                    prop_95_replicating, value_95_replicating = utils.percent_score(null_replicating, replicate_corr, how='right')
                                    corr_replicating_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                                batch_col: a_batch,
                                                                                'Assay_Plate_Barcode': a_plate,
                                                                                "num_features": len(feature_selected_df.columns),
                                                                                "dropout": str(dropout_group),
                                                                                "n_columns": len(col_list),
                                                                                'Replicating':[replicate_corr],
                                                                                'Null_Replicating':[null_replicating],
                                                                                'Percent_Replicating':prop_95_replicating,
                                                                                'Value_95':value_95_replicating,
                                                                                'sphering': sphere_bool}, index=[ind]))

                            if match_or_rep_or_both.casefold() == "matching" or match_or_rep_or_both.casefold() == "both":
                                if enable_sphering.casefold() == "yes" or enable_sphering.casefold() == "both":
                                    sphere_bool = True
                                    matching_corr_sphere, null_matching_sphere, prop_95_matching_sphere, value_95_matching_sphere = utilssphering.calculate_percent_matching_MOA("", "", data_df=feature_selected_df)
                                    corr_matching_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                            batch_col: a_batch,
                                                                            'Assay_Plate_Barcode': a_plate,
                                                                            "num_features": len(feature_selected_df.columns),
                                                                            "dropout": str(dropout_group),
                                                                            "n_columns": len(col_list),
                                                                            'Matching':[matching_corr_sphere],
                                                                            'Null_Matching':[null_matching_sphere],
                                                                            'Percent_Matching':prop_95_matching_sphere,
                                                                            'Value_95':value_95_matching_sphere,
                                                                            'sphering': sphere_bool}, index=[ind]))
                                
                                if enable_sphering.casefold() == "no" or enable_sphering.casefold() == "both": 
                                    sphere_bool = False
                                    plate_df = utils.remove_negcon_empty_wells(feature_selected_df)
                                    matching_corr = list(utils.corr_between_perturbation_pairs(plate_df, 'Metadata_moa', 'Metadata_broad_sample'))
                                    null_matching = list(utils.corr_between_perturbation_non_pairs(plate_df, n_samples=n_samples, metadata_common=metadata_common, metadata_perturbation=metadata_perturbation))
                                    prop_95_matching, value_95_matching = utils.percent_score(null_matching, matching_corr, how='right')
                                    corr_matching_list.append(pd.DataFrame({'Vendor': a_vendor,
                                                                            batch_col: a_batch,
                                                                            'Assay_Plate_Barcode': a_plate,
                                                                            "num_features": len(feature_selected_df.columns),
                                                                            "dropout": str(dropout_group),
                                                                            "n_columns": len(col_list),
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


df_rep, df_match = create_moa_dataframe(experiment_metadata, profile_parent_dir, match_or_rep_or_both="both", enable_sphering="yes", dropout_cols=['BrightField'])

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

df_rep = add_total_cell_counts(df_rep, profile_parent_dir)
df_match = add_total_cell_counts(df_match, profile_parent_dir)

## Checkpoint save

if not os.path.isdir("checkpoints"):
    os.mkdir("checkpoints")

df_rep.to_csv("checkpoints/moa-replicating-brightfield-dropout.csv", index_label='index', index=False)

df_match.to_csv("checkpoints/moa-matching-brightfield-dropout.csv", index_label='index', index=False)

# Rename columns
df_rep = df_rep.rename(columns={"Value_95": "value_95_replicating"})
df_match = df_match.rename(columns={"Value_95": "value_95_matching"})

# Find the unique columns that are to be included in the merge
set1 = set(df_rep.columns)
set2 = set(df_match.columns)
rep_set = set1 - set2
merge_cols = ["Vendor", "Batch", "Assay_Plate_Barcode", "sphering", "dropout"] + list(rep_set)

match_rep_df = pd.merge(df_rep[merge_cols], df_match, how="inner")

# Drop distributions to reduce filesize
match_rep_df = match_rep_df.drop(["Null_Replicating", "Replicating", "Matching", "Null_Matching"], axis=1)

to_remove = ["Scope1_Yokogawa_US_20X_6Ch_BRO0117033", "Scope1_MolDev_20X_Adaptive"]

match_rep_df = match_rep_df[~match_rep_df["Batch"].isin(to_remove)]

match_rep_df.to_csv("checkpoints/match_rep_df-brightfield-DROP.csv", index=False)