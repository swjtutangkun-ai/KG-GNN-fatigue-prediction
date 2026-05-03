# Code File Map

This directory keeps only the scripts used by the manuscript results, renamed by the corresponding manuscript section.

## Shared Utilities

- `config_single.py`: shared paths, constants, feature definitions, and ID parsing.
- `plot_style.py`: shared plotting style for analysis figures.

## Section 2.2 Stress-Field Graph Construction

- `02_02_01_node_feature_extraction.py`: extract top-N weld-toe nodes by principal stress.
- `02_02_02_graph_topology_construction.py`: build specimen-level k-NN graphs and hotspot descriptors.
- `02_02_03_graph_preprocessing_and_augmentation.py`: preprocess graph tensors and construct augmented data.

## Section 4 Training Strategy And Results

- `04_01_hyperparameter_tpe_search.py`: Bayesian/TPE hyperparameter search used for the final configuration.
- `04_02_train_kggnn_overall_performance.py`: final KG-GNN cross-validation training.
- `04_02_analyze_overall_performance.py`: overall prediction, S-N curve, and training-dynamics analysis.
- `04_02_03_analyze_dseq_posterior_consistency.py`: posterior consistency between inferred Delta Seq and top-5% FEA stress statistics.
- `04_03_01_baseline_comparison_raw_input.py`: baseline comparison using flattened raw node input.
- `04_03_01_baseline_comparison_handcrafted_input.py`: baseline comparison using handcrafted hotspot/statistical features.
- `04_03_02_analyze_baseline_comparison.py`: summary tables and figures for baseline comparison.

## Section 5 Sensitivity, Ablation, And Interpretability

- `05_01_analyze_hyperparameter_sensitivity.py`: hyperparameter sensitivity analysis from TPE results.
- `05_01_node_sampling_size_sweep.py`: top-N node sampling sensitivity sweep.
- `05_02_run_ablation_study.py`: ablation experiments.
- `05_02_analyze_ablation_study.py`: ablation summary tables and figures.
- `05_03_analyze_gatv2_attention_interpretability.py`: GATv2 attention interpretability analysis.

## Section 6 Extrapolation And Generalization

- `06_01_leave_one_out_extrapolation.py`: leave-one-out extrapolation.
- `06_01_data_efficiency_loo.py`: LOO data-efficiency analysis with reduced training fractions.
- `06_02_external_bw_node_feature_extraction.py`: external BW dataset node extraction.
- `06_02_external_bw_graph_construction.py`: external BW graph construction.
- `06_02_external_bw_graph_preprocessing.py`: external BW preprocessing using training-set standardization parameters.
- `06_02_external_bw_train_and_predict.py`: full-data training and external BW prediction.
- `06_02_external_bw_plot_analysis.py`: external BW result tables and figures.

## Not Included

- `2_preprocess_single.py`: superseded by `2_build_aug_data.py`, which keeps coordinate fields required by the baseline comparison.
- `4_4_generalization.py`: contains LOJO/LOCO routines that are not part of the current manuscript's reported section-6 workflow.
- `4_4b_sensitivity.py`: fixed-CV data-fraction sensitivity, replaced in the manuscript by LOO data-efficiency analysis.
- `analyze_dSeq_posterior.py`: superseded by the top-5% posterior-consistency analysis used in the manuscript.
