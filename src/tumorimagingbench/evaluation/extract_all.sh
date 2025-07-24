# Diagnostic
python luna_feature_extractor.py --output features/luna.pkl
python dlcs_feature_extractor.py --output features/dlcs.pkl

# Prognostic - lung
python nsclc_radiomics_feature_extractor.py --output features/nsclc_radiomics.pkl
python nsclc_radiogenomics_feature_extractor.py --output features/nsclc_radiogenomics.pkl

# Prognostic - others
python c4c_kits_feature_extractor.py --output features/c4c_kits.pkl
python colorectal_feature_extractor.py --output features/colorectal_liver_metastases.pkl