# Diagnostic
python luna_feature_extractor.py --output features/luna_pyramid.pkl
python dlcs_feature_extractor.py --output features/dlcs_pyramid.pkl

# Prognostic - lung
python nsclc_radiomics_feature_extractor.py --output features/nsclc_radiomics_pyramid.pkl
python nsclc_radiogenomics_feature_extractor.py --output features/nsclc_radiogenomic_pyramid.pkl

# Prognostic - others
python c4c_kits_feature_extractor.py --output features/c4c_kits_pyramid.pkl
python colorectal_feature_extractor.py --output features/colorectal_pyramid.pkl