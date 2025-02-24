from fmcib.run import get_features

features = get_features("/home/suraj/Repositories/FM-extractors-radiomics/data/eval/nsclc_radiomics/annotations.csv", batch_size=32)
features.to_csv("original.csv")