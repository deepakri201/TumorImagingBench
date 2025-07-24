"""
TumorImagingBench Models Module

This module contains various foundation model feature extractors.
Each extractor is imported conditionally based on available dependencies.
"""

# Dictionary to track available extractors
AVAILABLE_EXTRACTORS = {}

# Try to import each extractor, handling missing dependencies gracefully
try:
    from .ct_clip_vit import CTClipVitExtractor
    AVAILABLE_EXTRACTORS['CTClipVitExtractor'] = CTClipVitExtractor
except ImportError as e:
    print(f"Warning: CTClipVitExtractor not available due to missing dependencies: {e}")

try:
    from .ct_fm import CTFMExtractor
    AVAILABLE_EXTRACTORS['CTFMExtractor'] = CTFMExtractor
except ImportError as e:
    print(f"Warning: CTFMExtractor not available due to missing dependencies: {e}")

try:
    from .fmcib import FMCIBExtractor
    AVAILABLE_EXTRACTORS['FMCIBExtractor'] = FMCIBExtractor
except ImportError as e:
    print(f"Warning: FMCIBExtractor not available due to missing dependencies: {e}")

try:
    from .medimageinsight import MedImageInsightExtractor
    AVAILABLE_EXTRACTORS['MedImageInsightExtractor'] = MedImageInsightExtractor
except ImportError as e:
    print(f"Warning: MedImageInsightExtractor not available due to missing dependencies: {e}")

try:
    from .merlin import MerlinExtractor
    AVAILABLE_EXTRACTORS['MerlinExtractor'] = MerlinExtractor
except ImportError as e:
    print(f"Warning: MerlinExtractor not available due to missing dependencies: {e}")

try:
    from .modelsgen import ModelsGenExtractor
    AVAILABLE_EXTRACTORS['ModelsGenExtractor'] = ModelsGenExtractor
except ImportError as e:
    print(f"Warning: ModelsGenExtractor not available due to missing dependencies: {e}")

try:
    from .pasta import PASTAExtractor
    AVAILABLE_EXTRACTORS['PASTAExtractor'] = PASTAExtractor
except ImportError as e:
    print(f"Warning: PASTAExtractor not available due to missing dependencies: {e}")

try:
    from .suprem import SUPREMExtractor
    AVAILABLE_EXTRACTORS['SUPREMExtractor'] = SUPREMExtractor
except ImportError as e:
    print(f"Warning: SUPREMExtractor not available due to missing dependencies: {e}")

try:
    from .vista3d import VISTA3DExtractor
    AVAILABLE_EXTRACTORS['VISTA3DExtractor'] = VISTA3DExtractor
except ImportError as e:
    print(f"Warning: VISTA3DExtractor not available due to missing dependencies: {e}")

try:
    from .voco import VocoExtractor
    AVAILABLE_EXTRACTORS['VocoExtractor'] = VocoExtractor
except ImportError as e:
    print(f"Warning: VocoExtractor not available due to missing dependencies: {e}")

def get_available_extractors():
    """Return a list of available extractor classes."""
    return list(AVAILABLE_EXTRACTORS.keys())

def get_extractor(name):
    """Get an extractor class by name."""
    if name in AVAILABLE_EXTRACTORS:
        return AVAILABLE_EXTRACTORS[name]
    else:
        raise ValueError(f"Extractor '{name}' is not available. Available extractors: {get_available_extractors()}")

# Make available extractors accessible at module level
__all__ = ['AVAILABLE_EXTRACTORS', 'get_available_extractors', 'get_extractor'] + list(AVAILABLE_EXTRACTORS.keys())