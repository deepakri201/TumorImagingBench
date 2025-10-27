"""
TumorImagingBench: A comprehensive framework for evaluating and comparing 
foundation model feature extractors for radiomics in medical imaging.
"""

__version__ = "0.1.4"
__author__ = "Suraj Pai"
__email__ = "bspai@bwh.harvard.edu"

# Import main modules with error handling
_modules_loaded = {}

try:
    from . import models
    _modules_loaded['models'] = True
except ImportError as e:
    print(f"Warning: models module not available due to missing dependencies: {e}")
    _modules_loaded['models'] = False

try:
    from . import evaluation
    _modules_loaded['evaluation'] = True
except ImportError as e:
    print(f"Warning: evaluation module not available due to missing dependencies: {e}")
    _modules_loaded['evaluation'] = False

def get_available_modules():
    """Return a list of successfully loaded modules."""
    return [module for module, loaded in _modules_loaded.items() if loaded]

def check_module_status():
    """Print the status of all modules."""
    print("TumorImagingBench Module Status:")
    for module, loaded in _modules_loaded.items():
        status = "✓ Available" if loaded else "✗ Not Available"
        print(f"  {module}: {status}")
    
    if hasattr(models, 'get_available_extractors'):
        available_extractors = models.get_available_extractors()
        if available_extractors:
            print(f"\nAvailable Extractors ({len(available_extractors)}):")
            for extractor in available_extractors:
                print(f"  ✓ {extractor}")
        else:
            print("\nNo extractors available (missing dependencies)")

__all__ = ['__version__', '__author__', '__email__', 'get_available_modules', 'check_module_status']
