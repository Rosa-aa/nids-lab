"""
utils.py
--------
Small helper functions reused across several modules.
"""

import gc


def print_section(title: str) -> None:
    """Prints a readable section header in the console (matches the notebook's '='*50 style)."""
    print("=" * 50)
    print(title)
    print("=" * 50)


def port_category(port) -> str:
    """
    Buckets a port number into one of three standard categories:
    - well_known:  0-1023
    - registered:  1024-49151
    - dynamic:     49152-65535
    """
    if port <= 1023:
        return "well_known"
    elif port <= 49151:
        return "registered"
    return "dynamic"


def check_ram() -> None:
    """Prints current RAM usage (useful for tracking memory limits on Colab)."""
    try:
        import psutil
    except ImportError:
        print("psutil is not installed — skipping RAM check.")
        return
    ram = psutil.virtual_memory()
    print(f"RAM used:      {ram.used / 1024**3:.1f} GB")
    print(f"RAM available: {ram.available / 1024**3:.1f} GB")


def free_memory(namespace: dict, var_names: list) -> None:
    """
    Deletes the given variable names from a namespace and runs gc.collect().
    Usage: free_memory(globals(), ['df', 'X_sample'])
    """
    for name in var_names:
        if name in namespace:
            del namespace[name]
    gc.collect()
