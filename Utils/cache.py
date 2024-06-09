import functools
import os
import pickle
import pandas as pd
import numpy as np
import hashlib

def format_filename(args, kwargs):
    """Format the function arguments into a string suitable for filenames."""
    
    def hash_parameter(param):
        if isinstance(param, pd.DataFrame):
            return f"DataFrame_{hashlib.md5(param.values.tobytes()).hexdigest()}"
        elif isinstance(param, np.ndarray):
            return f"ndarray_{hashlib.md5(param.tobytes()).hexdigest()}"
        else:
            return str(param)
    
    parts = [hash_parameter(arg) for arg in args] + [f"{k}={hash_parameter(v)}" for k, v in kwargs.items()]

    return "__".join(parts).replace("/", "_").replace("\\", "_")


def cache(directory):
    def decorator(save_as_parquet=False, version=0):
        def inner_decorator(func):
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # Create the function-specific directory if it does not exist
                func_dir = os.path.join(directory, func.__name__)
                version_dir = os.path.join(func_dir, f"v{version}")
                if not os.path.exists(version_dir):
                    os.makedirs(version_dir)

                # Create a unique filename based on the function arguments
                filename = format_filename(args, kwargs)
                if save_as_parquet:
                    cache_file = os.path.join(version_dir, filename + ".parquet")
                else:
                    cache_file = os.path.join(version_dir, filename + ".pkl")

                # Check if the result is already cached
                if os.path.exists(cache_file):
                    if save_as_parquet:
                        return pd.read_parquet(cache_file)
                    else:
                        with open(cache_file, 'rb') as f:
                            return pickle.load(f)

                # Compute the result and cache it
                result = func(*args, **kwargs)
                if save_as_parquet and isinstance(result, pd.DataFrame):
                    result.to_parquet(cache_file)
                else:
                    with open(cache_file, 'wb') as f:
                        pickle.dump(result, f)
                return result

            return wrapper
        return inner_decorator
    return decorator


if __name__ == "__main__":

    # Initialize the cache directory
    cache_decorator = cache('Utils/cache_directory')

    # Example usage with different functions
    @cache_decorator(save_as_parquet=True, version=2024_09_06_01)
    def expensive_computation_df(x, y):
        print("Performing expensive computation returning DataFrame...")
        return pd.DataFrame({'result': [x * y]})

    @cache_decorator(version=2024_09_06_01)
    def expensive_computation(x, y):
        print("Performing expensive computation...")
        return x * y

    @cache_decorator(save_as_parquet=True, version=2024_09_06_01)
    def computation_with_dataframe(df, multiplier):
        print("Performing computation with dataframe...")
        return df * multiplier

    @cache_decorator(version=2024_09_06_01)
    def computation_with_numpy_array(arr, multiplier):
        print("Performing computation with numpy array...")
        return arr * multiplier

    # Test the caching decorator
    print(expensive_computation_df(2, 3)) 
    print(expensive_computation(2, 3))

    # Create a sample dataframe
    df = pd.DataFrame({'a': [1, 2.11, 3], 'b': [4, 5, 6]})
    print(computation_with_dataframe(df, 2))

    # Create a sample numpy array
    arr = np.array([1, 2, 3.01])
    print(computation_with_numpy_array(arr, multiplier=2))
    print(computation_with_numpy_array(arr, 2))
