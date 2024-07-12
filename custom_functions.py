# custom_functions.py
import numpy as np

def example_custom_func(tensor, ram_usage):
    """
    Example custom function to manage tensor operations based on RAM usage.
    Moves the tensor to GPU if RAM usage is high, otherwise keeps it on CPU.
    """
    if ram_usage > 70:
        tensor.save_to_ram()
        tensor.to('gpu')
    else:
        tensor.load_from_ram()
        tensor.to('cpu')

def example_important_func(data):
    """
    Example function to determine if the data is important.
    Marks the data as important if any element exceeds a threshold value.
    """
    threshold = 1.0
    return np.max(data) > threshold

def example_detailed_processing_func(data):
    """
    Example function to perform detailed processing on important data.
    For example, this function could apply a more complex transformation or analysis.
    """
    # Perform a detailed processing task
    # This is a placeholder for any complex operation you want to perform
    if isinstance(data, np.ndarray):
        print("Performing detailed processing on important data...")
        # Example detailed processing: normalize the data
        data -= np.mean(data)
        data /= np.std(data)
    else:
        import jax.numpy as jnp
        print("Performing detailed processing on important TPU data...")
        data -= jnp.mean(data)
        data /= jnp.std(data)
