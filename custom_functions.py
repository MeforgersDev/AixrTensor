# custom_functions.py
def example_custom_func(tensor, ram_usage):
    if ram_usage > 70:
        tensor.save_to_ram()
        tensor.to('gpu')
    else:
        tensor.load_from_ram()
        tensor.to('cpu')

def example_important_func(data):
    # A Basic Function: if float more than important once float accept that float
    threshold = 1.0
    return np.max(data) > threshold
