import time

def time_function(f):
    """
    A decorator that times a function evaluation and prints the time elapsed to the terminal.
    """
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = f(*args, **kwargs)
        end_time = time.time()
        print(f"Time elapsed for '{f.__name__}': {end_time - start_time} seconds")
        return result
    return wrapper