import cProfile


def profiler(func):
    def wrapper(*args, **kwargs):
        profile = cProfile.Profile()
        profile.enable()
        result = func(*args, **kwargs)
        profile.disable()
        profile.print_stats(sort="cumtime")
        profile.dump_stats(f"{func.__name__}.prof")
        return result

    return wrapper
