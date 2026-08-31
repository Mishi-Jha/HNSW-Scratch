from time import perf_counter
def measure_latency(fn, queries, *args, **kwargs) -> float:
    before_time=perf_counter()
    for query in queries:
        fn(query,*args,**kwargs)
    after_time=perf_counter()
    total_time=after_time-before_time
    return total_time/len(queries)    