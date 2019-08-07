import sys
import time
import multiprocessing as mp

def add(arg):
    time.sleep(arg[0])
    return arg[0] + arg[1]

if __name__ == "__main__":
    array = [(1,2), (2,3), (3,4)]
    pool = mp.Pool(processes=mp.cpu_count())
    results = pool.imap_unordered(add, array)
    pool.close() # No more work -> start!

    loading = ['|','/','-','\\']
    i = 0
    while (True): # show progress ...
        progress = results._index
        time.sleep(0.1)
        i = (i + 1) % len(loading)
        sys.stdout.write(f'\r{loading[i]} {progress} of {len(array)} finished')
        if (progress >= len(array)): break
    sys.stdout.write('\n')

    results = [i for i in results] # convert to list
    print(results) # [3, 5, 7]