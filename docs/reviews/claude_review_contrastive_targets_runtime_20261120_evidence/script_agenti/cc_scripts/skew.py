import sys, time, multiprocessing as mp
def work(start, q):
    while time.time() < start: time.sleep(0.001)
    import contrastive_release as R, contrastive_prepare as P
    t = time.time()
    R.verify_reviewed_sources("fbfde797b48644744c36547533acf0e8a7793535", P.source_digest())
    q.put((t - start, time.time() - start))
if __name__ == "__main__":
    for trial in range(3):
        q = mp.Queue(); start = time.time() + 2
        ps = [mp.Process(target=work, args=(start, q)) for _ in range(4)]
        [p.start() for p in ps]; [p.join() for p in ps]
        res = sorted(q.get() for _ in ps)
        ends = [e for _, e in res]
        print(f"trial {trial}: import+verify end times {[round(e,2) for e in ends]} s, skew {max(ends)-min(ends):.2f} s")
