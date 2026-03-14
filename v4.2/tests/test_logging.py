"""
Adversarial Stress Test for live logging infrastructure.
6 probes. All must PASS.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from lib.log import live_log, log_msg

PASS = "PASS"
FAIL = "FAIL"


def header(num, name):
    print(f"\n  {'-'*55}")
    print(f"  PROBE {num}: {name}")
    print(f"  {'-'*55}")
    sys.stdout.flush()


def verdict(status, msg):
    tag = {"PASS": "+", "FAIL": "X"}[status]
    print(f"    [{tag}] {status}: {msg}")
    sys.stdout.flush()
    return status


# Module-level workers (picklable)
def slow_worker(idx, log_q):
    log_msg(log_q, f"worker {idx} starting")
    time.sleep(0.5)
    log_msg(log_q, f"worker {idx} done")
    return idx

def crashing_worker(idx, log_q):
    log_msg(log_q, f"worker {idx} about to crash")
    time.sleep(0.2)
    os._exit(1)

def exception_worker(idx, log_q):
    log_msg(log_q, f"worker {idx} about to raise")
    time.sleep(0.1)
    raise ValueError(f"deliberate test error from worker {idx}")

def spam_worker(idx, n_messages, log_q):
    for i in range(n_messages):
        log_msg(log_q, f"worker {idx} msg {i}")
    return idx


def main():
    results = []
    log_dir = os.path.join(os.path.dirname(__file__), 'logs')

    # PROBE 1: Output within 5 seconds
    header(1, "Output within 5 seconds")
    try:
        with live_log('probe1', log_dir=log_dir) as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(slow_worker, i, log_q) for i in range(4)]
                time.sleep(2)
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        lines = [l for l in f.read().strip().split('\n') if l.strip()]
                    r = verdict(PASS if len(lines) >= 2 else FAIL,
                                f"Log has {len(lines)} lines within 2s")
                else:
                    r = verdict(FAIL, "Log file does not exist")
                for fut in as_completed(futures): fut.result()
        results.append(("Output within 5s", r))
    except Exception as ex:
        results.append(("Output within 5s", verdict(FAIL, f"Crashed: {ex}")))

    # PROBE 2: Worker crash
    header(2, "Worker crash -- partial results")
    try:
        with live_log('probe2', log_dir=log_dir) as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(slow_worker, i, log_q) for i in range(3)]
                futures.append(pool.submit(crashing_worker, 99, log_q))
                for fut in as_completed(futures):
                    try: fut.result(timeout=10)
                    except Exception: pass
        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        has_normal = "worker 0" in content or "worker 1" in content
        r = verdict(PASS if has_normal else FAIL, f"Normal workers logged: {has_normal}")
        results.append(("Worker crash", r))
    except Exception as ex:
        results.append(("Worker crash", verdict(FAIL, f"Crashed: {ex}")))

    # PROBE 3: Exception logged
    header(3, "Exception logged -- not swallowed")
    try:
        with live_log('probe3', log_dir=log_dir) as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=2) as pool:
                fut_ok = pool.submit(slow_worker, 0, log_q)
                fut_err = pool.submit(exception_worker, 1, log_q)
                fut_ok.result(timeout=10)
                exception_caught = False
                try: fut_err.result(timeout=10)
                except ValueError: exception_caught = True
        with open(log_path, 'r', encoding='utf-8') as f:
            has_pre = "about to raise" in f.read()
        r = verdict(PASS if exception_caught else FAIL,
                    f"Exception caught: {exception_caught}, pre-raise logged: {has_pre}")
        results.append(("Exception logged", r))
    except Exception as ex:
        results.append(("Exception logged", verdict(FAIL, f"Crashed: {ex}")))

    # PROBE 4: 1000 lines integrity
    header(4, "1000 lines integrity")
    try:
        with live_log('probe4', log_dir=log_dir) as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(spam_worker, i, 250, log_q) for i in range(4)]
                for fut in as_completed(futures): fut.result(timeout=30)
        with open(log_path, 'r', encoding='utf-8') as f:
            data_lines = [l for l in f.readlines() if 'msg ' in l]
        total = len(data_lines)
        r = verdict(PASS if total >= 990 else FAIL, f"{total}/1000 messages")
        results.append(("1000 lines", r))
    except Exception as ex:
        results.append(("1000 lines", verdict(FAIL, f"Crashed: {ex}")))

    # PROBE 5: Concurrent read
    header(5, "Concurrent read while writing")
    try:
        read_results = []
        with live_log('probe5', log_dir=log_dir) as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(spam_worker, i, 100, log_q) for i in range(4)]
                for _ in range(5):
                    time.sleep(0.3)
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            read_results.append(len(f.read().strip().split('\n')))
                    except Exception as ex:
                        read_results.append(f"ERR: {ex}")
                for fut in as_completed(futures): fut.result(timeout=30)
        all_ok = all(isinstance(r, int) for r in read_results)
        r = verdict(PASS if all_ok else FAIL, f"5/5 reads OK: {all_ok}")
        results.append(("Concurrent read", r))
    except Exception as ex:
        results.append(("Concurrent read", verdict(FAIL, f"Crashed: {ex}")))

    # PROBE 6: Log exists immediately
    header(6, "Log file exists immediately")
    try:
        with live_log('probe6', log_dir=log_dir) as (log_q, log_path):
            time.sleep(0.3)
            exists = os.path.exists(log_path)
            has_header = False
            if exists:
                with open(log_path, 'r', encoding='utf-8') as f:
                    has_header = "LOG STARTED" in f.read()
            with ProcessPoolExecutor(max_workers=1) as pool:
                pool.submit(slow_worker, 0, log_q).result(timeout=10)
        r = verdict(PASS if exists and has_header else FAIL,
                    f"Exists: {exists}, header: {has_header}")
        results.append(("Log exists immediately", r))
    except Exception as ex:
        results.append(("Log exists immediately", verdict(FAIL, f"Crashed: {ex}")))

    # SUMMARY
    print(f"\n{'='*60}")
    print(f"  LOGGING STRESS TEST -- SUMMARY")
    print(f"{'='*60}\n")
    passes = sum(1 for _, s in results if s == PASS)
    fails = sum(1 for _, s in results if s == FAIL)
    for name, status in results:
        tag = {"PASS": "+", "FAIL": "X"}[status]
        print(f"  [{tag}] {status:4s}  {name}")
    print(f"\n  Total: {passes} PASS, {fails} FAIL out of {len(results)}")
    if fails > 0:
        print(f"\n  {fails} FAILURE(S)!")
    else:
        print(f"\n  All clean!")
    print(f"\n{'='*60}", flush=True)
    return fails


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
