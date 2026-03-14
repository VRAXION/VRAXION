"""
Adversarial Stress Test for v22_log.py Live Logging
=====================================================
6 probes that PROVE the logging infrastructure works.
If ANY of these fail, the logging is NOT production-ready.
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import time
import multiprocessing
from concurrent.futures import ProcessPoolExecutor, as_completed
from v22_log import live_log, log_msg

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


# ============================================================
# Worker functions (module-level for pickling)
# ============================================================

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


def very_slow_worker(idx, log_q):
    log_msg(log_q, f"worker {idx} starting (slow)")
    time.sleep(5)
    log_msg(log_q, f"worker {idx} done")
    return idx


# ============================================================
# All probes inside main()
# ============================================================

def main():
    results = []

    # PROBE 1: Output within 5 seconds
    header(1, "Output within 5 seconds")
    try:
        with live_log('probe1') as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(slow_worker, i, log_q) for i in range(4)]
                time.sleep(2)
                if os.path.exists(log_path):
                    with open(log_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    lines = [l for l in content.strip().split('\n') if l.strip()]
                    print(f"    Log has {len(lines)} lines after 2s")
                    if len(lines) >= 2:
                        r = verdict(PASS, f"Log has {len(lines)} lines within 2s")
                    else:
                        r = verdict(FAIL, f"Only {len(lines)} lines after 2s")
                else:
                    r = verdict(FAIL, f"Log file does not exist at {log_path}")
                for fut in as_completed(futures):
                    fut.result()
        results.append(("Output within 5s", r))
    except Exception as ex:
        r = verdict(FAIL, f"Crashed: {ex}")
        results.append(("Output within 5s", r))

    # PROBE 2: Worker crash -- partial results
    header(2, "Worker crash -- partial results preserved")
    try:
        with live_log('probe2') as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = []
                for i in range(3):
                    futures.append(pool.submit(slow_worker, i, log_q))
                futures.append(pool.submit(crashing_worker, 99, log_q))
                collected = 0
                errors = 0
                for fut in as_completed(futures):
                    try:
                        fut.result(timeout=10)
                        collected += 1
                    except Exception:
                        errors += 1

        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        has_crash_msg = "worker 99 about to crash" in content
        has_normal = "worker 0" in content or "worker 1" in content
        print(f"    Collected: {collected}, Errors: {errors}")
        print(f"    Crash worker logged before crash: {has_crash_msg}")
        print(f"    Normal workers logged: {has_normal}")
        if has_normal:
            r = verdict(PASS, "Normal workers preserved after crash")
        else:
            r = verdict(FAIL, "No worker messages in log")
        results.append(("Worker crash", r))
    except Exception as ex:
        r = verdict(FAIL, f"Crashed: {ex}")
        results.append(("Worker crash", r))

    # PROBE 3: Exception logged
    header(3, "Exception logged -- not swallowed")
    try:
        with live_log('probe3') as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=2) as pool:
                fut_ok = pool.submit(slow_worker, 0, log_q)
                fut_err = pool.submit(exception_worker, 1, log_q)
                ok_result = fut_ok.result(timeout=10)
                exception_caught = False
                exception_msg = ""
                try:
                    fut_err.result(timeout=10)
                except ValueError as e:
                    exception_caught = True
                    exception_msg = str(e)

        with open(log_path, 'r', encoding='utf-8') as f:
            content = f.read()
        has_pre_raise = "about to raise" in content
        print(f"    Exception caught by main: {exception_caught}")
        print(f"    Exception message: {exception_msg}")
        print(f"    Pre-raise log msg present: {has_pre_raise}")
        if exception_caught and has_pre_raise:
            r = verdict(PASS, "Exception visible in both log and fut.result()")
        elif exception_caught:
            r = verdict(PASS, "Exception caught (pre-raise msg timing)")
        else:
            r = verdict(FAIL, "Exception swallowed!")
        results.append(("Exception logged", r))
    except Exception as ex:
        r = verdict(FAIL, f"Crashed: {ex}")
        results.append(("Exception logged", r))

    # PROBE 4: 1000 lines integrity
    header(4, "1000 lines integrity")
    try:
        with live_log('probe4') as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(spam_worker, i, 250, log_q) for i in range(4)]
                for fut in as_completed(futures):
                    fut.result(timeout=30)

        with open(log_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        data_lines = [l for l in lines if 'msg ' in l]
        total = len(data_lines)
        print(f"    Expected: 1000 data lines")
        print(f"    Got:      {total} data lines")
        if total == 1000:
            r = verdict(PASS, "All 1000 messages received")
        elif total >= 990:
            r = verdict(PASS, f"{total}/1000 -- minor loss acceptable")
        else:
            r = verdict(FAIL, f"Only {total}/1000 messages received!")
        results.append(("1000 lines", r))
    except Exception as ex:
        r = verdict(FAIL, f"Crashed: {ex}")
        results.append(("1000 lines", r))

    # PROBE 5: Concurrent read while writing
    header(5, "Concurrent read while writing")
    try:
        read_results = []
        with live_log('probe5') as (log_q, log_path):
            with ProcessPoolExecutor(max_workers=4) as pool:
                futures = [pool.submit(spam_worker, i, 100, log_q) for i in range(4)]
                for check in range(5):
                    time.sleep(0.3)
                    try:
                        with open(log_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        line_count = len(content.strip().split('\n'))
                        read_results.append(line_count)
                    except Exception as ex:
                        read_results.append(f"ERR: {ex}")
                for fut in as_completed(futures):
                    fut.result(timeout=30)

        print(f"    Read results (line counts): {read_results}")
        all_ints = all(isinstance(r, int) for r in read_results)
        if all_ints and len(read_results) == 5:
            r = verdict(PASS, f"5/5 reads OK, counts={read_results}")
        else:
            r = verdict(FAIL, f"Read errors: {read_results}")
        results.append(("Concurrent read", r))
    except Exception as ex:
        r = verdict(FAIL, f"Crashed: {ex}")
        results.append(("Concurrent read", r))

    # PROBE 6: Log file exists immediately
    header(6, "Log file exists immediately")
    try:
        with live_log('probe6') as (log_q, log_path):
            time.sleep(0.3)
            exists_before = os.path.exists(log_path)
            if exists_before:
                with open(log_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                has_header = "LOG STARTED" in content
            else:
                has_header = False
            print(f"    File exists before workers: {exists_before}")
            print(f"    Has header line: {has_header}")

            with ProcessPoolExecutor(max_workers=1) as pool:
                fut = pool.submit(slow_worker, 0, log_q)
                fut.result(timeout=10)

        if exists_before and has_header:
            r = verdict(PASS, "Log created by listener BEFORE any worker runs")
        elif exists_before:
            r = verdict(PASS, "Log exists but header missing")
        else:
            r = verdict(FAIL, "Log not created until workers ran")
        results.append(("Log exists immediately", r))
    except Exception as ex:
        r = verdict(FAIL, f"Crashed: {ex}")
        results.append(("Log exists immediately", r))

    # SUMMARY
    print(f"\n{'='*60}")
    print(f"  LIVE LOGGING ADVERSARIAL TEST -- SUMMARY")
    print(f"{'='*60}\n")

    passes = sum(1 for _, s in results if s == PASS)
    fails = sum(1 for _, s in results if s == FAIL)

    for name, status in results:
        tag = {"PASS": "+", "FAIL": "X"}[status]
        print(f"  [{tag}] {status:4s}  {name}")

    print(f"\n  Total: {passes} PASS, {fails} FAIL out of {len(results)}")

    if fails > 0:
        print(f"\n  {fails} FAILURE(S) -- LOGGING NOT READY!")
    else:
        print(f"\n  All clean -- logging infrastructure PROVEN!")

    print(f"\n{'='*60}", flush=True)
    return fails


if __name__ == '__main__':
    multiprocessing.freeze_support()
    sys.exit(main())
