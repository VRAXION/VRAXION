"""
v22_log.py — Live logging for ProcessPoolExecutor scripts
==========================================================
Solves: Windows stdout buffering, __file__ in subprocesses, silent exceptions.

Usage:
    from v22_log import live_log, log_msg

    def worker(args, log_q=None):
        # ... work ...
        log_msg(log_q, f"config={name} acc={acc:.1f}%")
        return result

    with live_log('my_script') as (log_q, log_path):
        with ProcessPoolExecutor(max_workers=N) as pool:
            futures = [pool.submit(worker, cfg, log_q) for cfg in configs]
            for fut in as_completed(futures):
                results.append(fut.result())
"""

import multiprocessing
import threading
import queue
import sys
import os
import time
from datetime import datetime
from contextlib import contextmanager


# Log directory: always absolute, never __file__-relative
LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                       'tests', 'logs')


def _timestamp():
    return datetime.now().strftime('%H:%M:%S')


def _listener(q, log_path):
    """Listener thread: reads from queue, writes to file + stdout."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', buffering=1, encoding='utf-8') as f:
        # Header
        header = f"[{_timestamp()}] === LOG STARTED === {log_path}\n"
        f.write(header)
        f.flush()
        sys.stdout.write(header)
        sys.stdout.flush()

        while True:
            try:
                msg = q.get(timeout=1.0)
            except queue.Empty:
                continue
            if msg is None:  # sentinel
                footer = f"[{_timestamp()}] === LOG ENDED ===\n"
                f.write(footer)
                f.flush()
                sys.stdout.write(footer)
                sys.stdout.flush()
                break
            line = msg if msg.endswith('\n') else msg + '\n'
            f.write(line)
            f.flush()
            sys.stdout.write(line)
            sys.stdout.flush()


def log_msg(q, msg):
    """Send a log message from any process. Safe to call from workers."""
    if q is None:
        # Fallback: direct print (for scripts not yet retrofitted)
        print(msg, flush=True)
        return
    stamped = f"[{_timestamp()}] {msg}"
    try:
        q.put_nowait(stamped)
    except Exception:
        # Last resort: print directly
        print(stamped, flush=True)


def start_live_log(script_name):
    """Start live logging. Returns (queue, log_path, listener_thread, manager).

    Uses Manager().Queue() instead of multiprocessing.Queue() because
    Windows 'spawn' context cannot pickle raw Queue objects passed as
    arguments to ProcessPoolExecutor.submit(). Manager queues are proxies
    that work across process boundaries on all platforms.
    """
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(LOG_DIR, f"{script_name}_{ts}.log")
    manager = multiprocessing.Manager()
    q = manager.Queue()
    t = threading.Thread(target=_listener, args=(q, log_path), daemon=True)
    t.start()
    return q, log_path, t, manager


def stop_live_log(q, listener_thread, manager, timeout=5.0):
    """Stop live logging. Sends sentinel and waits for listener to finish."""
    q.put(None)
    listener_thread.join(timeout=timeout)
    manager.shutdown()


@contextmanager
def live_log(script_name):
    """Context manager for live logging.

    Usage:
        with live_log('my_script') as (log_q, log_path):
            # log_q: pass to workers via pool.submit(fn, arg, log_q)
            # log_path: tell user where to tail -f
    """
    q, log_path, t, mgr = start_live_log(script_name)
    try:
        yield q, log_path
    finally:
        stop_live_log(q, t, mgr)
