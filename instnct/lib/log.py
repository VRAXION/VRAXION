"""
Live logging for ProcessPoolExecutor scripts.
Uses Manager().Queue() + listener thread for cross-process logging on Windows.

Usage:
    from lib.log import live_log, log_msg

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
from datetime import datetime
from contextlib import contextmanager


DEFAULT_LOG_DIR = os.path.join(os.getcwd(), 'logs')


def _timestamp():
    return datetime.now().strftime('%H:%M:%S')


def _listener(q, log_path):
    """Listener thread: drains queue, writes to file + stdout."""
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    with open(log_path, 'w', buffering=1, encoding='utf-8') as f:
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
            if msg is None:
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
        print(msg, flush=True)
        return
    stamped = f"[{_timestamp()}] {msg}"
    try:
        q.put_nowait(stamped)
    except Exception:
        print(stamped, flush=True)


def start_live_log(script_name, log_dir=None):
    """Start live logging. Returns (queue, log_path, listener_thread, manager)."""
    log_dir = log_dir or DEFAULT_LOG_DIR
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = os.path.join(log_dir, f"{script_name}_{ts}.log")
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
def live_log(script_name, log_dir=None):
    """Context manager for live logging.

    Args:
        script_name: prefix for the log file
        log_dir: directory for log files (default: ./logs/)

    Yields:
        (log_q, log_path) — pass log_q to workers via pool.submit()
    """
    q, log_path, t, mgr = start_live_log(script_name, log_dir)
    try:
        yield q, log_path
    finally:
        stop_live_log(q, t, mgr)
