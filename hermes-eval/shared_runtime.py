"""Process scheduling from qwen-eval-1016; no video/model inference logic."""
import multiprocessing as mp
from multiprocessing.connection import wait

def parallel_results(rows, args, worker_target=None):
    """Keep one outstanding sample per worker; yield completed rows immediately.

    Pipes bound queued work; process sentinels detect crashes without waiting
    forever for a result. Only the parent process writes output files.
    """
    if not rows:
        return
    ctx = mp.get_context('spawn')
    workers, active, assigned = [], {}, {}
    next_index = 0
    try:
        for rank in range(min(args.num_gpus, len(rows))):
            parent, child = ctx.Pipe()
            process = ctx.Process(target=worker_target, args=(child, args, rank), name=f'qwen-gpu-{rank}')
            try:
                process.start()
            except BaseException:
                parent.close()
                child.close()
                raise
            child.close()
            workers.append((process, parent))
            active[parent] = (process, rank)
            assigned[parent] = None
        while active:
            ready = wait(list(active) + [process.sentinel for process, _ in active.values()])
            # Drain messages before checking sentinels so fatal errors retain their details.
            for connection in list(active):
                if connection not in ready:
                    continue
                process, rank = active[connection]
                try:
                    kind, payload = connection.recv()
                except (EOFError, OSError) as exc:
                    raise RuntimeError(f'GPU worker {rank} exited without a result') from exc
                if kind == 'fatal':
                    raise RuntimeError(f'GPU worker {rank} failed: {payload}')
                if kind == 'result':
                    if payload['prediction']['row_index'] != assigned[connection]:
                        raise RuntimeError(f'GPU worker {rank} returned an unexpected row')
                    yield payload
                elif kind != 'ready' or assigned[connection] is not None:
                    raise RuntimeError(f'GPU worker {rank} sent unexpected message: {kind}')
                if next_index < len(rows):
                    assigned[connection] = next_index
                    connection.send((next_index, rows[next_index]))
                    next_index += 1
                else:
                    connection.send(None)
                    del active[connection]
            for connection, (process, rank) in active.items():
                if process.sentinel in ready:
                    # A message may have arrived just after wait returned.
                    if connection.poll():
                        continue
                    raise RuntimeError(f'GPU worker {rank} exited unexpectedly (exitcode={process.exitcode})')
    finally:
        # Also runs on Ctrl-C, write failures and worker initialization failures.
        for process, connection in workers:
            connection.close()
            if process.is_alive():
                process.terminate()
        for process, _ in workers:
            process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
