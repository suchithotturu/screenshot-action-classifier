"""Advisory worker lock using the platform's standard library file locking."""
import errno
import os
import time


def exclusive(file, blocking=True):
    if os.name!="nt":
        import fcntl
        fcntl.flock(file,fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        return
    import msvcrt
    if os.fstat(file.fileno()).st_size==0:
        file.write("\0")
        file.flush()
    while True:
        file.seek(0)
        try:
            msvcrt.locking(file.fileno(),msvcrt.LK_NBLCK,1)
            return
        except OSError as exc:
            if exc.errno not in (errno.EACCES,errno.EAGAIN,errno.EDEADLK):
                raise
            if not blocking:
                raise BlockingIOError(exc.errno,"Worker already running") from exc
            time.sleep(0.1)
