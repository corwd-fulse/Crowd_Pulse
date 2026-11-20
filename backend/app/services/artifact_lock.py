"""Advisory OS locks release automatically when an API process exits/crashes."""
import os
from contextlib import contextmanager

@contextmanager
def acquire(path):
    stream=open(path,'a+b')
    try:
        stream.seek(0,2)
        if stream.tell()==0:stream.write(b'0');stream.flush()
        stream.seek(0)
        try:
            if os.name=='nt':
                import msvcrt
                msvcrt.locking(stream.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(stream.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
        except OSError:raise ValueError('Artifact is being built by another API process; retry shortly') from None
        yield
    finally:
        # Closing releases the lock; do not unlink the shared lock inode.
        stream.close()
