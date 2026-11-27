import pytest
from classifier.locking import exclusive


def test_worker_lock_prevents_parallel_worker_and_releases_on_close(tmp_path):
    path = tmp_path/"worker.lock"
    with path.open("a") as first:
        exclusive(first,blocking=False)
        with path.open("a") as second:
            with pytest.raises(BlockingIOError):
                exclusive(second,blocking=False)
    with path.open("a") as next_worker:
        exclusive(next_worker,blocking=False)
