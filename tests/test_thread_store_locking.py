"""One sqlite connection shared across threads, so every statement needs the store's lock.

Live (2026-10-17): `ThreadStore` opened one connection with `check_same_thread=False` and let only
writers take the RLock. `open_segment` committed under the lock and then re-read the row *outside*
it, while `segment`, `segments`, `active_segment` and `provider_ordinal` read with no lock at all.
A stress run -- 8 threads, 80 compress-shaped iterations each -- failed 14 of 640 times with
`AttributeError: 'NoneType' object has no attribute 'segment_id'` and `InterfaceError: bad parameter
or other API misuse`. That is the layer holding the user's real conversation, and the compress path
is how they get the "unlimited context" feel, so a 2% failure rate there is a reliability hole, not
a corner case.

After wrapping every connection-touching method: 1920 of 1920 iterations clean over three runs.
"""

from __future__ import annotations

import inspect
import threading

from backend.thread.store import ThreadStore

# Helpers that never touch the connection themselves, and __init__/close which manage the lock by hand.
UNWRAPPED_OK = {"__init__", "close", "_seg", "_msg"}


def _touches_the_connection(func) -> bool:
    src = inspect.getsource(func)
    return "self.db." in src


def test_every_method_that_touches_the_connection_holds_the_lock():
    unwrapped = []
    for name, member in vars(ThreadStore).items():
        if not callable(member) or name in UNWRAPPED_OK:
            continue
        if _touches_the_connection(member) and not hasattr(member, "__wrapped__"):
            unwrapped.append(name)
    assert not unwrapped, f"these read or write sqlite without the lock: {unwrapped}"


def test_concurrent_segments_never_come_back_none(tmp_path):
    """The shape that actually broke: one thread opening a segment while another reads them."""
    store = ThreadStore(tmp_path / "race.db")
    tid = store.create_thread("housing thread")
    failures: list[str] = []
    done = 0
    guard = threading.Lock()

    def worker(index: int) -> None:
        nonlocal done
        for i in range(25):
            try:
                seg = store.open_segment(tid, f"prov{(index + i) % 4}", reason="stress")
                if seg is None or not seg.segment_id:
                    failures.append(f"open_segment returned {seg!r}")
                    continue
                if store.segment(seg.segment_id) is None:
                    failures.append("the row just committed was not readable")
                store.add_message(tid, "user", f"q{index}-{i}", provider=f"prov{index % 4}")
                store.active_segment(tid)
                store.segments(tid)
            except Exception as exc:  # noqa: BLE001 -- the race surfaced as a raised error, not a None
                failures.append(f"{type(exc).__name__}: {exc}")
            with guard:
                done += 1

    threads = [threading.Thread(target=worker, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert done == 150
    assert not failures, f"{len(failures)} of {done} iterations failed, e.g. {failures[:3]}"
    store.close()
