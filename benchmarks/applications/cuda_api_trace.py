"""Optional CUPTI API instrumentation for audits, never production numerics."""

import ctypes as ct
import struct
import time
from importlib.metadata import distribution


class CallbackData(ct.Structure):
    _fields_ = [
        ("site", ct.c_int),
        ("name", ct.c_char_p),
        ("params", ct.c_void_p),
        ("result", ct.c_void_p),
        ("symbol", ct.c_char_p),
        ("context", ct.c_void_p),
        ("context_uid", ct.c_uint32),
        ("correlation_data", ct.c_void_p),
        ("correlation", ct.c_uint32),
    ]


class RuntimeCopy(ct.Structure):
    _fields_ = [
        ("dst", ct.c_void_p),
        ("src", ct.c_void_p),
        ("count", ct.c_size_t),
        ("kind", ct.c_int),
        ("stream", ct.c_void_p),
    ]


class DriverCopy(ct.Structure):
    _fields_ = [
        ("dst", ct.c_uint64),
        ("src", ct.c_void_p),
        ("count", ct.c_size_t),
        ("stream", ct.c_void_p),
    ]


class ApiTrace:
    """Driver/runtime callbacks; timings include tracing overhead.

    Callback fields follow the installed CUPTI 12 header. No CUDA/CUPTI call is
    made from a callback. Small HtoD source bytes are read only at the API entry
    while the supplied source pointer is live. Graph replays execute no Python
    callback nodes: this instruments host API calls, not device graph nodes.
    """

    def __init__(self):
        dist = distribution("nvidia-cuda-cupti-cu12")
        library = next(dist.locate_file(p) for p in dist.files if str(p).endswith("libcupti.so.12"))
        self.lib = ct.CDLL(str(library))
        self.handle = ct.c_void_p()
        self.phase = None
        self.events = []
        self.errors = []
        self.callback = ct.CFUNCTYPE(None, ct.c_void_p, ct.c_int, ct.c_uint32, ct.c_void_p)(
            self._event
        )
        self.lib.cuptiSubscribe.argtypes = [ct.POINTER(ct.c_void_p), ct.c_void_p, ct.c_void_p]
        self.lib.cuptiEnableDomain.argtypes = [ct.c_uint32, ct.c_void_p, ct.c_int]
        self.lib.cuptiUnsubscribe.argtypes = [ct.c_void_p]
        self._check(self.lib.cuptiSubscribe(ct.byref(self.handle), self.callback, None))
        try:
            for domain in (1, 2):
                self._check(self.lib.cuptiEnableDomain(1, self.handle, domain))
        except Exception:
            self.close()
            raise

    @staticmethod
    def _check(status):
        if status:
            raise RuntimeError(f"CUPTI API status {status}")

    def _event(self, userdata, domain, callback_id, pointer):
        if self.phase is None or domain not in (1, 2):
            return
        try:
            data = ct.cast(pointer, ct.POINTER(CallbackData)).contents
            row = dict(
                phase=self.phase,
                domain=domain,
                name=data.name.decode(),
                site=data.site,
                correlation=data.correlation,
                time_ns=time.perf_counter_ns(),
            )
            if data.site == 0 and data.params:
                copy = None
                if domain == 2 and row["name"].startswith("cudaMemcpyAsync"):
                    candidate = ct.cast(data.params, ct.POINTER(RuntimeCopy)).contents
                    if candidate.kind == 1:  # cudaMemcpyHostToDevice
                        copy = candidate
                elif domain == 1 and row["name"].startswith("cuMemcpyHtoDAsync"):
                    copy = ct.cast(data.params, ct.POINTER(DriverCopy)).contents
                if copy is not None:
                    row.update(bytes=copy.count, source=hex(copy.src), destination=hex(copy.dst))
                    if copy.count <= 16:
                        payload = ct.string_at(copy.src, copy.count)
                        row["source_bytes_hex"] = payload.hex()
                        if copy.count == 8:
                            row["source_int32_pair"] = list(struct.unpack("ii", payload))
                            row["source_double"] = struct.unpack("d", payload)[0]
                            row["source_int64"] = struct.unpack("q", payload)[0]
            elif data.site == 1 and data.result:
                row["status"] = ct.cast(data.result, ct.POINTER(ct.c_int)).contents.value
            self.events.append(row)
        except Exception as exc:
            self.errors.append(repr(exc))

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()

    def close(self):
        self.phase = None
        if self.handle:
            self._check(self.lib.cuptiUnsubscribe(self.handle))
            self.handle = None
