"""Stable device workspace for cuDSS, including CUDA graph capture.

cuDSS 0.8's default allocator can call cudaMalloc during solve, which fails
under capture (CUDA error 900). This stream-local cache warms outside capture,
retains device allocations through graph lifetime, and reuses scratch addresses.
Destroy graphs before closing the owning CuDSSInverse. Not thread safe.
"""

import ctypes as ct

import warp as wp

Alloc = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.POINTER(ct.c_void_p), ct.c_size_t, ct.c_void_p)
Free = ct.CFUNCTYPE(ct.c_int, ct.c_void_p, ct.c_void_p, ct.c_size_t, ct.c_void_p)


class Handler(ct.Structure):
    _fields_ = [
        ("ctx", ct.c_void_p),
        ("device_alloc", Alloc),
        ("device_free", Free),
        ("name", ct.c_char * 64),
    ]


class Workspace:
    def __init__(self, solver):
        self.device = solver.device
        self.stream = wp.get_stream(self.device)
        self.buffers = {}
        self.available = set()
        self.error = None
        self.alloc = Alloc(self._alloc)
        self.free = Free(self._free)
        self.handler = Handler(None, self.alloc, self.free, b"warpack stable Warp workspace")
        lib = solver._lib
        # CudssSolver constructs data before exposing the handle. Recreate that
        # empty data with our allocator to avoid mixing allocation ownership.
        from warp_cudss._bindings import _check, cudssData_t

        _check(lib.cudssDataDestroy(solver._handle, solver._data), "cudssDataDestroy")
        lib.cudssSetDeviceMemHandler.argtypes = [ct.c_void_p, ct.POINTER(Handler)]
        lib.cudssSetDeviceMemHandler.restype = ct.c_int
        status = lib.cudssSetDeviceMemHandler(solver._handle, ct.byref(self.handler))
        if status:
            raise RuntimeError(f"cuDSS allocator setup failed: {status}")
        data = cudssData_t()
        _check(lib.cudssDataCreate(solver._handle, ct.byref(data)), "cudssDataCreate")
        solver._data = data

    def _alloc(self, ctx, out, size, stream):
        try:
            if size == 0:
                out[0] = None
                return 0
            if stream != self.stream.cuda_stream:
                raise RuntimeError("cuDSS workspace used on an unexpected stream")
            candidates = [ptr for ptr in self.available if self.buffers[ptr].size * 8 >= size]
            if candidates:
                ptr = min(candidates, key=lambda p: self.buffers[p].size)
                self.available.remove(ptr)
            else:
                if self.stream.is_capturing:
                    raise RuntimeError("Warm up the cuDSS solve before capture")
                with wp.ScopedStream(self.stream):
                    buf = wp.empty((size + 7) // 8, dtype=wp.uint64, device=self.device)
                ptr = buf.ptr
                self.buffers[ptr] = buf
            out[0] = ptr
            return 0
        except Exception as e:
            self.error = e
            return 1

    def _free(self, ctx, ptr, size, stream):
        if ptr:
            self.available.add(ptr)
        return 0
