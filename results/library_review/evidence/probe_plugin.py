"""Diagnostic instrumentation, never installed or used for qualification."""
import warp as wp
from warpack import BlockLOBPCG

@wp.kernel
def count(out:wp.array[wp.int32]):
    out[0]+=1

original=BlockLOBPCG._orthogonalize_x_full

def instrumented(self):
    if not hasattr(self,'_qr_probe_input'):
        if wp.get_stream(self.device).is_capturing:
            return original(self)
        self._qr_probe_input=wp.empty_like(self.sx)
        self._qr_probe_count=wp.zeros(1,dtype=wp.int32,device=self.device)
    if wp.get_stream(self.device).is_capturing:
        wp.copy(self._qr_probe_input,self.sx)
        wp.launch(count,1,[self._qr_probe_count],device=self.device)
    original(self)

BlockLOBPCG._orthogonalize_x_full=instrumented
