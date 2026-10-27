import json
from pathlib import Path
import warp as wp
from warpack import BlockLOBPCG
from warpack.block import Identity
wp.init();device=wp.get_device();solver=BlockLOBPCG(Identity(16,device),2,4);solver.solve(1)
a=wp.array([3.0],dtype=wp.float64);b=wp.zeros_like(a);other=wp.Stream(device);wp.synchronize()
class WarmupReached(Exception):pass

def forbidden(*args,**kwargs):raise WarmupReached('capture called initialize while another stream was capturing')
solver.initialize=forbidden
with wp.ScopedCapture(stream=other) as capture:
    state=dict(device_capture=bool(device.is_capturing),current_stream_capture=bool(wp.get_stream(device).is_capturing))
    try:solver.capture(1)
    except Exception as e:state.update(exception=type(e).__name__,message=str(e))
    wp.copy(b,a,stream=other)
wp.capture_launch(capture.graph,stream=other);wp.synchronize();state['outer_result']=b.numpy().tolist()
print(state)
Path('/run/openclaw/horde/warpack-review/evidence/other-stream-before.json').write_text(json.dumps(state,indent=2)+'\n')
