"""Diagnostic: pending H2D overwrite followed by fresh graph upload."""
import json,sys
from pathlib import Path
import numpy as np
import warp as wp
@wp.kernel
def snapshot(x:wp.array[wp.float64],out:wp.array[wp.float64]):
    i=wp.tid();out[i]=x[i]
wp.init()
root=Path('/run/openclaw/horde/warpack-review/evidence')
mode=sys.argv[1];repeats=int(sys.argv[2]);sizes=[96*295,104*319,120*367,204*619,1048576]
failures=[]
for size in sizes:
    x=wp.zeros(size,dtype=wp.float64);y=wp.empty_like(x);status=wp.zeros(1,dtype=wp.int32)
    old=np.full(size,1.25);new=np.full(size,7.5)
    wp.launch(snapshot,size,[x,y]);wp.synchronize()
    for trial in range(repeats):
        x.assign(old);wp.synchronize()
        with wp.ScopedCapture() as capture:
            status.zero_()
            wp.launch(snapshot,size,[x,y])
        if mode=='initialized':
            wp.capture_launch(capture.graph);wp.synchronize()
        x.assign(new)
        if mode=='sync':wp.synchronize()
        wp.capture_launch(capture.graph)
        result=y.numpy()
        if not np.array_equal(result,new):
            failures.append(dict(size=size,trial=trial,old_values=int((result==old).sum()),new_values=int((result==new).sum()),max_error=float(np.max(abs(result-new)))))
            print('FAIL',failures[-1],flush=True)
    print(size,'done',flush=True)
(root/f'copy-upload-{mode}.json').write_text(json.dumps(dict(mode=mode,repeats=repeats,sizes=sizes,failures=failures),indent=2)+'\n')
