"""Diagnostic only: snapshot the input at the start of a newly uploaded QR graph."""
import json, os
from pathlib import Path
import numpy as np
import scipy.sparse as sp
import warp as wp
from warpack import BlockLOBPCG, SparseOperator
from benchmarks.upstream.adapters import matrix
root=Path('/run/openclaw/horde/warpack-review/evidence')
width=104;n=3*width+7
source=np.random.default_rng(173).normal(size=(width,n))
m=sp.diags([-.1*np.ones(n-1),np.linspace(1,2,n),-.1*np.ones(n-1)],[-1,0,1],format='csr')
solver=BlockLOBPCG(SparseOperator(matrix(sp.eye(n,format='csr'))),4,width,mass=SparseOperator(matrix(m)))
snap=wp.empty_like(solver.sx)
results=[]
for synchronized in [False,True]:
    failures=[]
    for trial in range(300):
        solver.sx.assign(source);solver.status.zero_();solver.orthogonalize_x();wp.synchronize()
        with wp.ScopedCapture() as capture:
            solver.status.zero_()
            wp.copy(snap,solver.sx)
            solver.orthogonalize_x()
        solver.sx.assign(source)
        if synchronized:wp.synchronize()
        wp.capture_launch(capture.graph)
        status=int(solver.status.numpy()[0]);before=snap.numpy();scale=solver.xscale.numpy();expected=1/np.sqrt(np.sum(source*(m@source.T).T,axis=1))
        err=float(np.max(abs(before-source)))
        serr=float(np.max(abs(scale-expected)))
        if err or status or serr>1e-12:
            failures.append(dict(trial=trial,input_error=err,scale_error=serr,status=status))
            np.savez_compressed(root/f'probe-input-{synchronized}-{trial}.npz',source=source,before=before,scale=scale,after=solver.sx.numpy())
    results.append(dict(synchronized=synchronized,trials=300,failures=failures));print(results[-1],flush=True)
(root/'probe-input.json').write_text(json.dumps(results,indent=2)+'\n')
