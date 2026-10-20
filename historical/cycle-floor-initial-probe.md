# Initial shifted-cycle precision probe

This retains the initial diagnostic script used before the public fresh evaluation
API follow-up. It uses the cached solver at c295594, manual fresh applications,
and an 80-digit reference inverse. Numerical reference work stays outside the
production solve. There is no performance claim.

[Raw evidence](../results/performance_followup/cycle-floor-initial.json).

```python
import hashlib,json
from pathlib import Path
import mpmath as mp
import numpy as np
import warp as wp
from benchmarks.upstream.adapters import matrix
from warpack import CuDSSInverse,EigenpairEvaluation,KrylovSchur,SparseOperator
mp.mp.dps=80
n=20;sigma=-1e-6;eye=np.eye(n)
a=eye-.5*np.roll(eye,1,axis=0)-.5*np.roll(eye,-1,axis=0)
s=a-sigma*eye; exact=mp.matrix(s.tolist())**-1
expected,basis=np.linalg.eigh(a)
report=dict(source_base_revision='c295594',n=n,sigma=sigma,matrix_sha256=hashlib.sha256(s.tobytes()).hexdigest(),condition=float(np.linalg.cond(s)),precision_digits=80,rows=[])
out=Path('build/cycle-floor-initial.json')
for refinements in [0,1,2,3]:
 for k,ncv in [(3,6),(5,12),(6,12)]:
  for factor_instance in range(2):
   inv=CuDSSInverse(matrix(s),1,mtype='symmetric',refinement_steps=refinements)
   solver=KrylovSchur(inv,k,ncv=ncv,tol=1e-12)
   physical=EigenpairEvaluation(SparseOperator(matrix(a)),solver.eigenvectors,tol=1e-9)
   graph=solver.capture(100,finalizer=physical.run)
   fresh=wp.empty_like(solver.eigenvectors)
   for replay in range(2):
    wp.capture_launch(graph)
    v=solver.eigenvectors.numpy();mu=solver.eigenvalues.numpy();w=physical.values.numpy()
    for j in range(k):inv.apply(solver.eigenvectors[j:j+1],fresh[j:j+1])
    fv=fresh.numpy()
    ideal_res=[];application_error=[]
    for row,lam,fx in zip(v,mu,fv):
     mx=mp.matrix(row.tolist());ex=exact*mx
     ideal_res.append(float(mp.norm(ex-mp.mpf(float(lam))*mx)/max(1,abs(lam))))
     application_error.append(float(mp.norm(mp.matrix(fx.tolist())-ex)/max(1,abs(lam))))
    allowed=basis[:,expected <= expected[k-1]+1e-9]
    row=dict(refinement_steps=refinements,k=k,ncv=ncv,factor_instance=factor_instance,replay=replay,inverse_count=int(solver.converged.numpy()[0]),status=int(solver.status.numpy()[0]),physical_count=int(physical.converged.numpy()[0]),cached_inverse_residuals=solver.residuals.numpy().tolist(),fresh_inverse_residuals=(np.linalg.norm(fv-mu[:,None]*v,axis=1)/np.maximum(1,abs(mu))).tolist(),ideal_inverse_residuals=ideal_res,inverse_application_error=application_error,physical_residuals=physical.residuals.numpy().tolist(),spectrum_error=float(abs(np.sort(w)-expected[:k]).max()),subspace_error=float(np.linalg.norm(v.T-allowed@(allowed.T@v.T))),orthogonality_error=float(abs(v@v.T-np.eye(k)).max()))
    report['rows'].append(row);out.write_text(json.dumps(report,indent=2)+'\n')
    print(refinements,k,factor_instance,replay,row['inverse_count'],max(row['cached_inverse_residuals']),max(row['fresh_inverse_residuals']),max(ideal_res),flush=True)
   solver.graph=graph=None;wp.synchronize_device();inv.close()
```
