import os, subprocess, sys
from pathlib import Path
root=Path('/run/openclaw/horde/warpack-review')
names=['test_triangular_rotate','test_register_triangular','test_installed_register_triangular','test_installed_ritz_qr','test_metric_qr_reuse','test_ritz_image_refresh','test_shared_triangular']
for label in ['baseline','source']:
    env=os.environ | dict(PYTHONPATH=str(root)+':'+str(root/label)+':'+str(root/'warp-fsai'),WARP_CACHE_PATH=str(root/'cache-current'),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',WARPACK_FAILURE_STATE=str(root/'evidence'/('state-'+label)))
    with (root/'evidence'/('diagnostic-'+label+'.log')).open('w') as out:
        result=subprocess.run([sys.executable,'-m','pytest','-q','-p','failure_state',*['tests/'+name+'.py' for name in names]],cwd=root/label,env=env,stdout=out,stderr=subprocess.STDOUT)
    print(label,result.returncode,flush=True)
