import json, os, subprocess, sys
from pathlib import Path
root=Path('/run/openclaw/horde/warpack-review'); source=root/'source'; out=root/'evidence/qualified'
env=os.environ.copy()
env.update(OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4',WARP_CACHE_PATH=str(root/'cache-current'))
def execute(label, cmd, cwd=source, extra=None):
    settings=env | (extra or {})
    print('Starting',label,flush=True)
    with (out/(label+'.log')).open('w') as f:
        completed=subprocess.run(cmd,cwd=cwd,env=settings,stdout=f,stderr=subprocess.STDOUT)
    print('Finished',label,completed.returncode,flush=True)
    if completed.returncode:raise SystemExit(label+' failed')
for runtime in ['current','stock']:
    pythonpath=str(root/'installed-qualified')
    if runtime=='current':pythonpath+=':'+str(root/'warp-fsai')
    execute('wheel-'+runtime,[sys.executable,str(source/'scripts/verify_release_wheel.py'),'--installed',str(root/'installed-qualified'),'--out',str(out/('wheel-'+runtime+'.json'))]+(['--require-fast'] if runtime=='current' else []),cwd=root,extra={'PYTHONPATH':pythonpath,'WARP_CACHE_PATH':str(root/('cache-'+runtime))})
installed_check = 'import pathlib,warpack,pytest; assert pathlib.Path(warpack.__file__).resolve().is_relative_to(pathlib.Path("'+str(root/'installed-qualified')+'")); raise SystemExit(pytest.main(["--import-mode=importlib","-q",'+','.join(repr(str(source/'tests'/name)) for name in ['test_cudss_capture_setup.py','test_pivoting.py','test_cudss_stream_contract.py'])+']))'
execute('installed-cudss',[sys.executable,'-c',installed_check],cwd=root,extra={'PYTHONPATH':str(root/'installed-qualified')+':'+str(root/'warp-fsai')+':'+str(source)})
execute('focused-stock-final',[sys.executable,'-m','pytest','-q','tests/test_library_capture_contracts.py','tests/test_operator_storage_contracts.py','tests/test_graph_audit_boundary.py','tests/test_library_boundaries.py','tests/test_cudss_capture_setup.py','tests/test_pivoting.py','tests/test_cudss_stream_contract.py','tests/test_projected.py','tests/test_projected_convergence.py','tests/test_tiled_schur.py','tests/test_block_api.py','tests/test_general_budgets.py','tests/test_symmetric_capture_capabilities.py','tests/test_generalized_capture_capabilities.py'],extra={'PYTHONPATH':str(source),'WARP_CACHE_PATH':str(root/'cache-stock')})
execute('stock-mass-qr',[sys.executable,'-m','pytest','-q',*['tests/'+name+'.py' for name in ['test_triangular_rotate','test_register_triangular','test_installed_register_triangular','test_installed_ritz_qr','test_metric_qr_reuse','test_ritz_image_refresh','test_shared_triangular']]],extra={'PYTHONPATH':str(source),'WARP_CACHE_PATH':str(root/'cache-stock')})
print('FINAL_FRONTENDS_COMPLETE',flush=True)
