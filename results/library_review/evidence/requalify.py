from pathlib import Path
import os,json,hashlib,subprocess,sys
root=Path('/run/openclaw/horde/warpack-review');source=root/'source';out=root/'evidence'
env=os.environ|dict(PYTHONPATH=str(source)+':'+str(root/'warp-fsai'),WARP_CACHE_PATH=str(root/'cache-current'),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
manifest={str(p.relative_to(source)):hashlib.sha256(p.read_bytes()).hexdigest() for folder in ['warpack','tests'] for p in sorted((source/folder).rglob('*.py'))}
(out/'prepared-source.json').write_text(json.dumps(manifest,indent=2)+'\n')
names=['test_triangular_rotate','test_register_triangular','test_installed_register_triangular','test_installed_ritz_qr','test_metric_qr_reuse','test_ritz_image_refresh','test_shared_triangular']
for label,args in [('prepared-focused',['tests/'+n+'.py' for n in names]),('full-prepared',[])]:
 with (out/(label+'.log')).open('w') as log:
  p=subprocess.run([sys.executable,'-m','pytest','-q','--junitxml='+str(out/(label+'.xml')),*args],cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT)
 print(label,p.returncode,flush=True)
 if p.returncode:raise SystemExit(p.returncode)
print('PREPARED_FULL_COMPLETE',flush=True)
