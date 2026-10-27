import os,sys,subprocess,json
from pathlib import Path
root=Path('/run/openclaw/horde/warpack-review');out=root/'evidence'
for runtime in ['current','stock']:
 for mode in ['pending','sync','initialized']:
  label=f'upload-sequence-{runtime}-{mode}'
  env=os.environ|dict(PYTHONPATH=str(root/'source')+(':'+str(root/'warp-fsai') if runtime=='current' else ''),WARP_CACHE_PATH=str(root/('cache-'+runtime)),OPENBLAS_NUM_THREADS='4',OMP_NUM_THREADS='4')
  with (out/(label+'.log')).open('w') as log:
   p=subprocess.run([sys.executable,'-m','benchmarks.experimental.graph_input_upload','--mode',mode,'--out',str(out/(label+'.json'))],cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT)
  print(label,p.returncode,flush=True)
  if p.returncode:raise SystemExit(p.returncode)
  d=json.loads((out/(label+'.json')).read_text());print(len(d['failures']), 'observed failures',flush=True)
print('UPLOAD_CONTROLS_COMPLETE',flush=True)
