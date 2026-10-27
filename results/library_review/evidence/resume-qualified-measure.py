# Independent case-specific measurements: full qualification is STILL HELD (#198).
import json, os, subprocess, sys, time
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
selection={}
for trial,order in enumerate([['baseline','source'],['source','baseline'],['baseline','source']],1):
    for checkout in order:
        for suite in ['block','direct']:
            original_label=f'{suite}-{checkout}-{trial}'
            label=original_label
            attempt=0
            while (out/(label+'.json')).exists():
                existing=json.loads((out/(label+'.json')).read_text())
                occupancy=json.loads((out/(label+'.occupancy.json')).read_text())
                if not existing.get('accepted'):
                    raise SystemExit(label+' has a numerical failure; do not retry')
                if occupancy['timing_eligible']:
                    break
                attempt+=1
                label=original_label+f'-retry{attempt}'
            if (out/(label+'.json')).exists():
                print('Retaining uncontended worker',label,flush=True)
                selection[original_label]=label
                (out/'worker-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
                continue
            idle_since=None
            while True:
                processes=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name','--format=csv,noheader'],text=True).strip()
                if processes:
                    idle_since=None
                    print('Waiting for quiet GPU:',processes,flush=True)
                elif idle_since is None:
                    idle_since=time.monotonic()
                elif time.monotonic()-idle_since>=20:
                    break
                time.sleep(5)
            settings={'PYTHONPATH':str(root/checkout)+':'+str(root/'warp-fsai')}
            common=[sys.executable,'-m','benchmarks.library_regression','--suite',suite,'--reference-cache',str(root/'references'),'--out',str(out/(label+'.json'))]
            if suite=='block':
                args=['--case','clamped_consistent','--side','15','--modes','100','--width','104','--steps','300','--fsai-width','32','--fsai-layout','columns','--matrix-layout','columns','--repeats','5','--tol','1e-11','--public-api','--gram-chunk','4096','--symmetric-products','--fused-updates','--paired-rotation','--reuse-x-image','--active-search-bucket-size','4','--batched-products','--inner-cg','25','--batched-cg','--active-cg-bucket-size','4','--inner-fp32']
            else:
                args=['--cases','clamped_consistent','--side','15','--modes','20','--methods','warp','--ncv','64','--steps','8','--repeats','5','--worker']
            monitor=[sys.executable,'-m','benchmarks.monitor_gpu','--out',str(out/(label+'.occupancy.json')),'--log',str(out/(label+'.worker.log')),'--']+common+args
            execute(label,monitor,cwd=root/checkout,extra=settings)
            report=json.loads((out/(label+'.json')).read_text());occupancy=json.loads((out/(label+'.occupancy.json')).read_text())
            if suite == 'direct':
                if not all(check['solver_full_count'] and check['solver_status'] == 0 for check in report['validation']):
                    raise SystemExit(label+' failed native convergence/status gate')
            if not report.get('accepted') or not occupancy['timing_eligible']:
                raise SystemExit(label+' not accepted/uncontended; retain evidence')
            selection[original_label]=label
            (out/'worker-selection.json').write_text(json.dumps(selection,indent=2)+'\n')
print('FINAL_CASE_MEASUREMENTS_COMPLETE',flush=True)
