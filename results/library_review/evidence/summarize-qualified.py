import gzip,hashlib,json,statistics as s
from pathlib import Path
root=Path('/run/openclaw/horde/warpack-review');src=root/'evidence/qualified';dst=root/'source/results/library_review/evidence/qualified';dst.mkdir(exist_ok=True)
manifest=json.loads((dst.parent/'manifest.json').read_text());selection=json.loads((src/'worker-selection.json').read_text());assert len(selection)==12
audit=json.loads((root/'evidence/wheel-qualified-audit.json').read_text())
summary={'qualification':'case regression passed; full numerical suite plus final targeted checks documented separately','case':'clamped_consistent','n':34560,'workers_per_version':3,'replays_per_worker':5,'selection':selection,'excluded_workers':['block-baseline-2','block-source-3'],'comparisons':{}}
for suite in ['block','direct']:
 comparison={}
 for label in ['baseline','source']:
  reports=[]
  for i in (1,2,3):
   stem=selection[f'{suite}-{label}-{i}'];d=json.loads((src/(stem+'.json')).read_text());o=json.loads((src/(stem+'.occupancy.json')).read_text());assert d['accepted'] and o['timing_eligible'];reports.append(d)
   if label=='source':assert {'warpack/'+k:v for k,v in d['library_source_sha256'].items()}==audit['source_sha256']
  samples=[r['seconds'] for d in reports for r in d['rows']] if suite=='block' else [x for d in reports for x in d['samples_seconds']]
  setup=[d['total_setup_seconds' if suite=='block' else 'setup_seconds'] for d in reports];capture=[d['capture_seconds'] for d in reports]
  quality=[r['quality'] for d in reports for r in d['rows']] if suite=='block' else [r for d in reports for r in d['validation']]
  assert all(q['accepted'] for q in quality)
  comparison[label]=dict(replay_median_seconds=s.median(samples),replay_range_seconds=[min(samples),max(samples)],setup_samples_seconds=setup,capture_samples_seconds=capture,setup_capture_median_seconds=s.median([a+b for a,b in zip(setup,capture)]),max_residual=max(x['residual'] for x in quality),max_spectrum_error=max(x['spectrum_error'] for x in quality),max_requested_subspace_error=max(x['requested_subspace_error'] for x in quality),graphs=[d['graph' if suite=='block' else 'graph_audit'] for d in reports])
 comparison['replay_change_percent']=100*(comparison['source']['replay_median_seconds']/comparison['baseline']['replay_median_seconds']-1)
 comparison['setup_capture_change_percent']=100*(comparison['source']['setup_capture_median_seconds']/comparison['baseline']['setup_capture_median_seconds']-1)
 assert comparison['replay_change_percent']<10 and comparison['setup_capture_change_percent']<25
 assert comparison['baseline']['graphs']==comparison['source']['graphs']
 summary['comparisons'][suite]=comparison
 print(suite, {key:{k:v for k,v in val.items() if 'graphs'!=k} if isinstance(val,dict) else val for key,val in comparison.items()})
for p in sorted(src.iterdir()):
 if p.suffix not in ['.json','.log']:continue
 data=p.read_bytes();target=dst/(p.name+'.gz');target.write_bytes(gzip.compress(data,mtime=0));manifest.append(dict(file=str(target.relative_to(dst.parent)),original_bytes=len(data),original_sha256=hashlib.sha256(data).hexdigest()))
(dst.parent/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n');(dst/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
