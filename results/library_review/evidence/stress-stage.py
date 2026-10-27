"""Isolate the unchanged physical fixture; compare graph first-launch ordering."""
import inspect,json,sys,traceback,os
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from tests import test_triangular_rotate as fixture
from benchmarks.experimental.ritz_image_refresh import FreshRitzImageBlock
import failure_state
root=Path('/run/openclaw/horde/warpack-review/evidence')
variant=sys.argv[1];repeats=int(sys.argv[2]);name=sys.argv[3]
code=inspect.getsource(fixture.test_orthogonalization_with_changed_mass_scaling_and_cached_images)
code=code[code.index('def test_orthogonalization'):]
code=code.replace('for kind in [','for stage_index, kind in enumerate([').replace('"equilibrated_mass", "normal"]:', '"equilibrated_mass", "normal"]):')
if variant=='sync':code=code.replace('            wp.capture_launch(graph)','            wp.synchronize()\n            wp.capture_launch(graph)')
if variant=='initialized':code=code.replace('        solvers.append(solver)','        wp.capture_launch(capture.graph)\n        wp.synchronize()\n        solvers.append(solver)')
namespace=dict(fixture.__dict__);exec(code,namespace);test=namespace[fixture.test_orthogonalization_with_changed_mass_scaling_and_cached_images.__name__]
failures=[];os.environ['WARPACK_FAILURE_STATE']=str(root/('state-'+name))
for i in range(repeats):
    width=[96,104,120,204][i%4];cached=bool((i//4)%2)
    try:test(width,cached,FreshRitzImageBlock)
    except AssertionError as exc:
        frames=[SimpleNamespace(frame=SimpleNamespace(f_locals=frame.f_locals)) for frame,_ in traceback.walk_tb(exc.__traceback__)]
        relevant=next(e.frame.f_locals for e in frames if 'kind' in e.frame.f_locals)
        failures.append(dict(trial=i,width=width,cached=cached,kind=relevant['kind'],stage_index=relevant['stage_index'],line=traceback.extract_tb(exc.__traceback__)[-1].lineno))
        failure_state.pytest_runtest_makereport(SimpleNamespace(nodeid=f'{name}-{i}'),SimpleNamespace(when='call',excinfo=SimpleNamespace(traceback=frames)))
        print('FAIL',failures[-1],flush=True)
print(dict(variant=variant,repeats=repeats,failures=failures),flush=True)
(root/(name+'.json')).write_text(json.dumps(dict(variant=variant,repeats=repeats,failures=failures,fixture_source=code),indent=2)+'\n')
