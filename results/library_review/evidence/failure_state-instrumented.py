"""Read state only after pytest receives an assertion failure; no collection patch."""
import hashlib
import json
import os
from pathlib import Path


def pytest_runtest_makereport(item, call):
    if call.when != 'call' or call.excinfo is None:
        return
    for entry in call.excinfo.traceback:
        local = entry.frame.f_locals
        if 'solver' not in local or 'kind' not in local or 'mass_host' not in local:
            continue
        import numpy as np
        solver = local['solver']
        root = Path(os.environ['WARPACK_FAILURE_STATE'])
        root.mkdir(parents=True,exist_ok=True)
        stem = hashlib.sha256(item.nodeid.encode()).hexdigest()[:16]
        arrays = {name: local[name] for name in ['source','current'] if name in local}
        summary = {'nodeid':item.nodeid, 'kind':local['kind'], 'width':local.get('width'), 'cached':local.get('cached'), 'constructor':str(type(solver))}
        for name in ['status','gx','lx','xscale','sx','bx','_qr_probe_input','_qr_probe_count']:
            value=getattr(solver,name,None)
            if value is not None:
                value=value.numpy();arrays[name]=value
                summary[name]={'shape':value.shape,'nonfinite':int((~np.isfinite(value)).sum()),'min':float(np.nanmin(value)),'max':float(np.nanmax(value))}
        for name in ['offsets','columns','values']:
            arrays['device_mass_'+name]=getattr(solver.mass.matrix,name).numpy()
        m=local['m']
        import scipy.linalg as la
        q=la.qr(local['current'].T, mode='economic')[0]
        for i, output in enumerate(local.get('outputs', [])):
            arrays[f'output_{i}']=output.copy()
            summary[f'output_{i}_span_defect']=float(la.norm(output-output@q@q.T)/la.norm(output))
        import warp as wp
        for i, graph in enumerate(local.get('graphs', [])):
            wp.capture_debug_dot_print(graph, str(root/(stem+f'-graph-{i}.dot')))
        for i, instance in enumerate(local.get('solvers', [])):
            summary[f'solver_{i}_type']=str(type(instance))
            summary[f'solver_{i}_pointers']={name:getattr(instance,name).ptr for name in ['sx','gx','lx','xscale','bx','status']}
        for i, instance in enumerate(local.get('solvers', [])):
            for name in ['status','gx','lx','xscale','sx','bx','_qr_probe_input','_qr_probe_count']:
                value=getattr(instance,name,None)
                if value is not None:
                    arrays[f'solver_{i}_{name}']=value.numpy()
            output=arrays[f'solver_{i}_sx']
            if np.isfinite(output).all():
                summary[f'solver_{i}_span_defect']=float(la.norm(output-output@q@q.T)/la.norm(output))
                summary[f'solver_{i}_mass_defect']=float(la.norm(output@(m@output.T)-np.eye(instance.b)))
        
        arrays.update(m_data=m.data,m_indices=m.indices,m_indptr=m.indptr)
        summary['mass_values_exact']=bool(np.array_equal(arrays['device_mass_values'][:len(m.data)],m.data))
        summary['mass_columns_exact']=bool(np.array_equal(arrays['device_mass_columns'][:len(m.indices)],m.indices))
        summary['mass_offsets_exact']=bool(np.array_equal(arrays['device_mass_offsets'],m.indptr))
        if np.isfinite(arrays['sx']).all():
            summary['output_mass_defect']=float(np.linalg.norm(arrays['sx'] @ (m @ arrays['sx'].T)-np.eye(solver.b)))
        np.savez_compressed(root/(stem+'.npz'),**arrays)
        (root/(stem+'.json')).write_text(json.dumps(summary,indent=2)+'\n')
        break
