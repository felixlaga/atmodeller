"""Compare 920 K local equilibrium against the archived atmospheric calculation."""
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from model import make_atmosphere, extract, serializable
from atmodeller.engine import compute_implied_log_stability

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'


def main():
    start=time.time()
    book=pd.read_excel(HERE/'sources/HHeCNOSi_atm_magma_sol_real_1xsolar_melt100_1wtH.xlsx',sheet_name=None)
    mass={e:np.asarray(book['element_'+e]['total_mass'],float) for e in ['H','He','C','N','O','Si']}
    pressure=book['atmosphere']['pressure'].to_numpy(float)
    model=make_atmosphere(mass,pressure)
    mapping={'C_s':['C_cr'],'CSi_s':['CSi_b'],'Si_s':['Si_cr'],'N4Si3_s':['N4Si3_cr'],
             'O2Si_s':['O2Si_bqz','O2Si_aqz','O2Si_bcrt']}
    logn=[]
    for s in model.parameters.species_names:
        if s.endswith('_g'):
            mol=np.asarray(book[s]['atmosphere_moles'],float)
        else:
            mol=sum(np.asarray(book[ss]['total_moles'],float) for ss in mapping[s])
        logn.append(np.log(np.maximum(mol,1e-250)))
    logn=np.column_stack(logn)
    guess=np.concatenate([logn,np.asarray(compute_implied_log_stability(model.parameters,logn))],axis=1)
    print('Solving 50 archived atmospheric cases',flush=True)
    out=model.solve(guess)
    np.save(OUT/'reference_atmosphere_solution.npy',np.asarray(out.solution))
    data=extract(out,mass)
    comparisons={s:float(np.max(abs(x/np.asarray(book[s]['volume_mixing_ratio'],float)-1)))
                 for s,x in data['mole_fraction'].items()}
    ratio=data['mole_fraction']['H4Si_g']/data['mole_fraction']['CH4_g']
    ref_ratio=np.asarray(book['H4Si_g']['volume_mixing_ratio'],float)/np.asarray(book['CH4_g']['volume_mixing_ratio'],float)
    checks=dict(n_cases=50,n_success=int(sum(data['success'])),
                maximum_relative_element_error=max(float(np.max(abs(v))) for v in data['relative_element_errors'].values()),
                maximum_relative_mole_fraction_differences=comparisons,
                maximum_relative_ratio_difference=float(np.max(abs(ratio/ref_ratio-1))),
                seconds=time.time()-start)
    (OUT/'reference_atmosphere.json').write_text(json.dumps(serializable(data),indent=2))
    (OUT/'reference_atmosphere_validation.json').write_text(json.dumps(checks,indent=2))
    print(json.dumps(checks,indent=2),flush=True)
    assert np.all(data['success'])


if __name__=='__main__':
    main()
