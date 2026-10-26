"""Propagate MEB gas inventories into imposed local atmospheric equilibria.

Independent closed parcels, with all candidate condensates retained in equilibrium.
These are not a radiative-convective profile, rainout, quenching, or a spectrum.
"""
import json
from pathlib import Path
import time
import numpy as np
from model import *

HERE=Path(__file__).resolve().parent
OUT=HERE/'outputs'


def scenario_list(meb, condensed):
    cases=[]
    for i,c in enumerate(meb['cases']):
        if not condensed and not (c['mode']=='h2' and c['T_K']==3000 and c['H_wt_percent']==1):
            continue
        points={(920.,.01)}
        if not condensed:
            points={(920.,float(p)) for p in np.logspace(-3,2,11)}
        elif c['mode']=='h2' and c['T_K']==3000 and c['H_wt_percent']==1:
            points.update({(700.,.01),(1200.,.01)})
            if c['Z'] in [1,100] and c['factor'] in [.1,1.,10.]:
                points.update((920.,float(p)) for p in np.logspace(-3,2,11))
        for t,p in sorted(points):
            cases.append(dict(meb_index=i,T_atm_K=t,P_bar=p,condensates=condensed))
    return cases


def run(meb,condensed):
    start=time.time()
    cases=scenario_list(meb,condensed)
    idx=np.asarray([c['meb_index'] for c in cases])
    inventory={e:np.asarray(v)[idx] for e,v in meb['data']['element_phase_mass_kg']['gas'].items()}
    total=sum(inventory.values())
    # Normalize each closed parcel to one kg: fixed-P equilibrium fractions are scale invariant.
    mass={e:v/total for e,v in inventory.items()}
    pressure=np.asarray([c['P_bar'] for c in cases]);temperature=np.asarray([c['T_atm_K'] for c in cases])
    model=make_atmosphere(mass,pressure,temperature,condensates=condensed)
    print('Solving atmosphere:',len(cases),'condensates',condensed,flush=True)
    # The ideal-gas elemental-potential initializer includes a phase-selection pass.
    # Reinitialize each target directly instead of continuing a depleted metastable
    # condensate assemblage across a change of stable phases.
    out=model.solve_with_default()
    suffix='condensed' if condensed else 'gas_only'
    np.save(OUT/f'atmosphere_{suffix}_solution.npy',np.asarray(out.solution))
    data=extract(out,mass)
    f=data['fugacity_bar'];x=data['mole_fraction']
    species={s.name:s for s in make_gases(False)}
    signs={'H4Si_g':1,'CO_g':1,'OSi_g':-1,'CH4_g':-1}
    ln_k=-sum(sign*species[s].get_gibbs_over_RT(jnp.asarray(temperature),jnp.asarray(pressure)) for s,sign in signs.items())
    ln_q=sum(sign*np.log(f[s]) for s,sign in signs.items())
    gas_si_fraction=data['element_phase_mass_kg']['gas']['Si']/mass['Si']
    checks=dict(n_cases=len(cases),n_success=int(sum(data['success'])),
                maximum_relative_element_error=max(float(np.max(abs(v))) for v in data['relative_element_errors'].values()),
                maximum_abs_ln_reaction_residual=float(np.max(abs(ln_q-ln_k))),
                maximum_relative_pressure_error=float(np.max(abs(data['pressure_bar']/pressure-1))),
                seconds=time.time()-start)
    payload=dict(cases=cases,data=data,diagnostics=dict(ratio=x['H4Si_g']/x['CH4_g'],
                 gas_fraction_of_parcel_Si=gas_si_fraction,ln_Q_minus_ln_K=ln_q-ln_k),checks=checks)
    (OUT/f'atmosphere_{suffix}.json').write_text(json.dumps(serializable(payload),indent=2))
    print(json.dumps(checks,indent=2),flush=True)
    assert np.all(data['success'])
    assert checks['maximum_relative_element_error'] < 1e-7
    assert checks['maximum_abs_ln_reaction_residual'] < 1e-6


def main():
    meb=json.loads((OUT/'study_meb.json').read_text())
    assert all(meb['data']['success'])
    run(meb,True)
    run(meb,False)


if __name__=='__main__':
    main()
