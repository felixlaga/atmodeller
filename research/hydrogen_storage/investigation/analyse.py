"""Make scientific figures and independently test condensed-phase buffer identities."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from model import ChemicalSpecies, PurePhase, jnp, serializable

HERE=Path(__file__).resolve().parent;OUT=HERE/'outputs';FIG=HERE/'figures';FIG.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,
                     'grid.alpha':.18,'savefig.bbox':'tight','svg.fonttype':'none'})
COLORS={'h2':'#166eab','water':'#c05b17','both':'#19826a','fixed_pressure':'#6b548b'}
LABELS={'h2':'H₂ dissolution','water':'H₂O dissolution','both':'Both laws','fixed_pressure':'H₂, pressure fixed'}


def buffer_ratios(t,p):
    species={s:ChemicalSpecies.create_gas(s) for s in ['H4Si','CH4']}
    species.update({s:PurePhase.from_species(s,state='s').species.species[0] for s in ['Si','CSi','C']})
    g={s:np.asarray(sp.get_gibbs_over_RT(jnp.asarray(t),jnp.asarray(p))) for s,sp in species.items()}
    return np.exp(-(g['CSi']+g['H4Si']-2*g['Si']-g['CH4'])),np.exp(-(2*g['C']+g['H4Si']-g['CSi']-g['CH4']))


def main():
    meb=json.loads((OUT/'study_meb.json').read_text())
    atm=json.loads((OUT/'atmosphere_condensed.json').read_text())
    nogas=json.loads((OUT/'atmosphere_gas_only.json').read_text())
    mc=meb['cases'];ac=atm['cases'];nc=nogas['cases']
    def mselect(mode='h2',z=1,h=1,t=3000):
        return np.array([i for i,c in enumerate(mc) if c['mode']==mode and c['Z']==z and c['H_wt_percent']==h and c['T_K']==t])
    def aselect(ids,temperature=920,pressure=.01,gas_only=False):
        cases=nc if gas_only else ac
        mapping={c['meb_index']:i for i,c in enumerate(cases) if c['T_atm_K']==temperature and np.isclose(c['P_bar'],pressure)}
        return np.array([mapping[int(i)] for i in ids])
    at_ratio=np.asarray(atm['diagnostics']['ratio']);meb_ratio=np.asarray(meb['diagnostics']['ratio'])
    phase=atm['data']['phase_species_mass_kg']
    si=np.asarray(phase['Si_s']['Si_s']);sic=np.asarray(phase['CSi_s']['CSi_s']);carbon=np.asarray(phase['C_s']['C_s'])
    silicon_mask=(si>1e-12)&(sic>1e-12)
    carbon_mask=(carbon>1e-12)&(sic>1e-12)
    silicon_buffer,carbon_buffer=buffer_ratios([c['T_atm_K'] for c in ac],[c['P_bar'] for c in ac])
    buffering=dict(silicon_and_silicon_carbide_cases=int(sum(silicon_mask)),carbon_and_silicon_carbide_cases=int(sum(carbon_mask)),
                   other_cases=int(sum(~(silicon_mask|carbon_mask))),
                   maximum_relative_Si_SiC_buffer_error=float(np.max(abs(at_ratio[silicon_mask]/silicon_buffer[silicon_mask]-1))),
                   maximum_relative_C_SiC_buffer_error=float(np.max(abs(at_ratio[carbon_mask]/carbon_buffer[carbon_mask]-1))),
                   predicted_ratios_at_920K_10mbar=[float(v) for v in buffer_ratios(920.,.01)])
    rows=[]
    for mode in ['h2','water','both','fixed_pressure']:
        for z in [1,10,100]:
            ids=mselect(mode,z);aa=aselect(ids)
            for i,j in zip(ids,aa):
                c=mc[i]
                rows.append(dict(**c,P_MEB_bar=meb['data']['pressure_bar'][i],
                    H_dissolved_fraction=meb['diagnostics']['dissolved_H_fraction'][i],
                    ratio_MEB=meb_ratio[i],ratio_10mbar=at_ratio[j],
                    xSiH4_10mbar=atm['data']['mole_fraction']['H4Si_g'][j],
                    xCH4_10mbar=atm['data']['mole_fraction']['CH4_g'][j],
                    gas_fraction_Si=atm['diagnostics']['gas_fraction_of_parcel_Si'][j],
                    Si_s_kg_per_kg=si[j],SiC_s_kg_per_kg=sic[j],C_s_kg_per_kg=carbon[j],
                    buffer='Si+SiC' if silicon_mask[j] else 'C+SiC' if carbon_mask[j] else 'other'))
    local=[]
    for z in [1,10,100]:
        for h in [.1,.3,1]:
            ids=mselect('local_h2',z,h);aa=aselect(ids)
            local.append(dict(Z=z,H_wt_percent=h,
                dln_MEB_ratio_dln_H2_scale=float(np.log(meb_ratio[ids[1]]/meb_ratio[ids[0]])/(.02*np.log(10))),
                dln_atmospheric_ratio_dln_H2_scale=float(np.log(at_ratio[aa[1]]/at_ratio[aa[0]])/(.02*np.log(10)))))
    summary=dict(buffer_checks=buffering,selected_cases=rows,local_sensitivities=local,
                 checks={k:d['checks'] for k,d in [('meb',meb),('atmosphere',atm),('gas_only',nogas)]})
    (OUT/'analysis_summary.json').write_text(json.dumps(serializable(summary),indent=2))
    print(json.dumps(buffering,indent=2));print('LOCAL SLOPES',local)
    for r in rows:
        if r['Z']==1 and r['factor'] in [.1,1.,10.]: print(r)

    fig,axs=plt.subplots(2,2,figsize=(9,7.3),layout='constrained')
    for mode in ['h2','water','both','fixed_pressure']:
        ids=mselect(mode);aa=aselect(ids);s=np.array([mc[i]['factor'] for i in ids]);color=COLORS[mode]
        for ax,values in [(axs[0,0],100*np.asarray(meb['diagnostics']['dissolved_H_fraction'])[ids]),
                          (axs[0,1],meb_ratio[ids]),(axs[1,0],at_ratio[aa])]:
            ax.plot(s,values,'o-' if mode!='fixed_pressure' else '--',ms=3,color=color,label=LABELS[mode])
    ids=mselect();aa=aselect(ids);s=np.array([mc[i]['factor'] for i in ids])
    for species,label,color in [('H4Si_g','SiH₄','#166eab'),('CH4_g','CH₄','#c05b17')]:
        axs[1,1].plot(s,np.asarray(atm['data']['mole_fraction'][species])[aa],'o-',label=label,color=color,ms=3)
    axs[0,0].set(ylabel='Hydrogen dissolved (%)',title='Interior storage')
    axs[0,1].set(ylabel='SiH₄ / CH₄',yscale='log',title='At the magma boundary, 3000 K')
    axs[1,0].set(ylabel='SiH₄ / CH₄',yscale='log',title='After cooling: 920 K, 10 mbar')
    axs[1,0].set_ylim(.002,.02)
    axs[1,0].text(.5,.2,'All four curves coincide\nSi + SiC set the ratio',transform=axs[1,0].transAxes,
                  ha='center',fontsize=10,color='#444')
    axs[1,1].set(ylabel='Gas mole fraction',yscale='log',title='H₂-law perturbation, 10 mbar')
    axs[0,0].legend(fontsize=9);axs[1,1].legend()
    for ax in axs.flat: ax.set(xscale='log',xlabel='Solubility-law multiplier');ax.axvline(1,color='#888',lw=.7,zorder=0)
    fig.suptitle('Changing interior hydrogen storage\n1 wt% H, solar accreted gas, fully molten mantle',fontsize=14)
    save(fig,'storage_sensitivity')

    fig,axs=plt.subplots(2,1,figsize=(7.2,7.8),layout='constrained')
    for z,color in zip([1,10,100],['#166eab','#19826a','#c05b17']):
        ids=mselect(z=z);aa=aselect(ids);s=np.array([mc[i]['factor'] for i in ids])
        axs[0].plot(s,at_ratio[aa],'o-',color=color,label=f'{z}× solar')
    axs[0].axhline(buffering['predicted_ratios_at_920K_10mbar'][0],ls=':',color='#777')
    axs[0].axhline(buffering['predicted_ratios_at_920K_10mbar'][1],ls=':',color='#777')
    axs[0].set(title='Composition regimes at 920 K',ylabel='SiH₄ / CH₄',yscale='log');axs[0].legend()
    for t,color in zip([700,920,1200],['#19826a','#166eab','#c05b17']):
        ids=mselect();aa=aselect(ids,temperature=t);s=np.array([mc[i]['factor'] for i in ids])
        axs[1].plot(s,at_ratio[aa],'o-',color=color,label=f'{t} K')
    axs[1].set(title='Solar composition: temperature changes the buffer',ylabel='SiH₄ / CH₄',yscale='log')
    axs[1].legend(ncol=3,loc='center',bbox_to_anchor=(.5,.27))
    for ax in axs:ax.set(xscale='log',xlabel='H₂ solubility-law multiplier')
    fig.suptitle('Does the interior signal survive condensation?\n10 mbar; 1 wt% H; 3000 K magma boundary',fontsize=13)
    save(fig,'phase_buffers')

    fig,axs=plt.subplots(2,1,figsize=(7.2,7.8),layout='constrained')
    for ax,z in zip(axs,[1,100]):
        for factor,color in zip([.1,1,10],['#19826a','#166eab','#c05b17']):
            mid=next(i for i,c in enumerate(mc) if c['mode']=='h2' and c['T_K']==3000 and c['H_wt_percent']==1 and c['Z']==z and np.isclose(c['factor'],factor))
            for dataset,allcases,ls in [(atm,ac,'-'),(nogas,nc,'--')]:
                ii=[i for i,c in enumerate(allcases) if c['meb_index']==mid and c['T_atm_K']==920]
                pp=np.array([allcases[i]['P_bar'] for i in ii]);rr=np.array(dataset['diagnostics']['ratio'])[ii]
                ax.plot(pp,rr,ls,color=color,label=f'H₂ law ×{factor:g}' if ls=='-' else None)
        ax.set(xscale='log',yscale='log',ylabel='SiH₄ / CH₄',xlabel='Pressure (bar)',title=f'{z}× solar accreted gas')
        ax.legend(ncol=3,fontsize=9)
    fig.suptitle('Condensation controls the mapping to atmospheric gas\nSolid: equilibrium condensates retained · Dashed: condensates disabled',fontsize=12)
    save(fig,'condensation_control')


def save(fig,name):
    for extension in ['png','svg','pdf']:fig.savefig(FIG/f'{name}.{extension}',dpi=220)
    plt.close(fig)


if __name__=='__main__':main()
