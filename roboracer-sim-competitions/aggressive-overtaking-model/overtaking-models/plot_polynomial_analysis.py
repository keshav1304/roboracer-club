from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=Path('artifacts/coefficient_sensitivity_20261113')
fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
u=np.linspace(0,1,2000);q=10*u**3-15*u**4+6*u**5
qp=30*u**2-60*u**3+30*u**4;qpp=60*u-180*u**2+120*u**3
for length in (1.2,1.8,2.6):
 axes[0].plot(length*u,.5*q,label=f'{length:g} m transition')
 ds=.5*qp/length;dss=.5*qpp/length**2
 axes[1].plot(length*u,2.4**2*abs(dss)/(1+ds**2)**1.5)
axes[0].set(xlabel='Distance along a straight reference (m)',ylabel='Lateral offset (m)',title='Same polynomial, different move lengths')
axes[0].legend(fontsize=8)
axes[1].axhline(3.2,color='black',linestyle='--',label='Stage 4 passing allowance: 3.2 m/s²')
axes[1].set(xlabel='Distance along a straight reference (m)',ylabel='Required lateral acceleration (m/s²)',title='0.5 m shift at constant 2.4 m/s')
axes[1].legend(fontsize=8)
for ax in axes:ax.grid(alpha=.2)
fig.savefig(p/'polynomial-length-effects.png',dpi=170)
