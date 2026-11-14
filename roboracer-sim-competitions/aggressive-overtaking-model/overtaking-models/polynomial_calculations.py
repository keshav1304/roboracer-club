"""Derive the implemented quintics and quantify transition-length effects."""
import json
from pathlib import Path
import numpy as np
from numpy.polynomial import Polynomial

def solve(boundaries):
 rows=[]
 for t,order in ((0,0),(0,1),(0,2),(1,0),(1,1),(1,2)):
  rows.append([Polynomial.basis(i).deriv(order)(t) for i in range(6)])
 return np.linalg.solve(rows,np.array(boundaries,float))

def main():
 q=solve([0,0,0,1,0,0]);h=solve([0,1,0,0,0,0])
 np.testing.assert_allclose(q,[0,0,0,10,-15,6],atol=1e-12)
 np.testing.assert_allclose(h,[0,1,0,-6,8,-3],atol=1e-12)
 f=Polynomial(q);slope=Polynomial(h);u=np.linspace(0,1,100001);tables=[]
 for length in (1.2,1.8,2.6,3.5):
  ds=.5*f.deriv()(u)/length;dss=.5*f.deriv(2)(u)/length**2
  curvature=abs(dss)/(1+ds**2)**1.5
  tables.append(dict(entry_m=length,lateral_move_m=.5,speed_m_s=2.4,max_lateral_slope=float(abs(ds).max()),max_offset_second_derivative_per_m=float(abs(dss).max()),max_straight_reference_curvature_per_m=float(curvature.max()),max_lateral_acceleration_at_constant_speed_m_s2=float((2.4**2*curvature).max()),constant_speed_limit_for_3_2_m_s2=float(np.sqrt(3.2/curvature.max()))))
 report=dict(q_coefficients_ascending=q.tolist(),heading_basis_coefficients_ascending=h.tolist(),boundary_order='value/slope/second derivative at u=0 then u=1',q_boundary_values=[float(f.deriv(k)(t)) for t in (0,1) for k in (0,1,2)],h_boundary_values=[float(slope.deriv(k)(t)) for t in (0,1) for k in (0,1,2)],examples=tables,scope='Analytical straight-reference transitions, zero initial lateral slope, fixed 0.5m lateral displacement, no walls/opponent/return taper. Not full-track collision or performance tests.',score_example=dict(A=dict(C=2.50,D=.10,U=.02,J=2.50-.8*.10+.8*.02),B=dict(C=2.45,D=.02,U=.01,J=2.45-.8*.02+.8*.01)),coefficient_origin='Quintic coefficients solve endpoint constraints; D/U weights are hand-selected heuristic parameters, not fitted or proven optimal.')
 out=Path('artifacts/coefficient_sensitivity_20261113/polynomial_calculations.json');out.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
if __name__=='__main__':main()
