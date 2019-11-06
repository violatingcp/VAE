import numpy as np
from collections import namedtuple

def sqr(x):
    return np.square(x)

Vec = namedtuple('Vec', ['x','y','z','t'])
# (pt,eta,phi,m) ->(px,py,px,E)                                                                                                                                                                               
def convert4(pt, eta, phi, m):
    x = pt * np.cos(phi)
    y = pt * np.sin(phi)
    z = pt * np.sinh(eta)
    t = np.sqrt(sqr(pt * np.cosh(eta)) + sqr(m))
    return Vec(x,y,z,t)

# add 2 vectors and return mass                                                                                                                                                                                 
def mvv(pt0, eta0, phi0, m0, pt1, eta1, phi1, m1):
    v0 = convert4(pt0, eta0, phi0, m0)
    v1 = convert4(pt1, eta1, phi1, m1)
    return np.sqrt(sqr(v0.t + v1.t)
                   - sqr(v0.x + v1.x)
                   - sqr(v0.y + v1.y)
                   - sqr(v0.z + v1.z))

def reconstruct_genmH(df):
    df['genmH_mmdt'] = np.vectorize(mvv)(df['j_pt'], df['j_eta'], df['j_phi'], df['j_mass_mmdt'],df['met_pt'], df['met_eta'], df['met_phi'], df['met_m'])
    return df['genmH_mmdt']

def reconstruct_smearmH(df):
    df['smearmH_mmdt'] = np.vectorize(mvv)(df['j_pt'], df['j_eta'], df['j_phi'], df['j_mass_mmdt'],df['metsmear_pt'], df['met_eta'], df['metsmear_phi'], df['met_m'])
    return df['smearmH_mmdt']

def reconstruct_mH_vals(j_pt,j_eta,j_phi,j_m,met_pt,met_eta,met_phi,met_m):
    mh = mvv(j_pt,j_eta,j_phi,j_m,met_pt,met_eta,met_phi,met_m)
    return mh

def reconstruct_mH_fromdf(df,j_pt='j_pt',j_eta='j_eta',j_phi='j_phi',j_m='j_mass_mmdt',met_pt='reg',met_eta='met_eta',met_phi='metsmear_phi',met_m='met_m'):
    df['%smH'%met_pt] = np.vectorize(mvv)(df[j_pt],df[j_eta],df[j_phi],df[j_m],df[met_pt],df[met_eta],df[met_phi],df[met_m])
    return df['%smH'%met_pt]

def mvv_xyzt(pt0, eta0, phi0, m0,px1,py1,pz1,m1):
    v0 = convert4(pt0, eta0, phi0, m0)
    t1 = np.sqrt(px1**2 + py1**2 + pz1**2 + m**2)
    return np.sqrt(sqr(v0.t + t1)
                   - sqr(v0.x + x1)
                   - sqr(v0.y + y1)
                   - sqr(v0.z + z1))

def reconstruct_mH_fromdf_xyzt(df,j_pt='j_pt',j_eta='j_eta',j_phi='j_phi',j_m='j_mass_mmdt',met_px='metsmear_px',met_py='metsmear_py',met_pz='met_pz',met_pt='metsmear_pt'):
    df['%smH_xyzt'%met_pt] = np.vectorize(mvv_xyzt)(df[j_pt],df[j_eta],df[j_phi],df[j_m],df[met_px],df[met_py],df[met_pz],df[met_m])
    return df['%smH_xyzt'%met_pt]
