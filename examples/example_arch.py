import sys
import time
import numpy as np
import pyoti.sparse as oti
import scipy.stats as stats
import scipy.signal as signal
import matplotlib.pyplot as plt

import continuationHYPAD


notebook_time = time.time()

# Reset output folders for ROM data, figures, and MATSO objects
deleteFolders = True
if deleteFolders:
    import os
    import shutil
    folders = ["TSE", "Figures", "MATSO"]

    for folder in folders:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)
    print("Folders reset successfully.")

# Define input data
ndof = 3
maxOrder = 3
versionNumber = 1
formulation = 'frame'
showConvergenceData = False

# Homogenous properties
to = 0.01
bo = 1.00
Eo = 12.0e6
beta = 100.0
alpha = 215/2
R = beta*to

# Mean Chebyshev vector with OTI perturbations (a* = avg(c) + epsilon)
c0 = 6.0e-4 + oti.e(1, order=maxOrder)
c1 = 0.0    + oti.e(2, order=maxOrder)
c2 = 0.0    + oti.e(3, order=maxOrder)
c3 = 0.0    + oti.e(4, order=maxOrder)
c4 = 0.0    + oti.e(5, order=maxOrder)
c5 = 0.0    + oti.e(6, order=maxOrder)
c6 = 0.0    + oti.e(7, order=maxOrder)
c7 = 0.0    + oti.e(8, order=maxOrder)
c8 = 0.0    + oti.e(9, order=maxOrder)
c = [c0, c1, c2, c3, c4, c5, c6, c7, c8]

# Add an extra node if the number of nodes is even
# Always an node exactly at the crown.
nnodes = 16
if nnodes % 2 == 0:
    nnodes = nnodes + 1
node_id = nnodes//2

# Create nodes at equally spaced points along the span from -125° to 125°
export_node = int(np.ceil(nnodes/2))
theta = oti.array(np.linspace(1, -1, nnodes))
theta = np.pi * (theta * alpha + 90.0)/180.0
Coords = oti.zeros((nnodes, 2))
Coords[:,0] = R * oti.cos(theta)
Coords[:,1] = R * oti.sin(theta)
Connect = np.zeros((nnodes-1, 2))
Connect[:,0] = np.arange( nnodes-1 )
Connect[:,1] = np.arange( 1, nnodes )
Connect = Connect.astype(int)

# Elementary (Dirichlet) BCs
Ebc = np.zeros((nnodes, ndof))
Ebc[0, 0] = 1
Ebc[0, 1] = 1
Ebc[nnodes-1, :] = 1

# Load (Neumann) BCs
load = -10.0
Loads = oti.zeros((nnodes, ndof))
Loads[node_id, 1] = load

# Find imperfection field with Chebyshev constants
nelems = Connect.shape[0]
elems_theta = oti.array(np.linspace(1, -1, nelems)) * alpha + 90.0
del_appr = continuationHYPAD.utils.constantsToProfile(c, elems_theta)
thickness_appr = to - del_appr

# Material and cross section vector for all elements
E = Eo * oti.ones((nelems,1))
A = bo * thickness_appr
I = bo * thickness_appr**3 / 12

continuation_start = time.time()
U_step, lr_step, DOFexport = continuationHYPAD.continuationSolver(formulation, Coords, Connect, ndof, Ebc, Loads, E, A, I, export_node)
continuation_finish = time.time()
sim_time = continuation_finish - continuation_start

print('Analysis finished for beta = ' + str(beta) + ': '+ str(sim_time/60) + ' minutes')

SaveOTI = True
if SaveOTI:
    lr_oti = oti.array(lr_step)
    ux_oti = oti.zeros(len(U_step), 1)
    uy_oti = oti.zeros(len(U_step), 1)
    for col in range(len(U_step)):
        ux_oti[col,0] = -U_step[col][DOFexport[0], 0]
        uy_oti[col,0] =  U_step[col][DOFexport[1], 0]

    # Save all multi-geometry sensitivities with different filenames
    oti.save(lr_oti, 'MATSO/'+'lr_oti_beta'+str(beta)+'_v'+str(versionNumber))
    oti.save(ux_oti, 'MATSO/'+'ux_oti_beta'+str(beta)+'_v'+str(versionNumber))
    oti.save(uy_oti, 'MATSO/'+'uy_oti_beta'+str(beta)+'_v'+str(versionNumber))                
#end

# Postprocess KDF UQ for beta = 100, MC data is pre-loaded for beta = 100
filename = 'MC/C_mc_dof1_v2.npy'
C = np.load(filename, allow_pickle=True)
samples = C.shape[0]
timeincs = len(uy_oti.real.flatten())
timestamps = []
print('Loaded all samples for ROM evaluation (invCDF sampling of Chebyshev Coefficients)')

# Evaluate ROMs of different order
U_HP_perorder = []
L_HP_perorder = []
for order in range(1,maxOrder+1):
    
    # Truncate ROM to compare order performance
    tic = time.time()
    u_oti_truncated  = uy_oti.truncate_order(order+1)
    lr_oti_truncated = lr_oti.truncate_order(order+1)
    u_SMPL = np.zeros((timeincs, samples))
    l_SMPL = np.zeros((timeincs, samples))
         
    # Loop through samples of Chebyshev coefficients (C)
    for j in range(timeincs):
        # Evaluate ROMs at every solution increment
        u_SMPL[j,:] =  u_oti_truncated[j,0].rom_eval_object(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], [C[:,0]-6e-4,C[:,1],
            C[:,2],C[:,3],C[:,4],C[:,5],C[:,6],C[:,7],C[:,8]])

        l_SMPL[j,:] = lr_oti_truncated[j,0].rom_eval_object(
            [1, 2, 3, 4, 5, 6, 7, 8, 9], [C[:,0]-6e-4,C[:,1],
            C[:,2],C[:,3],C[:,4],C[:,5],C[:,6],C[:,7],C[:,8]])

    # Save to list
    U_HP_perorder.append(u_SMPL)
    L_HP_perorder.append(l_SMPL)
    
    # Export the ROM output per order if needed
    np.savez(('TSE/TSE_sz'+str(samples)+'_order'+str(order)+'_v'+str(versionNumber)+'.npz'), u_SMPL, l_SMPL)
    print('Done evaluating ROM Order '+str(order)+' in '+str(time.time() - tic)+' seconds')

# Plot UQ figures
lw = 2.0
Color_MC = np.array([1, 1, 1]) / 3
Color = [
    np.array([0, 255, 0]) / 255,
    np.array([0, 0, 255]) / 255,
    np.array([255, 0, 0]) / 255,
    np.array([255, 174, 66]) / 255,
]

ColorEdge = np.array([190, 143, 0]) / 255
ColorFill = np.array([255, 230, 153]) / 255
opacity = 1e-3
cutoffX = 725
cutoffY = -0.35
showDistr = True
if showDistr:  
    fig, axes = plt.subplots(3, 3, figsize=(11, 7))
    for i in range(C.shape[1]):
        mx = np.mean(C[:, i])
        
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif',size=15)
        ax = axes.flatten()[i]
        ax.hist(C[:, i], bins=30, density=True, color=ColorFill, edgecolor=ColorEdge)
        ax.axvline(mx, linestyle="--", color=ColorEdge, linewidth=lw)
        ax.set_xlabel(f"$c_{i}$")
        ax.set_ylabel(f"PDF$(c_{i})$")
        ax.set_xlim([np.min(C[:, i]), np.max(C[:, i])])
    plt.tight_layout()
    plt.show()

U_MC = np.load('MC/u_mc_dof1_v2.npy', allow_pickle=True)
L_MC = np.load('MC/l_mc_dof1_v2.npy', allow_pickle=True)
n = []
for i in range(samples):
    if len(U_MC[i]) != len(L_MC[i]):
        raise ValueError("Inconsistent size in MC results")
    n.append(len(U_MC[i]))

tempU = np.full((cutoffX, samples), np.nan)
tempL = np.full((cutoffX, samples), np.nan)
U_MC_cut, L_MC_cut = [], []
for k, N in enumerate(n):
    if N > cutoffX:
        U_MC_cut.append(U_MC[k][:cutoffX])
        L_MC_cut.append(L_MC[k][:cutoffX])
        n[k] = cutoffX
        tempU[:, k] = U_MC[k][:cutoffX]
        tempL[:, k] = L_MC[k][:cutoffX]
    else:
        U_MC_cut.append(U_MC[k])
        L_MC_cut.append(L_MC[k])
        tempU[:N, k] = U_MC[k]
        tempL[:N, k] = L_MC[k]

tic = time.time()
U_TSE = np.full((cutoffX, samples, maxOrder), np.nan)
L_TSE = np.full((cutoffX, samples, maxOrder), np.nan)
for order in range(1, maxOrder + 1):
    # Cut ROM results at the shape (n)
    U_HP_cut = []
    L_HP_cut = []
    for i in range(samples):
        U_HP_cut.append(U_HP_perorder[order-1][: n[i], i])
        L_HP_cut.append(L_HP_perorder[order-1][: n[i], i])
        U_TSE[: n[i], i, order - 1] = U_HP_cut[i]
        L_TSE[: n[i], i, order - 1] = L_HP_cut[i]
        
toc = time.time()
runtime = (toc - tic)
print("Re-organized ROM data in "+str(runtime)+" seconds")

# Save reshaped ROM data
np.savez('TSE/MatrixData_TSE.npz', U_TSE, L_TSE)

DPI = 300
showFigs  = True
saveFigs  = True
opacity   = 0.002
cutoff_samples = 3000

tic = time.time()
for order in range(1, maxOrder + 1):    
    if showFigs:
        # Create figure
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        # plt.rc('text', usetex=True)
        # plt.rc('font', family='serif',size=18)

        # Monte-Carlo UQ Plot
        for i in range(cutoff_samples):
            axes[0].plot(-U_MC_cut[i], "-", color=np.append(Color_MC, opacity), linewidth=lw)
            axes[1].plot( L_MC_cut[i], "-", color=np.append(Color_MC, opacity), linewidth=lw)
            axes[2].plot(-U_MC_cut[i], L_MC_cut[i], "-", color=np.append(Color_MC, opacity), linewidth=lw)

        axes[0].set(xlabel="$l$", ylabel="$u$")
        axes[0].set_xlim([1, cutoffX])
        axes[0].set_ylim([0, 2])
        axes[1].set(xlabel="$l$", ylabel="$\lambda$")
        axes[1].set_xlim([1, cutoffX])
        axes[1].set_ylim([cutoffY, 1])
        axes[2].set(xlabel="$u$", ylabel="$\lambda$")
        axes[2].set_xlim([0, 2])
        axes[2].set_ylim([cutoffY, 1])
        plt.tight_layout()
        
        # Export to Figures folder the first time around
        if (saveFigs and (order==1)):
            plt.savefig("Figures/UQ_MC.png", dpi=DPI)

        # Plot HYPAD-UQ on top
        for i in range(cutoff_samples):
            axes[0].plot(-U_TSE[:, i, order - 1], "-", color=np.append(Color[order - 1], opacity), linewidth=lw)
            axes[1].plot( L_TSE[:, i, order - 1], "-", color=np.append(Color[order - 1], opacity), linewidth=lw)
            axes[2].plot(-U_TSE[:, i, order - 1], L_TSE[:, i, order - 1], "-", color=np.append(Color[order - 1], opacity), linewidth=lw)

        axes[0].set_xlim([1, cutoffX])
        axes[0].set_ylim([0, 2])
        axes[1].set_xlim([1, cutoffX])
        axes[1].set_ylim([cutoffY, 1])
        axes[2].set_xlim([0, 2])
        axes[2].set_ylim([cutoffY, 1])

        # Export to Figures folder
        if saveFigs:
            plt.savefig(f"Figures/UQ_ROM_order{order}.png", dpi=DPI)
        plt.show()
        plt.close()
        
print("Generated UQ figures")

# Set plotting parameters
MinPeakProminence = 1e-1
capindex = 600
R = 1
t = 0.01
alpha = np.deg2rad(215)

# Load imperfection thickness deviation data \Delta t_i
D = np.load("MC/Da_mc_dof1_v2.npy", allow_pickle=True)
a = np.linspace(-alpha/2, alpha/2, D.shape[1])
dt = D - t
Ao = 2 * R * t * alpha
Ai = 2 * R * t * alpha + 2 * R * np.trapz(dt, a, axis=1)
p = Ai / Ao 
id_min = np.argmin(1 - p)

# Load MC FEM simulation results
Lc_theory = 0.92
nsamples = len(U_MC_cut)
dL_MC = []
for i in range(nsamples):
    if len(L_MC_cut[i]) > capindex:
        U_MC_cut[i] = U_MC_cut[i][:capindex]
        L_MC_cut[i] = L_MC_cut[i][:capindex]
    dL_MC.append(np.diff(L_MC_cut[i]))

indices = np.zeros(nsamples, dtype=int)
Lc = np.zeros(nsamples)
for col in range(nsamples):
    peaks, props = signal.find_peaks(L_MC_cut[col], prominence=MinPeakProminence)
    if len(peaks) == 0:
        Lc[col] = np.max(L_MC_cut[col])
        indices[col] = len(L_MC_cut[col]) - 1
    else:
        Lc[col] = L_MC_cut[col][peaks[0]]
        indices[col] = peaks[0]

indices_HP = np.zeros((maxOrder, nsamples))
Lc_HP = np.zeros((maxOrder, nsamples))
for j in range(maxOrder):
    for col in range(nsamples):
        peaks, props = signal.find_peaks(L_TSE[:,col,j], prominence=MinPeakProminence)
        if len(peaks) == 0:
            Lc_HP[j, col] = np.max(L_TSE[:,col,j])
            indices_HP[j, col] = len(L_TSE[:,col,j]) - 1
        else:
            Lc_HP[j, col] = L_TSE[peaks[0], col, j]
            indices_HP[j, col] = peaks[0]

K_MC = Lc / Lc_theory
K_HP = np.zeros((maxOrder, nsamples))
for j in range(maxOrder):
    K_HP[j, :] = Lc_HP[j, :] / Lc_HP[j, id_min]
inflimit, suplimit = 0.3, 1.3
K_HP[(K_HP < inflimit) | (K_HP > suplimit)] = np.nan

if showFigs:
    # PDF
    fig, ax = plt.subplots()
    ax.hist(K_MC, bins=30, density=True, facecolor='k', alpha=0.5, linewidth=1.5, edgecolor='k')
    ax.set_xlabel(r'$\kappa$ - Knockdown Factor')
    ax.set_ylabel(r'PDF$(\kappa)$')
    ax.set_ylim([0, 4])
    ax.set_xlim([0.36, 1.02])
    if saveFigs:
        plt.savefig(f'Figures/KDF_MC.png', dpi=DPI)
    for j in range(maxOrder):
        kde = stats.gaussian_kde(K_HP[j, ~np.isnan(K_HP[j, :])])
        x_vals = np.linspace(0.36, 1.02, 100)
        ax.plot(x_vals, kde(x_vals), linewidth=lw, color=Color[j], label=f'ROM Order {j+1}')
        ax.legend(loc='upper left', frameon=False)
        if saveFigs:
            plt.savefig(f'Figures/KDF_order{j+1}.png', dpi=DPI)
    plt.show()
    plt.close()
    
    # CDF
    fig, ax = plt.subplots()
    ax.hist(K_MC, bins=30, density=True, cumulative=True, facecolor='k', alpha=0.5, linewidth=1.5, edgecolor='k')
    ax.set_xlabel(r'KDF - $\kappa$')
    ax.set_ylabel(r'CDF$(\kappa)$')
    ax.set_ylim([0, 1])
    ax.set_xlim([0.36, 1.02])
    if saveFigs:
        plt.savefig(f'Figures/CDF_MC.png', dpi=DPI)
    for j in range(maxOrder):
        kde = stats.gaussian_kde(K_HP[j, ~np.isnan(K_HP[j, :])])
        x_vals = np.linspace(0.36, 1.02, 100)
        pdf_vals = kde(x_vals)
        cdf_vals = np.cumsum(pdf_vals)  
        cdf_vals /= cdf_vals[-1]  # Normalize to ensure max is 1
        ax.plot(x_vals, cdf_vals, color=Color[j], linewidth=lw, label=f'ROM Order {j+1}')
        ax.legend(loc='upper left', frameon=False)
        if saveFigs:
            plt.savefig(f'Figures/CDF_order{j+1}.png', dpi=DPI)
    plt.show()
    plt.close()

print("Generated PDF figures")

x = 1 - p # gamma
y = K_MC  # KDF
sort_idx = np.argsort(x)
x_sorted, y_sorted = x[sort_idx], y[sort_idx]
xMC, yMC = [x_sorted[0]], [y_sorted[0]]
for i in range(1, len(x_sorted)):
    if y_sorted[i] < yMC[-1]:
        xMC.append(x_sorted[i])
        yMC.append(y_sorted[i])

X, Y = {}, {}
for j in range(maxOrder):
    sort_idx = np.argsort(x)
    x_sorted, y_sorted = x[sort_idx], K_HP[j, sort_idx]
    x_lower, y_lowerHP = [x_sorted[0]], [y_sorted[0]]
    for i in range(1, len(x_sorted)):
        if y_sorted[i] < y_lowerHP[-1]:
            x_lower.append(x_sorted[i])
            y_lowerHP.append(y_sorted[i])
    X[j], Y[j] = np.array(x_lower), np.array(y_lowerHP)

opacity = 0.01
gamma = 1 - p
if showFigs:
    for j in range(maxOrder):
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(gamma, K_MC, c='k', alpha=opacity, label='MC')
        ax.set_xlabel(r'Thickness Deviation Factor - $\gamma$')
        ax.set_ylabel(r'Knockdown Factor - $\kappa$')
        ax.set_xscale('log')
        ax.set_xlim([1e-5, 1e0])
        ax.set_ylim([0.4, 1])
        plt.tight_layout()
        
        if saveFigs and j == 0:
            plt.savefig(f'Figures/Imperfection_VS_KDF_MC.png', dpi=DPI)

        Lc_HP_pristine = Lc_HP[j, id_min]
        ax.scatter(1 - p, Lc_HP[j, :] / Lc_HP_pristine, c=Color[j], alpha=opacity, label=f'HYPAD-FEM {j+1}')
        ax.set_xlim([1e-5, 5e-1])
        ax.set_ylim([0.4, 1.05])
        plt.tight_layout()
        
        if saveFigs:
            plt.savefig(f'Figures/Imperfection_VS_KDF_order{j+1}.png', dpi=DPI)
        plt.show()
        plt.close()

print("Generated design envelope figures")

runtime = (time.time() - notebook_time)/60
print("All done in "+str(runtime)+" minutes")
