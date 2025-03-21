import numpy as np
import time
import pyoti.sparse as oti
from .utils import elemLength, elemAngle, checkSingularMtx, updRefConfiguration, predictedIncrement, correctedIncrement, solveLinearSystem, oti_cross, intForcesUL, tangStiffMtxUL

def continuationSolver(formulation, Coords, Connect, ndof, Ebc, Loads, E, A, I, export_node, max_steps=1000, init_incr=0.00075, trg_iter=3, max_ratio=1.00, tol=1e-05, max_iter=500, showConvergenceData=False): 
    """
    Solve the nonlinear FEM problem using a continuation method.

    Parameters:
    -----------
    formulation : str
        Type of structural formulation ("frame", "shell", etc.).
    Coords : np.array
        Nodal coordinates array (nnodes x 2).
    Connect : np.array
        Element connectivity (nelems x 2).
    ndof : int
        Degrees of freedom per node.
    Ebc : np.array
        Boundary conditions (nnodes x ndof).
    Loads : np.array
        Applied loads (nnodes x ndof).
    E : np.array
        Young’s modulus array per element.
    A : np.array
        Cross-sectional area per element.
    I : np.array
        Second moment of area per element.
    export_node : int
        Node index for exporting displacement data.
    max_steps : int, optional
        Maximum number of continuation steps.
    init_incr : float, optional
        Initial arc-length step size.
    trg_iter : int, optional
        Number of target iterations.
    max_ratio : float, optional
        Load ratio limit.
    tol : float, optional
        Convergence tolerance.
    max_iter : int, optional
        Maximum iterations per step.

    Returns:
    --------
    U_step : list
        List of displacement solutions per step.
    lr_step : list
        List of load ratio values per step.
    """

	# Mesh coordinates and connectivity
    x  =  Coords[:, 0]
    y  =  Coords[:, 1]
    n1 = Connect[:, 0]
    n2 = Connect[:, 1]
    nnp = Coords.shape[0]
    nel = Connect.shape[0]
    neq = ndof * nnp

    # Assemble nodes DOF ids matrix
    neqc = 0
    dof  = np.zeros((nnp, 3), dtype=int)
    for i in range(nnp):
        for j in range(3):
            if Ebc[i, j] == 1:
                neqc += 1
                dof[i, j] = 1

    # Assign equation numbers for free and constrained degrees of freedom  
    # Free DOFs are numbered first (countF), followed by constrained DOFs (countS).  
    # This ensures that equations are correctly indexed for solving the system.  
    neqf = neq - neqc
    countS = neqf
    countF = 0
    for i in range(nnp):
        for j in range(3):
            if dof[i, j] == 0:
                dof[i, j] = countF
                countF += 1
            else:
                dof[i, j] = countS
                countS += 1

    # Assemble element gather vectors
    gle = np.zeros((nel, 6), dtype=int)
    for i in range(nel):
        for j in range(6):
            if j <= 2:
                gle[i, j] = dof[Connect[i, 0], j]
            else:
                gle[i, j] = dof[Connect[i, 1], j - 3]

    # Assemble load vector (based on Neumann BCs)
    Pref = oti.zeros((neq, 1))
    for i in range(nnp):
        for j in range(3):
            Pref[int(dof[i, j]), 0] += Loads[i, j]

    # Find export DOF ID
    DOFexport = []
    DOFexport.append(int(dof[export_node, 0]))
    DOFexport.append(int(dof[export_node, 1]))
    
    # Initialize element data (preloads, initial orientations/extensions, "SVAR" initialization)
    n1      = Connect[:, 0]
    n2      = Connect[:, 1]
    L       = oti.zeros((1, nel))
    L_1     = oti.zeros((1, nel))
    fi      = oti.zeros((6, nel))
    fi_1    = oti.zeros((6, nel))
    angle   = oti.zeros((1, nel))
    angle_1 = oti.zeros((1, nel))

    # Element loop
    for i in range(nel):
        N1 = int(n1[i])
        N2 = int(n2[i])

        dY = y[N2]-y[N1]
        dX = x[N2]-x[N1]

        # Element orientation
        vl = oti.array([dX, dY, 0.0])
        vx = vl / oti.norm(vl)
        vz = oti.array([0.0, 0.0, 1.0])
        vy = oti_cross(vz,vx) 
        
        # Assemble rotation transformation matrix for one node
        T                  = oti.zeros((3,3))  #
        T[0,:]             = oti.transpose(vx) # np.vstack((vx, vy, vz))
        T[1,:]             = oti.transpose(vy) # 
        T[2,:]             = oti.transpose(vz) #
        L[0, i]            = oti.norm(vl)     # np.linalg.norm(vl)
        L_1[0, i]          = oti.norm(vl)     # np.linalg.norm(vl) << Pre-elongation    
        angle[0, i]        = oti.atan(dY/dX)
        angle_1[0, i]      = oti.atan(dY/dX)

        real_angle = np.arctan2(dY.real, dX.real)[0][0]
        angle[0, i]   += real_angle - angle[0, i].real
        angle_1[0, i] += real_angle - angle_1[0, i].real

    # Initialize solution variables
    lr      = 0.0
    U       = oti.zeros((neq,1))
    U_step  = []
    lr_step = []

    # Start incremental process
    step = 0
    while step < max_steps:
        step += 1
        start_step = time.time()

        # Update reference configuration for UL formulation
        L_1, angle_1, fi_1 = updRefConfiguration(nel, angle, fi, U, n1, n2, dof, x, y)
        Kt, ke = tangStiffMtxUL(neq, nel, U, n1, n2, dof, x, y, E, A, I, fi, gle, formulation)
        if checkSingularMtx(neqf, Kt):
            print(f'Singular tangent matrix!\nStep = {step}\nIter = 0')
            break

        # Tangent increment of displacements for predicted solution
        d_Ut = solveLinearSystem(neqf, neqc, Kt, Pref)
        if step == 1:
            s = 1
            d_lr = init_incr
            nrm2 = oti.dot_product(d_Ut, d_Ut)
            oti_dlr = False
        else:
            # Generalized Stiffness Parameter
            GSP = nrm2 / oti.dot_product(d_Ut, d_Ut_1) 
            if GSP.real < 0:
                s = -s

            J = 1
            d_lr = predictedIncrement(neqf, s, J, Pref, D_lr, D_U, d_Ut)
            oti_dlr = True

        # Limit increment of load ratio
        if ((max_ratio > 0.0 and (lr + d_lr).real > max_ratio) or (max_ratio < 0.0 and (lr + d_lr).real < max_ratio)):
            d_lr = max_ratio - lr
            oti_dlr = False

        # Store predicted increment of displacement and load
        while isinstance(d_lr, (list, np.ndarray)) and len(d_lr) == 1:
            d_lr = d_lr[0]
        if oti_dlr:
            d_lr = d_lr[0,0]
        d_U    = d_lr * d_Ut
        d_lr_1 = d_lr
        d_Ut_1 = d_Ut
        D_lr   = d_lr
        D_U    = d_U

        # Total values of load ratio and displacements
        lr += d_lr
        U  += d_U
        
        # Start corrective iterations
        iter = 0
        conv = 0
        while conv == 0 and iter < max_iter:            
            P = lr * Pref
            F, fi, angle = intForcesUL(neq, nel, U, D_U, 1, n1, n2, dof, x, y, gle, fi_1, angle_1, L_1, ke)

            # Unbalanced forces
            R = P - F
        
            # Check for single-step method or convergence
            stopping_criteria = oti.norm(R[:neqf]) / oti.norm(Pref[:neqf])
            conv = stopping_criteria.real < tol
            if conv == 1:
                break

            # Start/keep corrective cycle of iterations
            iter += 1

            # Tangent stiffness matrix
            Kt, ke = tangStiffMtxUL(neq, nel, U, n1, n2, dof, x, y, E, A, I, fi, gle, formulation)
            if checkSingularMtx(neqf, Kt):
                print(f'Singular tangent matrix!\nStep = {step}\nIter = {iter}')
                break

            # Tangent and residual increments of displacements
            d_Ut = solveLinearSystem(neqf, neqc, Kt, Pref)
            d_Ur = solveLinearSystem(neqf, neqc, Kt, R)
            
            # Corrected increment of load ratio
            d_lr = correctedIncrement(neqf, d_Ut, d_Ur)

            # Store corrected increment of displacement and load
            while isinstance(d_lr, (list, np.ndarray)) and len(d_lr) == 1:
                d_lr = d_lr[0]
            d_lr = d_lr[0,0]
            d_U   = d_lr * d_Ut + d_Ur
            D_lr += d_lr
            D_U  += d_U
            lr   += d_lr
            U    += d_U
        #end - iterations
        
        # Check for convergence fail reached or complex value of increment
        if conv == 0:
            print(f'Convergence not reached!\nStep = {step}\nIter = {iter}')
            break
        # end
        
        # Store equilibrium configuration
        finish_step = time.time()
        step_time = finish_step - start_step
        U_step.append(U.copy())
        lr_step.append(lr)

        if showConvergenceData:
            print(f'Step:{step} | ratio:{lr.real:.2f} | Iter:{iter} | Time:{step_time}')
        
        # Check if maximum load ratio was reached
        if lr.real >= 0.9999 * max_ratio:
            break
    #end - step increments

    return U_step, lr_step, DOFexport