import numpy as np
import pyoti.sparse as oti

def updRefConfiguration(nel, angle, fi, U, n1, n2, dof, x, y):
    L_1     = oti.zeros((1,nel))
    fi_1    = oti.zeros((6,nel))
    angle_1 = oti.zeros((1,nel))
    for i in range(nel):
        beamlength    = elemLength(i, U, n1, n2, dof, x, y)
        L_1[0, i]     = beamlength
        angle_1[0, i] = angle[0, i]
        fi_1[:, i]    = fi[:, i]
    return L_1, angle_1, fi_1
def tangStiffMtxUL(neq, nel, U, n1, n2, dof, x, y, E, A, I, fi, gle, formulation):
    # Ke = oti.zeros((6, 6, nel))
    Ke = []
    Kt = oti.zeros((neq, neq))
    for i in range(nel):
        ke = elasticStiffMtxUL(i, U, n1, n2, dof, x, y, E[i,0], A[i,0], I[i,0], formulation)
        kg = geometricStiffMtxUL(i, U, n1, n2, dof, x, y, A[i,0], I[i,0], fi, formulation)
        kt = ke + kg

        # Store elastic stiffness matrix to be used in computation of internal forces
        Ke.append(ke)

        # Rotation matrix from local to global coordinate system
        angle = elemAngle(i, U, n1, n2, dof, x, y)
        c = oti.cos(angle)
        s = oti.sin(angle)
        
        rot = oti.zeros((6, 6))
        rot[0, 0] =  c
        rot[0, 1] = -s
        rot[1, 0] =  s
        rot[1, 1] =  c
        rot[2, 2] =  1.0
        rot[3, 3] =  c
        rot[3, 4] = -s
        rot[4, 3] =  s
        rot[4, 4] =  c
        rot[5, 5] =  1.0
        
        # Tangent stiffness matrix in global system
        k = oti.dot(rot, oti.dot(kt, oti.transpose(rot)))

        # Assemble element matrix to global matrix
        for m in range(6):
            for n in range(6):
                Kt[int(gle[i, m]), int(gle[i, n])] += k[m, n]
    return Kt, Ke
def elasticStiffMtxUL(i, U, n1, n2, dof, x, y, E, A, I, formulation):
    
    # Element properties
    EA = E * A
    L = elemLength(i, U, n1, n2, dof, x, y)
    
    if formulation.lower() == 'frame':
        # Simplifications
        L2 = L * L
        L3 = L2 * L
        EI = E * I
        t12 = EA / L
        t13 = 12 * EI / L3
        t14 = 6 * EI / L2
        t15 = 4 * EI / L
        t16 = 2 * EI / L
        
        ke = oti.zeros((6, 6))
        ke[0, 0] =  t12; ke[0, 3] = -t12
        ke[1, 1] =  t13; ke[1, 2] =  t14; ke[1, 4] = -t13; ke[1, 5] =  t14
        ke[2, 1] =  t14; ke[2, 2] =  t15; ke[2, 4] = -t14; ke[2, 5] =  t16        
        ke[3, 0] = -t12; ke[3, 3] =  t12        
        ke[4, 1] = -t13; ke[4, 2] = -t14; ke[4, 4] =  t13; ke[4, 5] = -t14        
        ke[5, 1] =  t14; ke[5, 2] =  t16; ke[5, 4] = -t14; ke[5, 5] =  t15
    
    else:  # truss
        ke = oti.array([
            [EA / L, 0, 0, -EA / L, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [-EA / L, 0, 0, EA / L, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0]
        ])

    return ke
def geometricStiffMtxUL(i, U, n1, n2, dof, x, y, A, I, fi, formulation):
    # Element properties
    L = elemLength(i, U, n1, n2, dof, x, y)

    if formulation.lower() == 'frame':
        P = fi[3, i]
        M1 = fi[2, i]
        M2 = fi[5, i]

        # Simplifications
        L2  = L * L
        L3  = L2 * L
        PI  = P * I
        AL  = A * L
        AL2 = A * L2
        AL3 = A * L3
        t0 = (P / L)
        t1 = t0 * (1.0)
        t2 = t0 * (6.0/5.0)
        t3 = t0 * (L/10.0)
        t4 = t0 * (2.0*L2/15.0)
        t5 = t0 * (L2/30.0)
        
        # First order terms 
        kg1 = oti.zeros((6,6))
        kg1[0,0] = t1; kg1[0,3] =-t1
        kg1[1,1] = t2; kg1[1,2] = t3; kg1[1,4] =-t2; kg1[1,5] = t3
        kg1[2,1] = t3; kg1[2,2] = t4; kg1[2,4] =-t3; kg1[2,5] =-t5     
        kg1[3,0] =-t1; kg1[3,3] = t1
        kg1[4,1] =-t2; kg1[4,2] =-t3; kg1[4,4] = t2; kg1[4,5] =-t3
        kg1[5,1] = t3; kg1[5,2] =-t5; kg1[5,4] =-t3; kg1[5,5] = t4
        
        # Second order terms       
        t6  = M1 / L
        t7  = M2 / L
        t8  = 12 * PI / AL3
        t9  = 6 * PI / AL2
        t10 = 4 * PI / AL
        t11 = 2 * PI / AL
        
        kg2 = oti.zeros((6,6))
        kg2[0, 2] = -t6;  kg2[0, 5] = -t7
        kg2[1, 1] =  t8;  kg2[1, 2] =  t9;  kg2[1, 4] = -t8;   kg2[1, 5] =  t9
        kg2[2, 0] = -t6;  kg2[2, 1] =  t9;  kg2[2, 2] =  t10;  kg2[2, 3] =  t6;  kg2[2, 4] = -t9;  kg2[2, 5] =  t11
        kg2[3, 2] =  t6;  kg2[3, 5] =  t7
        kg2[4, 1] = -t8;  kg2[4, 2] = -t9;  kg2[4, 4] =  t8;   kg2[4, 5] = -t9
        kg2[5, 0] = -t7;  kg2[5, 1] =  t9;  kg2[5, 2] =  t11;  kg2[5, 3] =  t7;  kg2[5, 4] = -t9;  kg2[5, 5] =  t10
        
    else:  # truss
        P = fi[3, i]
        kg1 = (P / L) * oti.array([
            [ 0, 0,  0,  0, 0,  0],
            [ 0, 1,  0,  0, -1,  0],
            [ 0, 0,  0,  0, 0,  0],
            [ 0, 0,  0,  0, 0,  0],
            [ 0, -1,  0,  0, 1,  0],
            [ 0, 0,  0,  0, 0,  0]
        ])
        # Second order terms
        kg2 = oti.zeros((6, 6))

    # Geometric matrix
    kg = kg1 + kg2
    return kg
def intForcesUL(neq, nel, U, D_U, update_angle, n1, n2, dof, x, y, gle, fi_1, angle_1, L_1, Ke):
    # Initialize global vector of internal forces
    F  = oti.zeros((neq,1))
    fi = oti.zeros((6, nel))
    if update_angle:
        angle = oti.zeros((1, nel))
    for i in range(nel):
        N1 = n1[i]
        N2 = n2[i]

        # Lengths: Beginning of step, current, and step increment
        L_o = L_1[0, i]
        L_c = elemLength(i, U, n1, n2, dof, x, y)
        D_L = L_c - L_o

        # Rigid body rotation from step beginning and current angle
        rbr = elemAngleIncr(i, U, D_U, n1, n2, dof, x, y)
        angle_tmp = angle_1[0, i] + rbr[0, 0]

        # Update element angle
        if update_angle:
            angle[0, i] = angle_tmp

        # Rotation matrix from local to global coordinate system
        c = oti.cos(angle_tmp)
        s = oti.sin(angle_tmp)
        
        rot = oti.zeros((6, 6))
        rot[0, 0] =  c
        rot[0, 1] = -s
        rot[1, 0] =  s
        rot[1, 1] =  c
        rot[2, 2] =  1.0
        rot[3, 3] =  c
        rot[3, 4] = -s
        rot[4, 3] =  s
        rot[4, 4] =  c
        rot[5, 5] =  1.0

        # Relative rotations
        r1 = D_U[int(dof[N1, 2])] - rbr[0, 0]
        r2 = D_U[int(dof[N2, 2])] - rbr[0, 0]

        dl = oti.zeros(6,1)
        dl[2,0] = r1
        dl[3,0] = D_L
        dl[5,0] = r2

        # Increment/Total internal local force
        D_fl = oti.dot(Ke[i], dl)
        fi[:, i] = fi_1[:, i] + D_fl

        # Transform and assemble internal forces
        fg = oti.dot(rot, fi[:, i])    
        for m in range(6):
            F[int(gle[i, m]), 0] += fg[m, 0]        
    return F, fi, angle
def predictedIncrement(neqf, s, J, Pref, D_lr, D_U, d_Ut):
    Pref = Pref[:neqf]
    D_U  = D_U[:neqf]
    d_Ut = d_Ut[:neqf]
    # d_lr = J * np.sqrt((D_U.T @ D_U + D_lr**2 * (Pref.T @ Pref)) / (d_Ut.T @ d_Ut + Pref.T @ Pref))
    d_lr = J * ((oti.dot(oti.transpose(D_U), D_U) + D_lr**2 * (oti.dot(oti.transpose(Pref), Pref))) / (oti.dot(oti.transpose(d_Ut), d_Ut) + oti.dot(oti.transpose(Pref), Pref)))**0.5
    d_lr = s * d_lr
    return d_lr
def correctedIncrement(neqf, d_Ut, d_Ur):
    d_Ut = d_Ut[:neqf]
    d_Ur = d_Ur[:neqf]
    # d_lr = -(d_Ut.T @ d_Ur) / (d_Ut.T @ d_Ut)
    d_lr = -oti.dot(oti.transpose(d_Ut), d_Ur) / oti.dot(oti.transpose(d_Ut), d_Ut)
    return d_lr
def elemLength(i, U, n1, n2, dof, x, y):
    # Element nodes
    N1 = int(n1[i])
    N2 = int(n2[i])
    X1 = x[N1,0] + U[int(dof[N1, 0]),0]
    Y1 = y[N1,0] + U[int(dof[N1, 1]),0]
    X2 = x[N2,0] + U[int(dof[N2, 0]),0]
    Y2 = y[N2,0] + U[int(dof[N2, 1]),0]

    # Element length
    dX = X2 - X1
    dY = Y2 - Y1
    L  = (dX**2 + dY**2)**0.5
    return L
def elemAngle(i, U, n1, n2, dof, x, y):
    N1 = int(n1[i])
    N2 = int(n2[i])
    X1 = x[N1,0] + U[int(dof[N1, 0]),0]
    Y1 = y[N1,0] + U[int(dof[N1, 1]),0]
    X2 = x[N2,0] + U[int(dof[N2, 0]),0]
    Y2 = y[N2,0] + U[int(dof[N2, 1]),0]

    # Element length
    dX = X2 - X1
    dY = Y2 - Y1

    real_angle = np.arctan2(dY.real, dX.real)
    angle = oti.atan(dY/dX)
    angle.set_deriv(real_angle, 0)
    return angle
def elemAngleIncr(i, U, d_U, n1, n2, dof, x, y):
    # Displacements on previous configuration
    U_0 = U - d_U

    N1 = int(n1[i])
    N2 = int(n2[i])

    # Old nodal displacements
    dx1_0 = U_0[int(dof[N1, 0]), 0]
    dy1_0 = U_0[int(dof[N1, 1]), 0]
    dx2_0 = U_0[int(dof[N2, 0]), 0]
    dy2_0 = U_0[int(dof[N2, 1]), 0]

    # Old nodal coordinates
    x1_0 = x[N1,0] + dx1_0
    y1_0 = y[N1,0] + dy1_0
    x2_0 = x[N2,0] + dx2_0
    y2_0 = y[N2,0] + dy2_0

    # Old axial direction vector
    vx_0 = oti.array([x2_0-x1_0, y2_0-y1_0, 0])

    # New nodal displacements
    dx1_1 = U[int(dof[N1, 0]), 0]
    dy1_1 = U[int(dof[N1, 1]), 0]
    dx2_1 = U[int(dof[N2, 0]), 0]
    dy2_1 = U[int(dof[N2, 1]), 0]

    # New nodal coordinates
    x1_1 = x[N1,0] + dx1_1
    y1_1 = y[N1,0] + dy1_1
    x2_1 = x[N2,0] + dx2_1
    y2_1 = y[N2,0] + dy2_1

    # Axial direction vector
    vx_1 = oti.array([x2_1-x1_1, y2_1-y1_1, 0])

    # Increment of angle
    dir = oti_cross(vx_0, vx_1)
    # ang_incr = np.arctan2(np.linalg.norm(dir), np.dot(vx_0, vx_1))
    real_angle = np.arctan2(oti.norm(dir).real, oti.dot_product(vx_0, vx_1).real)
    ang_incr = oti.atan(oti.norm(dir)/oti.dot_product(vx_0, vx_1))
    ang_incr.set_deriv(real_angle, 0)

    # Angle direction
    s = np.sign(dir[2].real)
    ang_incr = s * ang_incr
    return ang_incr
def checkSingularMtx(neqf, K):
    singular = 0
    if np.linalg.cond(K[:neqf, :neqf].real) < 10e-12:
        singular = 1
    return singular
def solveLinearSystem(neqf, neqc, K, P):
    Kff = K[:neqf, :neqf]
    Pf  = P[:neqf, 0]
    Ds  = oti.zeros((neqc,1))
    Df  = oti.solve(Kff, Pf)
    D   = oti.zeros((neqc+neqf,1)) # np.concatenate((Df, Ds))
    D[:neqf, 0] = Df
    D[neqf:, 0] = Ds
    return D
def oti_cross(a,b):
    c = oti.zeros((a.shape))
    c[0,0] = a[1,0]*b[2,0]-a[2,0]*b[1,0]
    c[1,0] = a[2,0]*b[0,0]-a[0,0]*b[2,0]
    c[2,0] = a[0,0]*b[1,0]-a[1,0]*b[0,0]
    return c
def constantsToProfile(c, theta):
    # x = Normalize(theta)
    a = np.min(theta.real)
    b = np.max(theta.real)
    x = (2 * (theta - a) / (b - a)) - 1

    # Chebyshev polynomials
    T_0 = oti.ones(x.shape)
    T_1 = x
    T_2 = 2*x**2 - 1
    T_3 = 4*x**3 - 3*x
    T_4 = 8*x**4 - 8*x**2 + 1
    T_5 = 16*x**5 - 20*x**3 +  5*x
    T_6 = 32*x**6 - 48*x**4 + 18*x**2 -  1
    T_7 = 64*x**7 - 112*x**5 + 56*x**3 - 7*x
    T_8 = 128*x**8 - 256*x**6 + 160*x**4 - 32*x**2 + 1

    y = c[0]*T_0 + c[1]*T_1 + c[2]*T_2 + c[3]*T_3 + c[4]*T_4 + c[5]*T_5 + c[6]*T_6 + c[7]*T_7 + c[8]*T_8
    return y
