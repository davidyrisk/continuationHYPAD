# continuationHYPAD

## Paper and Contributers
Developed by David Y. Risk-Mora and collaborators. 
See our published paper:
1. <a href="https://.../doi/..." style="color:#268cd7"> **David Y. Risk-Mora**, Mauricio Aristizabal, Harry Millwater, David Restrepo, "*Efficient Uncertainty Quantification for Critical Buckling Loads and Determination of Knockdown Factors in Nonlinear Structures*", n/a, n/a, (2025).</a>

### Citation
If you use this framework/dataset, build on or find our research useful for your work please cite as, 
```
@article{EfficientCriticalBucklingLoadUQ_2025,
  title={Efficient Uncertainty Quantification for Critical Buckling Loads and Determination of Knockdown Factors in Nonlinear Structures},
  author={Risk-Mora, David Y and Aristizabal, Mauricio and Millwater, Harry and Restrepo, David},
  journal={n/a},
  volume={n/a},
  number={n/a},
  pages={n/a},
  year={2025},
  publisher={n/a}
}

```

## Overview
`continuationHYPAD` is a Python library for nonlinear finite element analysis using continuation methods. It enables efficient computation of post-buckling and critical buckling loads by leveraging hypercomplex automatic differentiation (HYPAD) within finite element models (FEM). The package provides a solver for nonlinear structural problems and includes utilities for sensitivity analysis, imperfection modeling, and reduced-order modeling.

### Features
- **Continuation Solver**: Implements an arc-length continuation method for nonlinear FEM problems.
- **Hypercomplex Automatic Differentiation (HYPAD)**: Enables efficient sensitivity analysis.
- **Imperfection Modeling**: Uses Chebyshev polynomials for reduced-order imperfection representation.

### Installation
To install `continuationHYPAD`, clone the repository and install using `pip`:
```bash
git clone https://github.com/....git
cd continuationHYPAD
pip install .
```

Ensure you have the required dependencies:
```bash
pip install -r requirements.txt
```

### Requirements
This package is tested on:
- **Operating System:** Ubuntu running under Windows Subsystem for Linux (WSL)
- **Python:** 3.8+

### Dependencies:
- `numpy`
- `scipy`
- `matplotlib`
- `pyoti`

### Repository Structure
```
continuationHYPAD/
│── continuationHYPAD/
│   ├── __init__.py
│   ├── continuationHYPAD.py
│   ├── utils.py
│── setup.py
│── requirements.txt
│── README.md
│── examples/
```

# Function Documentation
## `continuationHYPAD.continuationSolver`
`continuationSolver` implements a nonlinear finite element method (FEM) solver using a continuation method. This function incrementally applies the load while updating the stiffness matrix to track the structure's response, even through instabilities.

### Inputs
- **formulation** *(str)*: Structural formulation type (currently "frame" is the only verified/supported option).
- **Coords** *(oti.array)*: Nodal coordinates (nnodes x 2).
- **Connect** *(np.array)*: Element connectivity (nelems x 2).
- **ndof** *(int)*: Degrees of freedom per node.
- **Ebc** *(np.array)*: Boundary condition matrix (nnodes x ndof).
- **Loads** *(oti.array)*: Applied external forces (nnodes x ndof).
- **E** *(oti.array)*: Young’s modulus per element.
- **A** *(oti.array)*: Cross-sectional area per element.
- **I** *(oti.array)*: Second moment of area per element.
- **export_node** *(int)*: Index of the node for exporting displacement data.
- **max_steps** *(int, optional)*: Maximum continuation steps (default: 1000).
- **init_incr** *(float, optional)*: Initial arc-length step size (default: 0.00075).
- **trg_iter** *(int, optional)*: Target iterations per step (default: 3).
- **max_ratio** *(float, optional)*: Maximum applied load ratio (default: 1.00).
- **tol** *(float, optional)*: Convergence tolerance for residual forces (default: 1e-05).
- **max_iter** *(int, optional)*: Maximum iterations per step (default: 500).
- **showConvergenceData** *(bool, optional)*: Whether to print convergence details per step (default: False).

### Returns
- **U_step** *(list of oti.array)*: Displacement history at each continuation step.
- **lr_step** *(list of oti.num)*: Load ratio history at each step.
- **DOFexport** *(list of int)*: Indices of degrees of freedom for exporting results.

### Key Processes
1. **Initialization**:
   - Extracts mesh connectivity and nodal degrees of freedom.
   - Assembles the global stiffness matrix.
   - Sets up boundary conditions and external loads.

2. **Incremental Load Application**:
   - Iteratively applies load in small increments.
   - Updates the reference configuration for the Updated Lagrangian (UL) formulation.
   - Computes the tangent stiffness matrix.
   - Checks for singularities in the system.

3. **Predictor-Corrector Scheme**:
   - Predicts the next equilibrium state using a tangent stiffness update.
   - Corrects the solution iteratively until convergence is achieved.

4. **Convergence Check**:
   - Ensures the residual force is below the given tolerance.
   - Stops if convergence fails or the load ratio limit is reached.

5. **Postprocessing**:
   - Stores displacement and load ratio history for further analysis.
   - Allows exporting results for a selected degree of freedom.

This function enables efficient continuation-based solution tracking of nonlinear structural problems, making it suitable for postbuckling and stability analysis.

### Usage
```python
import continuationHYPAD

# Define input parameters (example)
Coords = ...
Connect = ...
ndof = 3
Ebc = ...
Loads = ...
E = ...
A = ...
I = ...
export_node = 5

# Run the solver
U_step, lr_step, DOFexport = continuationHYPAD.continuationSolver(
    "frame", Coords, Connect, ndof, Ebc, Loads, E, A, I, export_node)
```

### Accessing Utility Functions `continuationHYPAD.utils`

## List of Utility Functions
1.  **`updRefConfiguration(nel, angle, fi, U, n1, n2, dof, x, y)`**
    
    *   **Inputs**:
        *   `nel`: Number of elements
        *   `angle`: Element angles
        *   `fi`: Internal force vector
        *   `U`: Displacement vector
        *   `n1, n2`: Element connectivity arrays
        *   `dof`: Degrees of freedom mapping
        *   `x, y`: Nodal coordinates
    *   **Outputs**:
        *   `L_1`: Updated element lengths
        *   `angle_1`: Updated element angles
        *   `fi_1`: Updated internal forces
    *   **Process**: Computes updated element lengths, angles, and internal forces for use in updated reference configurations.

2.  **`tangStiffMtxUL(neq, nel, U, n1, n2, dof, x, y, E, A, I, fi, gle, formulation)`**
    
    *   **Inputs**: Stiffness matrix parameters, element connectivity, nodal coordinates, and material properties.
    *   **Outputs**:
        *   `Kt`: Global tangent stiffness matrix
        *   `Ke`: List of element stiffness matrices
    *   **Process**: Computes the tangent stiffness matrix by assembling the elastic and geometric stiffness matrices for all elements.

3.  **`elasticStiffMtxUL(i, U, n1, n2, dof, x, y, E, A, I, formulation)`**
    
    *   **Inputs**: Element index, displacements, material properties, and formulation type.
    *   **Outputs**: Local elastic stiffness matrix for an element.
    *   **Process**: Computes the elastic stiffness matrix based on frame or truss formulations.

4.  **`geometricStiffMtxUL(i, U, n1, n2, dof, x, y, A, I, fi, formulation)`**
    
    *   **Inputs**: Element index, displacements, forces, and section properties.
    *   **Outputs**: Local geometric stiffness matrix for an element.
    *   **Process**: Computes the geometric stiffness matrix considering axial and moment contributions.

5.  **`intForcesUL(neq, nel, U, D_U, update_angle, n1, n2, dof, x, y, gle, fi_1, angle_1, L_1, Ke)`**
    
    *   **Inputs**: FEM system parameters, element forces, and displacement increments.
    *   **Outputs**:
        *   `F`: Global internal force vector
        *   `fi`: Updated element forces
        *   `angle`: Updated element angles
    *   **Process**: Computes internal forces considering rigid body rotation and stiffness contributions.

6.  **`predictedIncrement(neqf, s, J, Pref, D_lr, D_U, d_Ut)`**
    
    *   **Inputs**: Free DOFs, step size, previous increments, load vector, displacement updates.
    *   **Outputs**: Predicted load increment (`d_lr`).
    *   **Process**: Computes a load increment based on previous displacements using an arc-length method.

7.  **`correctedIncrement(neqf, d_Ut, d_Ur)`**
    
    *   **Inputs**: Free DOFs, tangent displacement, residual displacement increments.
    *   **Outputs**: Corrected load increment (`d_lr`).
    *   **Process**: Uses the residual displacement correction to update the load increment.

8.  **`elemLength(i, U, n1, n2, dof, x, y)`**
    
    *   **Inputs**: Element index, displacements, nodal coordinates.
    *   **Outputs**: Current element length.
    *   **Process**: Computes the length of an element considering displacements.

9.  **`elemAngle(i, U, n1, n2, dof, x, y)`**
    
    *   **Inputs**: Element index, displacements, nodal coordinates.
    *   **Outputs**: Current element angle.
    *   **Process**: Computes the orientation of an element.

10.  **`elemAngleIncr(i, U, d_U, n1, n2, dof, x, y)`**
    
    *   **Inputs**: Element index, total and incremental displacements, nodal coordinates.
    *   **Outputs**: Increment of angle due to deformation.
    *   **Process**: Computes the rigid body rotation increment for an element.

11.  **`checkSingularMtx(neqf, K)`**
    
    *   **Inputs**: Free DOFs, stiffness matrix.
    *   **Outputs**: Boolean indicating if the matrix is singular.
    *   **Process**: Checks if the stiffness matrix is singular by evaluating its condition number.

12.  **`solveLinearSystem(neqf, neqc, K, P)`**
    
    *   **Inputs**: Free and constrained DOFs, stiffness matrix, force vector.
    *   **Outputs**: Displacement increments.
    *   **Process**: Solves the linear system using a partitioned stiffness matrix approach.

13.  **`oti_cross(a, b)`**
    
    *   **Inputs**: Two vectors (`a`, `b`).
    *   **Outputs**: Cross product of the vectors.
    *   **Process**: Computes the 3D cross product of two vectors.

14.  **`constantsToProfile(c, theta)`**
    
    *   **Inputs**: Chebyshev coefficients (`c`), spatial coordinate (`theta`).
    *   **Outputs**: Function profile evaluated at `theta`.
    *   **Process**: Uses Chebyshev polynomials to generate a smooth profile from coefficients.

### Usage
You import utility functions:
```python
from continuationHYPAD.utils import function
result = function(args)
```

Or access them through the module:
```python
import continuationHYPAD.utils
result = continuationHYPAD.utils.function(args)
```

## License
MIT License
Copyright (c) 2025 David Y. Risk-Mora

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
