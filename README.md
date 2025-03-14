# Neural Differential Equations with Adjoint Sensitivity Method

This project explores the use of Neural Differential Equations (NDEs) to model free-fall dynamics with drag, and it demonstrates how the Adjoint Sensitivity Method can be used for efficient gradient computation.

## Project Files

- **Group_18_Discussion.pdf**  
  The complete report detailing our discussion topic, including theory, derivations, and experimental comparisons.

- **Neural Ordinary Differential Equations.pdf**  
  A primary reference that provides the theoretical background, proofs, and experimental results related to Neural Differential Equations.

- **pytorch_demo.py**  

- **julia_demo.jl**  

## How to Run the Code

### PyTorch (Python) Version

1. **Install Dependencies:**

   Ensure you have Python installed (preferably 3.7+). Then install the required packages via pip:

   ```bash
   pip install torch matplotlib torchdiffeq
   ```

2. **Run the Script:**

   Execute the PyTorch demonstration code from your terminal:

   ```bash
   python pytorch_demo.py
   ```

   This will generate the free-fall simulation, train the Neural ODE model, and display the plots for the free-fall dynamics and training loss.

### Julia Version

1. **Install Julia:**

   Download and install Julia from [julialang.org](https://julialang.org/downloads/).

2. **Install Required Packages:**

   In the Julia REPL, install the necessary packages:

   ```julia
   using Pkg
   Pkg.add("Flux")
   Pkg.add("DifferentialEquations")
   Pkg.add("DiffEqFlux")
   Pkg.add("Plots")
   Pkg.add("Printf")
   ```

3. **Run the Script:**

   From the terminal or Julia REPL, run the Julia demonstration code:

   ```bash
   julia julia_demo.jl
   ```

   The script will run the simulation, train the Neural ODE model, and display the plots for the predicted dynamics and training loss.
