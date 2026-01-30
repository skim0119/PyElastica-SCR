<div align='center'>
<h1> PyElastica-SCR-Extension </h1>
 </div>

Experimental plugin of Stable-Cosserat-Rod (SCR) and implicit time-stepping scheme for PyElastica.

> This is an experimental extension of PyElastica. The code is still under development and may contain bugs. Use it at your own risk.

> Timestepping schema is different from the original PyElastica, and not every features are supported yet.

## Dev status

- [x] Migrate PyElastica structure for rod implementation and timestepper.
- [x] Implement SCR within PyElastica rod.
    - [x] Rotation equation
    - [x] Implicit update for translation
    - [x] Incorporate external force and couple
    - [x] Angular velocity and acceleration (not used during stepping, but for user information)
        - Include as part of `update_orientation`
    - [ ] numba optimization
    - [ ] use solve_banded for solve operations.
    - [ ] GPU
- [x] Single-rod case without any external force (butterfly)
- [ ] Single-rod cases with general setup
    - [ ] Axial stretching
    - [x] Catenary
    - [ ] Timoshenko beam
- [ ] CROEF (CR one-end-fixed) implementation
    - [ ] SCR with BC
        - [x] Position + velocity
        - [ ] Orientation + angular velocity
- [ ] Consider block implementation with ghosting (not sure if it beneficial)
- [ ] Logging
    - [ ] Residual monitoring
    - [ ] Reactive forces
- [ ] Revisit
    - [ ] SOR iteration for orientation finding: constitutive relation can help
## Note on the model

- SCR achieves decouling of linear and angular momentum equation by treating the angular dynamics as quasi-static, which converts solving the orientation equation to be algebraic equation instead of ODE.
- This enables both full-implicit time-stepping scheme and larger stable time-step size.
- Physical interpretation of rotational stiffness becomes vague.
- Variable dependencies has changed. Original PyElastica treats velocity and rotation states (director, omega, alpha) as part of the state variable. In the implementation of SCR and implicit stepping, velocity and accelerations are not an independent state variable, and all rotation variables are directly solved from simplified differential equation.
- All segments are quasi-static. In result, orientation updateds locally assuming the remaining orientation remains fixed.

## Physical discrepancies

These are few points in the implementation that was handled without careful concerns of physical nature of rod. They are assumed to be close to physical behavior (either the error can be iteratively reduced or comparable to the time-stepper accuracy), but maybe worth bringing up in the future.

- Solving the rotation equation directly (by setting `internal_torques` to be zero) yield `internal_couple` and `kappa`, but obtaining `director` from curvature is not well-posed. Currently, it uses SOR-style iteration from previous `director` to find the closest directors that match the curvature, but this is technically not a physical dynamics.
    - As a result, slight bit of rotation shows in butterfly example. (I am not 100% sure if this is the cause, but seems roughly related.)

## References

The implementation is based on the following paper with some modifications to fit into PyElastica framework:

- Hsu J., Wang T., Wu K., Yuksel C., "Stable Cosserat Rods" Proceedings of SIGGRAPH 2025, 2025.
    - https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/Stable_Cosserat_Rods-SIGGRAPH25.pdf
