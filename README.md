<div align='center'>
<h1> PyElastica-SCR-Extension </h1>
 </div>

Experimental plugin of Stable-Cosserat-Rod (SCR) implementation and its time-stepping scheme for PyElastica.

> This is an experimental extension of PyElastica. The code is still under development and may contain bugs. Use it at your own risk.

> Timestepping schema is different from the original PyElastica, and not every features are supported yet.

## Note on the model

- SCR achieves decouling of linear and angular momentum equation by treating the angular dynamics as quasi-static, which converts solving the orientation equation to be algebraic equation instead of ODE.
- This enables both full-implicit time-stepping scheme and larger stable time-step size.
- Physical interpretation of rotational stiffness becomes vague.
- All segments are quasi-static. In result, orientation updateds locally assuming the remaining orientation remains fixed.

## References

The implementation is based on the following paper with some modifications to fit into PyElastica framework:

- Hsu J., Wang T., Wu K., Yuksel C., "Stable Cosserat Rods" Proceedings of SIGGRAPH 2025, 2025.
    - https://graphics.cs.utah.edu/research/projects/stable-cosserat-rods/Stable_Cosserat_Rods-SIGGRAPH25.pdf
