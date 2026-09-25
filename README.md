# FEniCSx-GEMS: four modules, one input file

Edit `case.yaml`, then run:

```bash
python coupling.py case.yaml
```

On this Windows computer, double-click `Run.cmd`. It uses the installed WSL Python:

```bash
/home/xi_b/miniforge3/envs/fenicsx/bin/python coupling.py case.yaml
```

## Structure

```text
FEniCSx_GEMS_Beginner/
|-- chemistry.py
|-- transport.py
|-- coupling.py
|-- output.py
|-- case.yaml
|-- chemistry_data/       # Required GEMS project files
|-- results/              # Generated results
|-- README.md
|-- Run.cmd
`-- previous_version.zip  # Backup of the previous code, inputs, and results
```

There are exactly four active Python files and one input YAML. No separate main script or configuration module is needed.

| Python file | YAML section | Responsibility |
|---|---|---|
| `chemistry.py` | `chemistry` | Load GEMS, read element/phase names, initialise recipes, calculate local equilibrium |
| `transport.py` | `transport` | Create the mesh, define boundary conditions, assemble and solve diffusion |
| `coupling.py` | `coupling` | Read the YAML, initialise shared arrays, run the time loop, exchange data, update phi/De, check conservation |
| `output.py` | `output` | Write CSV/XDMF, plots, progress logs, summaries, and failure reports |

Chemistry does not import transport. Transport does not import chemistry. Coupling calls both; output only receives their results.

The small classes `Chemistry`, `Transport`, and `Output` keep their engine, solver, or open files together. A class here simply groups related data and the functions that use it; there is no inheritance or plugin framework. Read the numbered steps in `coupling.run()` to follow the full calculation.

## Browser tools

Open the HTML files locally in a browser, for example by double-clicking them after downloading or cloning this repository. No Python installation or local server is needed to use the pages. Both include embedded example data; use the file controls to inspect your current files.

### `case_builder.html` — build a `case.yaml`

[GEMS Case Builder](case_builder.html) provides a form for the four configuration sections: chemistry, transport, coupling, and output. It updates a YAML preview as you edit, checks basic input values and loaded chemical names, and lets you **Copy** or **Download** the configuration.

1. Start from the embedded example or use **Import case.yaml** to load an existing basic case.
2. Under Chemistry, click **Choose GEMS folder** and select the folder containing the project `.lst` and its DCH data (`.json` or text `.dat`). Select the required list file if several are found. The loaded data supplies element, phase, and species names. Check that `chemistry.project` is relative to the location where you will save `case.yaml`.
3. Set the sample and boundary recipes, temperature, pressure, phases, and optional Gibbs energy overrides. Then set the 1D geometry, porosity, diffusivity, boundaries, solver, calculation mode, time settings, feedback, and output location. Time units selected in the form are converted to seconds in YAML.
4. Review validation messages and the preview, click **Download**, and place the resulting `case.yaml` beside the Python modules (or in your chosen case folder with suitable paths). Run it separately with `python coupling.py case.yaml` from an environment containing FEniCSx and GEMS.

The page prepares configuration files; it does not run simulations or verify chemical convergence. Its exporter covers the basic single-material diffusion case. Advanced settings such as zones, Darcy flux, outflow boundaries, separate left/right chemical recipes, and warm-start/failure options are not supported by the form and are not reliably preserved on export. Edit those cases directly in YAML. YAML import requires the externally loaded `js-yaml` library to be available; the page reports an error if it cannot load it.

### `gems_inspector.html` — inspect GEMS chemistry data

[GEMS Chemistry Inspector](gems_inspector.html) makes the exported chemical system readable before preparing a case. Click **Open folder…**, **Open files…**, or drag files onto the page. Include the `.lst` and the DCH/IPM files it references; choose the required list file when several are present. JSON (`-j`) and GEMS text (`-t`) exports are supported; binary (`-b`) exports are not.

The tabs show independent components (elements and charge), phases and their species, searchable species properties, and the stoichiometry matrix. Species details include composition, charge, molar mass, and available standard thermodynamic properties. Use the temperature–pressure selector when the DCH contains multiple stored points.

In the **Reactions** tab, enter a reaction using exact, case-sensitive DCH species names, for example `H2O@ = H+ + OH-`. Separate terms with spaces around `+`, and place coefficients before names, such as `2 H2O@`. The page checks element and charge balance and calculates reaction Gibbs energy, enthalpy, entropy, heat capacity, volume, and `log10(K) = -ΔrG° / (R T ln(10))` from the selected DCH data.

The inspector is a read-only data viewer and thermodynamic calculator. The current loader skips DBR state files, so it does not display a solved equilibrium state or run GEMS. Values come from stored DCH points, without interpolation or the in-memory `chemistry.gibbs` overrides in `case.yaml`.

## The four YAML sections

```yaml
chemistry:
  # GEMS project, temperature, pressure, phase names, and chemical recipes

transport:
  geometry:
    # Length and cell count
  material:
    # Initial phi and De
  components:
    # Tracer names, used only in diffusion mode
  initial:
    # Tracer concentrations, used only in diffusion mode
  boundary:
    # Left/right boundary types and tracer concentrations
  solver:
    # PETSc solver settings

coupling:
  mode: coupled
  time:
    # Time step and end time
  # Porosity/diffusivity feedback switches and exponent

output:
  # Result folder and profile-saving interval
```

The actual file contains all values. Do not add extra top-level sections.

Use `coupling.mode` to choose `coupled`, `diffusion`, or `chemistry`. The default is coupled cement diffusion with porosity and De feedback. `chemistry` calculates one equilibrium state without constructing a FEniCSx mesh. `diffusion` uses the tracer entries under `transport` and does not load GEMS.

For the earlier diffusion example, edit the same file: select `diffusion`, disable both feedback switches, set De to `1.0e-10`, dt to `100`, end to `80000`, and the right boundary to `dirichlet`. The supplied tracer boundary values are 1 on the left and 0 on the right. To preserve existing results, change `output.folder` before a new run.

In coupled mode, chemical element names and initial amounts come from `chemistry`. Every Dirichlet boundary uses the equilibrated `chemistry.inlet`; the tracer concentration entries are inactive. A `no_flux` boundary never exchanges material.

## How data moves between modules

| Quantity | Meaning | Units |
|---|---|---|
| `state['total']` | All elements, including aqueous and immobile parts | mol/m3 bulk |
| `state['aqueous']` | Elements in the aqueous phase | mol/m3 bulk |
| `c = aqueous / phi` | Dissolved concentration | mol/m3 pore water |
| `phi` | Porosity | Dimensionless |
| `De` | Effective diffusion coefficient in the flux equation | m2/s |

Arrays have one row per cell and one column per element. GEMS determines the chemical element order.

Each time step in `coupling.py` does the following:

```python
immobile = state['total'] - state['aqueous']
# transport.step(...) returns updated aqueous amounts and boundary input.
state['total'] = immobile + state['aqueous']
# chemistry.equilibrate(...) redistributes these totals among water and minerals.
# Update phi and De, check conservation, then call output.write(...).
```

Local equilibrium preserves element totals. Only aqueous element amounts move during diffusion. The chemical example transports Ca, Cl, H, O, and Si, including water's H/O. Charge Zz is not transported.

During each transport substep, phi is fixed:

```text
phi * (c_new - c_old) / dt = div(De * grad(c_new))
```

The scheme is backward Euler with a uniform 1D DG0 two-point flux. DG0 has one value per cell; diffusion is carried by face fluxes, using the harmonic mean of De. This is not the DG1/SWIP method in `ADE_DG.py`.

Feedback in `coupling.py` is:

```python
phi = phi0 + initial_solid - solid
De = De0 * (phi / phi0) ** exponent
```

Solid quantities here are volume fractions. Unrepresented initial solid volume is fixed inert material. When phi changes, aqueous amounts stay fixed and concentrations are recalculated as `aqueous / phi`.

## Chemical data and limitations

The four GEMS input files were copied unchanged from `GEMS_ADE_1D_Template/gems/`. The recipes and explicit Gibbs energy adjustments, H2O@ = -237183 J/mol and Ca+2 = -552400 J/mol, come from that template's `cement.py`. These are example-specific adjustments applied in memory.

Chemical recipes use mol per 1 dm3 bulk reference volume; arrays use mol/m3 bulk. GEMS phase volumes are converted from m3 per reference sample to volume fractions. The inlet concentration uses its entire equilibrated aqueous-phase volume.

Independent cold starts avoid transferring a neighbouring cell's solver state. Floating-point input changes still cause about 3e-5 pH drift and 1e-4 relative Ca concentration differences in the closed example. These small differences are solver precision effects.

The demonstration assumes saturated diffusion. The aqueous-volume/porosity mismatch is reported, not corrected with a water-flow calculation. The previous example showed about 0.25% mismatch. The feedback exponent is prescribed, not fitted to experiments.

Current scope: one material, one GEMS system, uniform 1D mesh, one process, fixed boundaries, common De, and equilibrium reactions. No 2D/3D, XCT, electromigration, kinetics, or reactive convergence study is claimed.

## Results and verification

The output folder contains `profiles.csv`, `mass_balance.csv`, `profiles.png`, `summary.txt`, `simulation.log`, and `case_used.json`. Transport modes also produce `fields.xdmf` and `fields.h5`. The JSON snapshot records actual settings while keeping just one YAML input file.

Inventories and boundary exchanges are in mol/m2 for unit cross-sectional area. Conservation checks compare current inventory minus initial inventory against independently calculated cumulative boundary input. GEMS element reconstruction is checked separately using its stoichiometry matrix and species amounts.

Failures stop the run, write `failure.txt`, and mark the summary FAILED. Completed runs are marked COMPLETED. No negative concentration or invalid porosity is silently clipped.

Refactoring checks are recorded in `results/verification.json`. They compare the new results with the archived version and check analytical diffusion, standalone chemistry, closed boundaries, and failure reporting. The older code, test script, configurations, and results remain recoverable from `previous_version.zip`.

## Advection (optional)

`transport.material.darcy_flux` (m/s, flow towards +x) adds upwind advection:
`phi*dc/dt + d/dx(q*c - De*dc/dx) = 0`. Leave it out or set 0 for pure diffusion.
With flow, the left boundary must be `dirichlet` (inflow) and the right one `outflow`
(advective outflow, zero dispersive gradient) or `dirichlet`. Boundary inflow and outflow
are both counted in the mass balance. Example: `exsample/demo_bench5` (Bench5, MgCl2 into calcite).

## Zones and boundary solutions (optional)

`transport.zones` splits the column into named parts, each with its own `phi` and `De`:

```yaml
transport:
  zones:
    - {name: clay, from: 0.0, to: 0.25, phi: 0.51, De: 5.1e-10}
    - {name: cement, from: 0.25, to: 0.5, phi: 0.52, De: 5.2e-10}
chemistry:
  initial:              # one recipe per zone name (or a single recipe for the whole column)
    clay: {Ca: ..., Si: ...}
    cement: {Ca: ..., Si: ...}
  inlet: {left: clay, right: {Na: ..., Cl: ...}}   # optional
```

A cell belongs to a zone when its centre lies in `[from, to)`. `chemistry.inlet` may be one
recipe (all Dirichlet boundaries), a zone name (that zone's equilibrated pore water), or
`{left: ..., right: ...}` with either form per side. It can be left out when no boundary is
Dirichlet. Examples: `example/demo_NaCl_diffusion`, `demo_cement_clay_interface`, `demo_cebama`.

## GEMS start mode (optional)

`chemistry.start: cold` (default) solves every cell from scratch on one shared engine.
`chemistry.start: warm` gives every cell its own engine and starts each step from that
cell's previous equilibrium; a failed warm start falls back to cold starts with scaled
system sizes. Use `warm` for large systems that do not converge from cold starts
(example: `example/demo_cebama`).

With `warm`, a cell that fails is first re-solved at its previous composition and then moved
to the new one in four warm-started steps. `chemistry.on_failure: skip` (default `stop`)
leaves a cell that still fails unreacted for that step; totals are unchanged, and every skip
is printed and logged with a total at the end.
