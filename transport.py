"""Mesh, boundary conditions, diffusion and optional advection. No chemistry imports.

One DG0 value represents one uniform 1D cell. Face terms implement a
two-point finite-volume flux (diffusion) and an upwind flux (advection, when
material.darcy_flux > 0); time integration uses backward Euler.
"""
import numpy as np
from mpi4py import MPI
from petsc4py import PETSc
from dolfinx import fem, mesh
from dolfinx.fem.petsc import assemble_matrix, assemble_vector
import ufl


class Transport:
    """Keep the mesh and solver together for reuse at each time step."""

    def __init__(self, config):
        geometry = config['geometry']
        boundary = config['boundary']
        solver = config['solver']
        if MPI.COMM_WORLD.size != 1:
            raise ValueError('This version supports one MPI process.')
        self.L = float(geometry['length'])
        n = geometry['cells']
        if self.L <= 0 or not np.isfinite(self.L) or type(n) is not int or n < 1:
            raise ValueError('Use a positive length and a positive integer cell count.')
        if float(solver['rtol']) <= 0 or solver['max_iterations'] < 1:
            raise ValueError('Use positive solver tolerance and iteration count.')
        self.h = self.L / n
        self.boundary = boundary
        for side in ('left', 'right'):
            if boundary[side]['type'] not in ('dirichlet', 'no_flux', 'outflow'):
                raise ValueError('Boundary type must be dirichlet, no_flux, or outflow.')
        # Darcy flux q = phi * pore velocity, m/s, flowing in +x. Optional; 0 = pure diffusion.
        self.q = float(config.get('material', {}).get('darcy_flux', 0.0))
        if not np.isfinite(self.q) or self.q < 0:
            raise ValueError('material.darcy_flux must be finite and >= 0 (flow towards +x).')
        if self.q > 0 and boundary['left']['type'] != 'dirichlet':
            raise ValueError('With flow, the left (inflow) boundary must be dirichlet.')
        if self.q > 0 and boundary['right']['type'] == 'no_flux':
            raise ValueError('With flow, the right boundary must be outflow or dirichlet.')

        self.domain = mesh.create_interval(MPI.COMM_WORLD, n, [0.0, self.L])
        self.V = fem.functionspace(self.domain, ('DG', 0))
        self.x = self.V.tabulate_dof_coordinates()[:, 0].copy()
        self.weights = np.full(n, self.h)  # Integrating along x gives mol/m2.

        # Material zones (optional). A cell belongs to a zone when its centre lies in
        # [from, to); the zone may override phi and De. Cells outside every zone keep
        # transport.material and have zone name None.
        material = config.get('material', {})
        self.phi0 = np.full(n, float(material.get('phi', 1.0)))
        self.De0 = np.full(n, float(material.get('De', 1.0)))
        self.zone = [None] * n
        names = []
        for z in config.get('zones') or []:
            name = str(z['name'])
            if name in names:
                raise ValueError(f'Zone name {name!r} is used twice.')
            names.append(name)
            lo, hi = float(z.get('from', 0.0)), float(z.get('to', self.L))
            inside = (self.x >= lo) & (self.x < hi)
            if not inside.any():
                raise ValueError(f'Zone {name!r} [{lo}, {hi}) contains no cell centre.')
            for key, array in (('phi', self.phi0), ('De', self.De0)):
                if key in z:
                    array[inside] = float(z[key])
            for i in np.flatnonzero(inside):
                self.zone[i] = name
        if np.any(self.phi0 <= 0) or np.any(self.phi0 > 1) or np.any(self.De0 <= 0):
            raise ValueError('Every cell needs 0 < phi <= 1 and De > 0 (material or zone).')
        self.edge_cells = [int(np.argmin(self.x)), int(np.argmax(self.x))]
        self.phi = fem.Function(self.V)
        self.De = fem.Function(self.V)
        self.old = fem.Function(self.V)
        self.c = fem.Function(self.V)
        self.dt = fem.Constant(self.domain, PETSc.ScalarType(1.0))
        self.values = [fem.Constant(self.domain, PETSc.ScalarType(0.0)),
                       fem.Constant(self.domain, PETSc.ScalarType(0.0))]

        # Boundary location and boundary equations both belong here.
        left = mesh.locate_entities_boundary(self.domain, 0, self.left_boundary)
        right = mesh.locate_entities_boundary(self.domain, 0, self.right_boundary)
        indices = np.concatenate((left, right))
        labels = np.concatenate((np.full(len(left), 1), np.full(len(right), 2))).astype(np.int32)
        order = np.argsort(indices)
        tags = mesh.meshtags(self.domain, 0, indices[order], labels[order])
        ds = ufl.Measure('ds', domain=self.domain, subdomain_data=tags)
        u = ufl.TrialFunction(self.V)
        v = ufl.TestFunction(self.V)
        # DG0 has zero gradient inside a cell. These face fluxes carry diffusion.
        D_face = 2 * self.De('+') * self.De('-') / (self.De('+') + self.De('-'))
        diffusion = D_face / self.h * ufl.jump(u) * ufl.jump(v) * ufl.dS
        rhs = self.old * v * ufl.dx  # old: aqueous amount per BULK volume.
        # Advection, upwind: q*c leaves each cell through its downstream face.
        # qn = q.n on a face; q_out = max(qn, 0) is the part leaving that side.
        normal = ufl.FacetNormal(self.domain)
        qn = self.q * normal[0]
        q_out = (qn + abs(qn)) / 2
        q_in = (abs(qn) - qn) / 2
        if self.q > 0:
            diffusion += (v('+') - v('-')) * (q_out('+') * u('+') - q_out('-') * u('-')) * ufl.dS
        for side, label in enumerate(('left', 'right')):
            kind = boundary[label]['type']
            if kind == 'dirichlet':
                # The boundary is h/2 from the cell centre.
                diffusion += 2 * self.De / self.h * u * v * ds(side + 1)
                rhs += self.dt * 2 * self.De / self.h * self.values[side] * v * ds(side + 1)
            if self.q > 0 and kind in ('dirichlet', 'outflow'):
                diffusion += q_out * u * v * ds(side + 1)                 # carried out
            if self.q > 0 and kind == 'dirichlet':
                rhs += self.dt * q_in * self.values[side] * v * ds(side + 1)  # carried in
        self.a = fem.form(self.phi * u * v * ufl.dx + self.dt * diffusion)
        self.rhs = fem.form(rhs)
        self.solver = PETSc.KSP().create(self.domain.comm)
        self.solver.setType(solver['ksp_type'])
        self.solver.getPC().setType(solver['pc_type'])
        self.solver.setTolerances(rtol=float(solver['rtol']), atol=1e-14,
                                 max_it=solver['max_iterations'])

    def left_boundary(self, points):
        return np.isclose(points[0], 0.0)

    def right_boundary(self, points):
        return np.isclose(points[0], self.L)

    def boundary_concentrations(self, names, inlet=None):
        """Use the chemical inlet, or read tracer concentrations from YAML."""
        values = np.zeros((2, len(names)))
        for side, label in enumerate(('left', 'right')):
            if self.boundary[label]['type'] == 'dirichlet':
                if isinstance(inlet, dict):
                    if label not in inlet:
                        raise ValueError(f'chemistry.inlet gives no solution for the {label} boundary.')
                    values[side] = inlet[label]
                elif inlet is not None:
                    values[side] = inlet
                else:
                    recipe = self.boundary[label]['concentrations']
                    if set(recipe) != set(names):
                        raise ValueError('Provide every component at each Dirichlet boundary.')
                    for k, name in enumerate(names):
                        values[side, k] = float(recipe[name])
        if not np.isfinite(values).all() or np.min(values) < -1e-12:
            side, k = np.unravel_index(np.argmin(values), values.shape)
            raise ValueError(f'Boundary concentrations must be finite and nonnegative; '
                             f'{("left", "right")[side]} {names[k]} = {values[side, k]:.3g} mol/m3.')
        return values

    def step(self, aqueous, phi, De, dt, boundary_c, mobile):
        """Return aqueous amounts (mol/m3 bulk) and boundary input (mol/m2)."""
        self.phi.x.array[:] = phi
        self.De.x.array[:] = De
        self.dt.value = PETSc.ScalarType(dt)
        A = assemble_matrix(self.a)
        A.assemble()
        self.solver.setOperators(A)
        updated = aqueous.copy()
        exchange = np.zeros(aqueous.shape[1])
        try:
            for k in mobile:
                self.old.x.array[:] = aqueous[:, k]
                self.values[0].value = PETSc.ScalarType(boundary_c[0, k])
                self.values[1].value = PETSc.ScalarType(boundary_c[1, k])
                b = assemble_vector(self.rhs)
                try:
                    self.solver.solve(b, self.c.x.petsc_vec)
                finally:
                    b.destroy()
                if self.solver.getConvergedReason() <= 0:
                    raise RuntimeError(f'Transport failed for component column {k}.')
                if not np.isfinite(self.c.x.array).all() or np.min(self.c.x.array) < -1e-12:
                    raise RuntimeError(f'Invalid concentration in component column {k}; no clipping.')
                for side, label in enumerate(('left', 'right')):
                    kind = self.boundary[label]['type']
                    i = self.edge_cells[side]
                    qn = -self.q if side == 0 else self.q   # q.n, n pointing out of the column
                    flux_in = 0.0
                    if kind == 'dirichlet':
                        flux_in += 2 * De[i] / self.h * (boundary_c[side, k] - self.c.x.array[i])
                        flux_in += max(-qn, 0.0) * boundary_c[side, k]
                    if kind in ('dirichlet', 'outflow'):
                        flux_in -= max(qn, 0.0) * self.c.x.array[i]
                    exchange[k] += dt * flux_in
                updated[:, k] = phi * self.c.x.array
        finally:
            A.destroy()
        return updated, exchange

    def close(self):
        self.solver.destroy()
