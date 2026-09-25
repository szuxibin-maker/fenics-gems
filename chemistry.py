"""GEMS setup and local equilibrium only. No mesh, transport, or output imports."""
import numpy as np
import xgems


class Chemistry:
    """Keep the GEMS engine and its element/phase names in one place."""

    def __init__(self, config):
        self.config = config
        for key in ('temperature', 'pressure'):
            config[key] = float(config[key])
            if not np.isfinite(config[key]) or config[key] <= 0:
                raise ValueError(f'chemistry.{key} must be finite and positive.')
        self.g = xgems.ChemicalEngine(config['project'])
        self.retries = 0    # scaled re-solves needed so far (see equilibrate)
        # start: cold (default) = every call starts from scratch on one shared engine;
        # warm = one engine per cell, each call starts from that cell's previous result.
        self.start = config.get('start', 'cold')
        if self.start not in ('cold', 'warm'):
            raise ValueError('chemistry.start must be cold or warm.')
        self.engines = {}
        self.solved = set()
        self.last_b = {}
        for name, value in config['gibbs'].items():
            self.g.setStandardMolarGibbsEnergy(name, float(value))
        self.names = []
        for k in range(self.g.numElements()):
            self.names.append(self.g.elementName(k))
        self.phases = config['solid_phases']
        available = []
        for k in range(self.g.numPhases()):
            available.append(self.g.phaseName(k))
        for name in self.phases + [config['aqueous_phase']]:
            if name not in available:
                raise ValueError(f'Unknown GEMS phase: {name}')
        # Sample: one recipe, or {zone name: recipe} when transport.zones is used.
        spec = config['initial']
        if spec and all(isinstance(v, dict) for v in spec.values()):
            self.initial = {zone: self.amounts(recipe) for zone, recipe in spec.items()}
        else:
            self.initial = self.amounts(spec)
        # Boundary solution(s) for Dirichlet boundaries, as mol/m3 water:
        #   a recipe (both sides), the name of a zone (its equilibrated pore water),
        #   or {left: ..., right: ...} with either form per side. Optional.
        inlet = config.get('inlet')
        if inlet is None:
            self.inlet_c = None
        elif isinstance(inlet, dict) and inlet and set(inlet) <= {'left', 'right'}:
            self.inlet_c = {side: self.boundary_solution(v) for side, v in inlet.items()}
        else:
            self.inlet_c = self.boundary_solution(inlet)

    def boundary_solution(self, spec):
        """Equilibrate a recipe, or a zone's sample, and return mol/m3 of its pore water."""
        if isinstance(spec, str):
            if not isinstance(self.initial, dict) or spec not in self.initial:
                raise ValueError(f'chemistry.inlet names zone {spec!r}, which has no recipe in chemistry.initial.')
            total = self.initial[spec]
        else:
            total = self.amounts(spec)
        state = self.equilibrate(total)
        # Use the equilibrated solution's entire aqueous-phase volume.
        c = state['aqueous'] / state['water']
        # Charge (Zz) is never transported; its dissolved total is 0 up to round-off.
        # Round-off can also leave other tiny negative totals; set those to 0.
        if 'Zz' in self.names:
            c[self.names.index('Zz')] = 0.0
        c[(c < 0) & (c > -1e-9 * np.max(np.abs(c)))] = 0.0
        return c

    def amounts(self, recipe):
        """Convert YAML amounts (mol/1 dm3 bulk) to mol/m3 in GEMS element order."""
        total = np.zeros(len(self.names))
        for name, value in recipe.items():
            if name not in self.names:
                raise ValueError(f'Unknown GEMS element: {name}')
            if not np.isfinite(float(value)) or float(value) < 0:
                raise ValueError('Recipe amounts must be finite and nonnegative.')
            total[self.names.index(name)] = float(value) * 1000.0
        return total

    def engine(self, cell):
        """The shared engine, or (start: warm) the engine that belongs to one cell."""
        if self.start != 'warm' or cell is None:
            return self.g
        if cell not in self.engines:
            g = xgems.ChemicalEngine(self.config['project'])
            for name, value in self.config['gibbs'].items():
                g.setStandardMolarGibbsEnergy(name, float(value))
            self.engines[cell] = g
        return self.engines[cell]

    def equilibrate(self, total, cell=None):
        """One local equilibrium. Element and mineral amounts are mol/m3 bulk."""
        g = self.engine(cell)
        b = total / 1000.0  # Restore the original 1 dm3 reference sample.
        T, P = self.config['temperature'], self.config['pressure']
        def solve(warm, target, size):
            if warm:
                g.setWarmStart()
            else:
                g.setColdStart()
            status = g.equilibrate(T, P, target * size)
            actual = np.asarray(g.formulaMatrix()) @ g.speciesAmounts() / size
            ok = status in (2, 6) and np.allclose(actual, target, rtol=1e-8, atol=1e-8)
            return ok, status, actual

        ok, size = False, 1.0
        if cell in self.solved:
            # start: warm. 1) Start from this cell's own previous equilibrium.
            ok, status, actual = solve(True, b, 1.0)
            if not ok and cell in self.last_b:
                # 2) Continuation: re-solve the previous composition, then approach
                #    the new one in four steps, each warm-started from the last.
                self.retries += 1
                base = self.last_b[cell]
                if solve(False, base, 1.0)[0]:
                    for fraction in (0.25, 0.5, 0.75, 1.0):
                        ok, status, actual = solve(True, base + fraction * (b - base), 1.0)
                        if not ok:
                            break
        if not ok:
            # 3) Cold starts. Equilibrium does not depend on system size, so when GEMS
            #    stops with a poor element balance, solve again at a scaled size and
            #    divide back (as GEMS_ADE_1D_Template/simple/chemistry.py does).
            for size in (1.0, 1.0, 10.0, 100.0, 0.1, 0.01):
                ok, status, actual = solve(False, b, size)
                if ok:
                    break
                self.retries += 1
        if not ok:
            raise RuntimeError(f'GEMS status={status}; total_mol_dm3={b.tolist()}; residual={(actual-b).tolist()}')
        if self.start == 'warm' and cell is not None:
            self.solved.add(cell)
            self.last_b[cell] = b.copy()

        aq_id = g.indexPhase(self.config['aqueous_phase'])
        aqueous = np.asarray(g.elementAmountsInPhase(aq_id)).copy() / size * 1000.0
        volumes = np.asarray(g.phaseVolumes()) / size * 1000.0  # m3/sample -> volume fractions
        minerals = []
        solid = 0.0
        for name in self.phases:
            phase = g.indexPhase(name)
            minerals.append(g.phaseAmount(phase) / size * 1000.0)
            solid += volumes[phase]
        values = np.concatenate((aqueous, minerals, [g.pH(), solid, volumes[aq_id]]))
        if not np.isfinite(values).all() or volumes[aq_id] <= 0:
            raise RuntimeError('Non-finite chemical state or no aqueous phase.')
        return {'aqueous': aqueous, 'pH': g.pH(), 'minerals': minerals,
                'solid': solid, 'water': volumes[aq_id],
                'error': float(np.max(np.abs(actual - b)) * 1000.0)}
