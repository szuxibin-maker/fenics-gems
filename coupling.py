"""Run: python coupling.py case.yaml

This file owns input reading, data exchange, the time loop, and phi/De feedback.
It contains no GEMS calls, boundary equations, finite-element forms, or writers.
"""
from pathlib import Path
import sys
import traceback
import numpy as np
import yaml
from output import Output


def read_case(filename):
    """Read the single input file and check the shared simulation settings."""
    path = Path(filename).resolve()
    config = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
    if set(config) != {'chemistry', 'transport', 'coupling', 'output'}:
        raise ValueError('Use exactly four sections: chemistry, transport, coupling, output.')
    if config['coupling']['mode'] not in ('diffusion', 'chemistry', 'coupled'):
        raise ValueError('mode must be diffusion, chemistry, or coupled.')
    for section, key in [(config['transport']['geometry'], 'length'),
                         (config['coupling']['time'], 'dt'),
                         (config['coupling']['time'], 'end'),
                         (config['transport']['material'], 'phi'),
                         (config['transport']['material'], 'De')]:
        value = float(section[key])
        if not np.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be finite and positive.')
        section[key] = value
    if config['transport']['material']['phi'] > 1:
        raise ValueError('phi must be <= 1.')
    for key in ('update_porosity', 'update_diffusivity'):
        if type(config['coupling'][key]) is not bool:
            raise ValueError(f'{key} must be true or false.')
    if config['coupling']['update_diffusivity'] and not config['coupling']['update_porosity']:
        raise ValueError('Enable porosity feedback before diffusivity feedback.')
    if config['coupling']['mode'] == 'diffusion' and config['coupling']['update_porosity']:
        raise ValueError('Porosity feedback requires chemistry.')
    exponent = float(config['coupling']['exponent'])
    if not np.isfinite(exponent) or exponent <= 0:
        raise ValueError('The diffusivity exponent must be finite and positive.')
    config['coupling']['exponent'] = exponent
    if config['coupling']['mode'] != 'diffusion':
        config['chemistry']['project'] = str((path.parent / config['chemistry']['project']).resolve())
        if not Path(config['chemistry']['project']).is_file():
            raise ValueError('GEMS project file does not exist.')
    config['output']['folder'] = str((path.parent / config['output']['folder']).resolve())
    return config


def run(config):
    """Follow the physical workflow; return the final arrays and diagnostics."""
    output = Output(config)
    transport = None
    try:
        # 1. Initialise independent chemistry and transport objects.
        mode = config['coupling']['mode']
        phi0 = config['transport']['material']['phi']
        De0 = config['transport']['material']['De']
        inlet = None
        phases = []
        if mode == 'diffusion':
            names = config['transport']['components']
            if not names or len(set(names)) != len(names) or 'Zz' in names:
                raise ValueError('Use unique, nonempty tracer names; Zz is reserved for charge.')
            recipe = config['transport']['initial']
            if set(recipe) != set(names):
                raise ValueError('Provide every tracer in transport.initial.')
            initial = np.zeros(len(names))   # mol/m3 pore water; times phi per cell below
            for k, name in enumerate(names):
                initial[k] = float(recipe[name])
            if not np.isfinite(initial).all() or np.any(initial < 0):
                raise ValueError('Initial concentrations must be finite and nonnegative.')
        else:
            from chemistry import Chemistry
            chemistry = Chemistry(config['chemistry'])
            names = chemistry.names
            phases = chemistry.phases
            initial = chemistry.initial
            inlet = chemistry.inlet_c

        end = config['coupling']['time']['end']
        V = None
        if mode == 'chemistry':
            # One chemical state, without constructing a FEniCSx mesh.
            x = np.array([config['transport']['geometry']['length'] / 2])
            weights = np.array([config['transport']['geometry']['length']])
            end = 0.0
        else:
            from transport import Transport
            transport = Transport(config['transport'])
            x = transport.x
            weights = transport.weights
            V = transport.V
            boundary_c = transport.boundary_concentrations(names, inlet)

        # 2. Shared arrays: rows are cells, columns are elements in names.
        n = len(x)
        if transport is not None:
            # Per-cell porosity and De, from transport.material and transport.zones.
            phi0, De0, zones = transport.phi0.copy(), transport.De0.copy(), transport.zone
        else:
            phi0, De0, zones = np.full(n, phi0), np.full(n, De0), [None] * n
        if isinstance(initial, dict):
            # Zoned sample: each cell takes the recipe of its zone. Without a mesh
            # (chemistry mode) the first zone is used.
            rows = []
            for zone in zones:
                key = zone if zone is not None or mode != 'chemistry' else next(iter(initial))
                if key not in initial:
                    raise ValueError(f'chemistry.initial has no recipe for zone {key!r}.')
                rows.append(initial[key])
            total0 = np.array(rows)
        elif mode == 'diffusion':
            total0 = phi0[:, None] * initial   # tracer concentration -> mol/m3 bulk
        else:
            total0 = np.tile(initial, (n, 1))
        state = {'time': 0.0, 'step': 0,
                 'total': total0.copy(),
                 'aqueous': total0.copy(),
                 'phi': phi0.copy(), 'De': De0.copy(),
                 'pH': np.full(n, np.nan), 'minerals': np.zeros((n, len(phases)))}
        mobile = []
        for k, name in enumerate(names):
            if name != 'Zz':
                mobile.append(k)
        solid = np.zeros(n)
        water = state['phi'].copy()
        initial_mass = np.sum(state['total'] * weights[:, None], axis=0)
        exchange = np.zeros(len(names))
        max_balance = 0.0
        max_chemistry = 0.0
        max_water_mismatch = 0.0
        skipped = 0          # cell-steps left unreacted (chemistry.on_failure: skip)
        output.start(x, names, phases, V)

        # 3. Time loop. At step 0 only establish the initial equilibrium.
        while True:
            if state['step'] > 0:
                dt = min(config['coupling']['time']['dt'], end - state['time'])
                immobile = state['total'] - state['aqueous']
                state['aqueous'], incoming = transport.step(
                    state['aqueous'], state['phi'], state['De'], dt, boundary_c, mobile)
                state['total'] = immobile + state['aqueous']
                exchange += incoming
                state['time'] += dt

            # 4. Local reactions preserve total, but redistribute water and solids.
            chemistry_error = 0.0
            if mode != 'diffusion':
                for i in range(n):
                    try:
                        result = chemistry.equilibrate(state['total'][i], cell=i)
                    except RuntimeError as error:
                        message = f"Chemistry failed: step={state['step']}, cell={i}, x={x[i]:.6g} m. {error}"
                        # chemistry.on_failure: skip -> leave this cell unreacted for this
                        # step (transported water, previous minerals). Totals are unchanged,
                        # so mass is conserved. Default: stop.
                        if state['step'] == 0 or config['chemistry'].get('on_failure', 'stop') != 'skip':
                            raise RuntimeError(message) from error
                        skipped += 1
                        warning = f"WARNING skipped: step={state['step']}, cell={i}, x={x[i]:.6g} m (GEMS did not converge)"
                        print(warning, flush=True)
                        output.log.write(warning + '\n')
                        continue
                    state['aqueous'][i] = result['aqueous']
                    state['pH'][i] = result['pH']
                    state['minerals'][i] = result['minerals']
                    solid[i] = result['solid']
                    water[i] = result['water']
                    chemistry_error = max(chemistry_error, result['error'])
                if state['step'] == 0:
                    initial_solid = solid.copy()
                    # Only the porosity update needs room for solids. A sample whose GEMS
                    # state fills the whole bulk (pore water + solids = 1, e.g. an inert
                    # skeleton phase) sits at 0 up to round-off, so allow 0.1 % of the bulk.
                    overfill = phi0 + initial_solid - 1.0
                    if config['coupling']['update_porosity'] and np.any(overfill > 1e-3):
                        i = int(np.argmax(overfill))
                        raise ValueError(f'Initial solids plus porosity exceed the bulk volume by '
                                         f'{overfill[i]:.3g} at x={x[i]:.6g} m (phi0={phi0[i]:.4g}, '
                                         f'solids={initial_solid[i]:.4g}).')
                if config['coupling']['update_porosity']:
                    state['phi'] = phi0 + initial_solid - solid
                    if np.any(state['phi'] <= 0) or np.any(state['phi'] > 1):
                        raise RuntimeError('Porosity outside (0,1]; no clipping.')
                if config['coupling']['update_diffusivity']:
                    state['De'] = De0 * (state['phi'] / phi0) ** config['coupling']['exponent']
                # Keep aqueous amounts fixed when phi changes: c=aqueous/phi.

            # 5. Compare integrated totals with independently calculated boundary input.
            mass = np.sum(state['total'] * weights[:, None], axis=0)
            balance = mass - initial_mass - exchange
            water_mismatch = float(np.max(np.abs(water - state['phi']) / state['phi']))
            max_balance = max(max_balance, float(np.max(np.abs(balance))))
            max_chemistry = max(max_chemistry, chemistry_error)
            max_water_mismatch = max(max_water_mismatch, water_mismatch)
            if np.any(np.abs(balance) > 1e-9 + 1e-8 * (np.abs(initial_mass) + np.abs(exchange))):
                raise RuntimeError(f"Mass balance failed at step {state['step']}: {balance.tolist()}")
            diagnostics = {'mass': mass, 'exchange': exchange.copy(), 'balance': balance,
                           'chemistry_error': chemistry_error, 'water_mismatch': water_mismatch,
                           'max_balance': max_balance, 'max_chemistry': max_chemistry,
                           'max_water_mismatch': max_water_mismatch}
            output.write(state, diagnostics)
            if state['time'] >= end:
                break
            state['step'] += 1

        if skipped:
            note = f'WARNING: {skipped} cell-steps were left unreacted because GEMS did not converge.'
            print(note, flush=True)
            output.log.write(note + '\n')
        output.finish(state, diagnostics)
        return state, diagnostics
    except Exception:
        output.fail(traceback.format_exc())
        raise
    finally:
        if transport is not None:
            transport.close()
        output.close()


if __name__ == '__main__':
    filename = Path(__file__).with_name('case.yaml')
    if len(sys.argv) > 1:
        filename = Path(sys.argv[1])
    run(read_case(filename))
