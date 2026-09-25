"""All result files, progress messages, plots, and failure reports live here."""
from pathlib import Path
import csv
import json
import os
import numpy as np


class Output:
    """Keep output files together; no transport or chemistry is calculated here."""

    def __init__(self, config):
        self.config = config
        every = config['output']['every']
        if type(every) is not int or every < 1:
            raise ValueError('output.every must be a positive integer.')
        self.folder = Path(config['output']['folder']).resolve()
        self.folder.mkdir(parents=True, exist_ok=True)
        self.profile_file = None
        self.balance_file = None
        self.fields = None
        self.log = (self.folder / 'simulation.log').open('w', encoding='utf-8', buffering=1)
        (self.folder / 'summary.txt').write_text('RUNNING; results are incomplete.\n', encoding='utf-8')
        # A JSON snapshot preserves the settings without adding another input YAML.
        (self.folder / 'case_used.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
        (self.folder / 'failure.txt').unlink(missing_ok=True)
        self.previous_folder = Path.cwd()
        os.chdir(self.folder)  # Native GEMS logs also belong in the result folder.

    def start(self, x, names, phases, V=None):
        """Prepare tables; create an XDMF file only when a transport mesh exists."""
        self.x = x
        self.names = names
        self.phases = phases
        self.profile_file = (self.folder / 'profiles.csv').open('w', newline='', buffering=1)
        self.profiles = csv.writer(self.profile_file)
        header = ['time_s', 'x_m', 'phi', 'De_m2_s', 'pH']
        for name in names:
            header.extend([name + '_c_mol_m3_water', name + '_total_mol_m3_bulk'])
        for name in phases:
            header.append(name + '_mol_m3_bulk')
        self.profiles.writerow(header)
        self.balance_file = (self.folder / 'mass_balance.csv').open('w', newline='', buffering=1)
        self.balances = csv.writer(self.balance_file)
        self.balances.writerow(['step', 'time_s', 'component', 'inventory_mol_m2',
                               'cumulative_net_in_mol_m2', 'balance_error_mol_m2',
                               'chemistry_error_mol_m3', 'water_relative_mismatch'])
        if V is not None:
            from dolfinx import fem, io
            self.fields = io.XDMFFile(V.mesh.comm, str(self.folder / 'fields.xdmf'), 'w')
            self.fields.write_mesh(V.mesh)
            self.field = fem.Function(V)

    def write(self, state, diagnostics):
        """Save balances every step and profiles at the configured interval."""
        time = state['time']
        step = state['step']
        for k, name in enumerate(self.names):
            self.balances.writerow([step, time, name, diagnostics['mass'][k],
                                    diagnostics['exchange'][k], diagnostics['balance'][k],
                                    diagnostics['chemistry_error'], diagnostics['water_mismatch']])
        end = self.config['coupling']['time']['end']
        if self.config['coupling']['mode'] == 'chemistry':
            end = 0
        if step % self.config['output']['every'] != 0 and time < end:
            return
        for i in np.argsort(self.x):
            row = [time, self.x[i], state['phi'][i], state['De'][i], state['pH'][i]]
            for k in range(len(self.names)):
                row.extend([state['aqueous'][i, k] / state['phi'][i], state['total'][i, k]])
            row.extend(state['minerals'][i])
            self.profiles.writerow(row)
        if self.fields is not None:
            for k, name in enumerate(self.names):
                if name != 'Zz':
                    self.field.name = name + '_c_mol_m3_water'
                    self.field.x.array[:] = state['aqueous'][:, k] / state['phi']
                    self.fields.write_function(self.field, time)
            for name in ('phi', 'De'):
                self.field.name = name
                self.field.x.array[:] = state[name]
                self.fields.write_function(self.field, time)
            if self.phases:
                self.field.name = 'pH'
                self.field.x.array[:] = state['pH']
                self.fields.write_function(self.field, time)
        message = (f"step={step}, time={time:.6g} s, "
                   f"balance={diagnostics['max_balance']:.3e} mol/m2, "
                   f"chemistry={diagnostics['chemistry_error']:.3e} mol/m3, "
                   f"water mismatch={diagnostics['water_mismatch']:.2%}")
        print(message, flush=True)
        self.log.write(message + '\n')

    def finish(self, state, diagnostics):
        """Finish tables before plotting, then mark the run as completed."""
        self.profile_file.close()
        self.balance_file.close()
        if self.fields is not None:
            self.fields.close()
            self.fields = None
        plot_profiles(self.folder, self.names, self.phases)
        config = self.config
        summary = f'''COMPLETED
Mode: {config['coupling']['mode']}
Cells/states: {len(self.x)}; reference length: {config['transport']['geometry']['length']} m
Scheme: backward Euler, DG0 two-point cell flux (transport modes only)
Linear solver: {config['transport']['solver']['ksp_type']} / {config['transport']['solver']['pc_type']}
Final time: {state['time']} s; steps: {state['step']}
Amounts: mol/m3 bulk; concentrations: mol/m3 pore water
Inventory and boundary exchange: mol/m2 (unit cross-sectional area)
Maximum balance error: {diagnostics['max_balance']:.6e} mol/m2
Maximum chemical element reconstruction error: {diagnostics['max_chemistry']:.6e} mol/m3
Maximum aqueous-volume/porosity mismatch: {diagnostics['max_water_mismatch']:.6%}
GEMS initialisation: independent cold start for every equilibrium call
Final porosity range: {state['phi'].min():.9g} to {state['phi'].max():.9g}
Final De range: {state['De'].min():.9g} to {state['De'].max():.9g} m2/s
Porosity feedback: {config['coupling']['update_porosity']}
Diffusivity feedback: {config['coupling']['update_diffusivity']}
Limit: numerical demonstration, not experimental validation.
Limit: saturated pore volume is assumed; aqueous volume mismatch is diagnostic only.
'''
        (self.folder / 'summary.txt').write_text(summary, encoding='utf-8')
        self.log.write(summary)
        print('Saved:', self.folder, flush=True)

    def fail(self, message):
        (self.folder / 'failure.txt').write_text(message, encoding='utf-8')
        (self.folder / 'summary.txt').write_text('FAILED\n' + message, encoding='utf-8')
        self.log.write(message)

    def close(self):
        """Release files even after a failed calculation, and restore the folder."""
        if self.profile_file is not None:
            self.profile_file.close()
        if self.balance_file is not None:
            self.balance_file.close()
        if self.fields is not None:
            self.fields.close()
            self.fields = None
        self.log.close()
        os.chdir(self.previous_folder)


def plot_profiles(folder, names, phases):
    """Plot saved profiles without rerunning transport or chemistry."""
    import pandas as pd
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    data = pd.read_csv(folder / 'profiles.csv')
    components = names.copy()
    if 'Zz' in components:
        components.remove('Zz')
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    for time, group in data.groupby('time_s'):
        label = f'{time / 86400:.2g} d'
        marker = None
        if len(group) == 1:
            marker = 'o'
        axes[0, 0].plot(group.x_m * 1000, group[components[0] + '_c_mol_m3_water'], label=label, marker=marker)
        axes[0, 1].plot(group.x_m * 1000, group.phi, label=label, marker=marker)
        if phases:
            axes[1, 0].plot(group.x_m * 1000, group.pH, label=label, marker=marker)
            phase = phases[0]
            if 'Portlandite' in phases:
                phase = 'Portlandite'
            axes[1, 1].plot(group.x_m * 1000, group[phase + '_mol_m3_bulk'], label=label, marker=marker)
        else:
            axes[1, 0].plot(group.x_m * 1000, group[components[0] + '_total_mol_m3_bulk'], label=label)
            axes[1, 1].plot(group.x_m * 1000, group.De_m2_s, label=label)
    labels = [components[0] + ' concentration [mol/m3 water]', 'Porosity [-]',
              'Total [mol/m3 bulk]', 'De [m2/s]']
    if phases:
        labels[2] = 'pH'
        labels[3] = phase + ' [mol/m3 bulk]'
    for ax, label in zip(axes.flat, labels):
        ax.set_xlabel('x [mm]')
        ax.set_ylabel(label)
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(folder / 'profiles.png', dpi=160)
    plt.close(fig)
