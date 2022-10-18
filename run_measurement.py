# -*- coding: utf-8 -*-
"""
Created on Wed Jan  8 12:08:16 2020

@author: LocalAdmin
"""


import os
import time
import signal
import numpy as np
import matplotlib.pyplot as plt
from threading import Event
import pandas as pd
import ADwin

SETUP = 'probe-station'  # 'cactus'or 'probe-station'
# Paths
if SETUP == 'probe-station':
    ROOT_DIR = r'C:\Users\LocalAdmin\Documents\Measurement Station\MatteoMetingen'
elif SETUP == 'cactus':
    ROOT_DIR = r'C:\Users\LocalAdmin\Documents\UserData\Matteo'
else:
    raise ValueError(f"Unknown SETUP '{SETUP}'.")

CHIP = 'SQC1'
DATA_DIR = os.path.join(ROOT_DIR, CHIP)
os.makedirs(DATA_DIR, exist_ok=True)
SCRIPTS_DIR = os.path.join(ROOT_DIR, 'scripts')
NUM = 178

# ignore temperature controller and write fake temperature to file, None otherwise
TEMP_FORCE_SET = None
# set a temperature and wait for it, None to use current temperature
TEMP = [None]
# TEMP = [9, 10, 11, 12, 14, 16, 18, 21, 24, 27, 30, 35, 40, 45, 50, 55, 60, 70, 80, 90, 100, 115, 130, 145, 160, 180, 200, 225, 250, 275, 300]
# timeout for constant measurement
DEFAULT_TIMEOUT = 10

X1 = 8
X1GAIN = 30
X2 = 0
X2GAIN = 90
YGAIN = 1e-7
AVERAGING = 2000
DELAY = 1200
# AVERAGING = 1900
# DELAY = 800

CLIP_THRESH = 3.5 * YGAIN

MODE = 'set_vg-take_iv'
# MODE = 'test_gate'

# assign units
UNITS = {k: 'A' if 0 < v < 1e-2 else 'V' for k, v in zip(['x1', 'x2', 'y'], [X1GAIN, X2GAIN, YGAIN])}
UNITS['t'] = 's'

# ADwin parameters
DEVICENUMBER = 1
RAISE_EXCEPTIONS = 1
PROCESS = 'general.T91'
ADW_CLOCK_TIME = 25e-9
ADW_MAX_INPUT = 10  # from -10V to 10V
ADW_STEPS = 32768  # steps for half sweep: 0 = -10V, 2 * ADW_STEPS = 10V
ADW_MAX_ARRAY_SIZE = 100000

# get telegram bot assistant
LAB_ASSISTANT_ENABLED = False
if LAB_ASSISTANT_ENABLED:
    try:
        lab_assistant
    except NameError:
        from telegram_assistant import TelegramAssistant
        # create a bot using BotFather, which will give youthe <token>
        # insert your <chat_id> or the one of a group
        lab_assistant = TelegramAssistant(token='<token>', chat_id='<chat_id>')


def broadcast(msg=str, end='\n'):
    print(msg, end=end)
    if LAB_ASSISTANT_ENABLED:
        try:
            lab_assistant.send(msg)
        except Exception:
            pass


def wait(seconds):
    e = Event()

    def sig_handler(sig, frame):
        e.set()
    signal.signal(signal.SIGINT, sig_handler)
    dt = min(seconds / 10, 1)
    t0 = time.monotonic()
    while time.monotonic() - t0 < seconds:
        e.wait(dt)
        if e.is_set():
            raise KeyboardInterrupt
    signal.signal(signal.SIGINT, signal.SIG_DFL)


# define temperature related functions
def find_heater_range(temp, thresh_low, thresh_medium, thresh_high=301):
    if temp is not None:
        hr = 'low'
        if temp > thresh_low:
            hr = 'medium'
        if temp > thresh_medium:
            hr = 'high'
        if temp > thresh_high:
            hr = 'off'
        return hr


def get_temperatures():
    global temp_controller
    if TEMP_FORCE_SET is None:
        if SETUP == 'cactus':
            temp_all = [temp_controller.temperature_A] * 4
        elif SETUP == 'probe-station':
            temp_all = temp_controller.get_all_kelvin_reading()
    else:
        temp_all = [TEMP_FORCE_SET] * 4
    return temp_all


def set_temperature(temp):
    global temp_controller
    if temp is None:
        return

    tol = 1e-2
    delta_temp = abs(temp - get_temperatures()[0])
    if delta_temp < tol:
        if not ask_yn(f'Temperature {temp} already reached. Do you need to stabilize?'):
            return
    else:
        # reach temperature
        if SETUP == 'cactus':
            temp_controller.setpoint_1 = temp
            while temp_controller.setpoint_1 != temp:
                wait(0.1)
            broadcast(f'[{time.ctime()}] Temperature set to {temp}\n')
            hr = find_heater_range(temp, 100, 200)
            temp_controller.heater_range = hr
            temp_controller.wait_for_temperature()
        elif SETUP == 'probe-station':
            # Using P: 100, I: 25, D: 0
            temp_controller.set_control_setpoint(1, temp)
            broadcast(f'[{time.ctime()}] Temperature set to {temp}')
            hr = find_heater_range(temp, 8, 40)
            temp_controller.set_heater_range(1, getattr(Model336HeaterRange, hr.upper()))
            while abs(temp-temp_controller.get_kelvin_reading(1))/temp > tol:
                wait(0.2)
            temp_controller.set_heater_range(1, getattr(Model336HeaterRange, hr.upper()))
            temp_controller.set_control_setpoint(1, temp)

    # wait for it to stabilize
    stab_time = 60 * 10 * (1 + int(temp > 100))
    broadcast('[{}] Temperature reached, now stablizing for {}s...'.format(
        time.ctime(),
        time.strftime('%Mm %Ss', time.gmtime(stab_time))
    ), end='')
    wait(stab_time)
    broadcast('Done({})'.format(adw.Process_Status(1)))


# Functions to run measurement
def get_phase(x1=None, x2=None, delay=1300, averaging=500, ramp=True, timeout=0):
    """Define a measurement phase.
    """
    phase = {}
    phase['delay'] = delay
    phase['averaging'] = averaging
    event_time = ADW_CLOCK_TIME * delay * averaging
    phase['event_time'] = event_time
    phase['ramp'] = ramp
    phase['max_events'] = int(timeout / event_time)
    if x1 is None:
        phase['x1'] = -1
    else:
        phase['x1'] = ADW_STEPS + int(ADW_STEPS * (x1 / X1GAIN) / ADW_MAX_INPUT)
    if x2 is None:
        phase['x2'] = -1
    else:
        phase['x2'] = ADW_STEPS + int(ADW_STEPS * (x2 / X2GAIN) / ADW_MAX_INPUT)
    return phase


def get_measurement_length(phases):
    """Estimate the length (time and steps) the measurement defined by 'phases'.
    Note: This needs to be run before the execution of the measurement!
    """
    global adw

    if not isinstance(phases, list):
        phases = [phases]

    total_time = 0
    total_length = 0
    x1 = adw.Get_Par(1)
    x2 = adw.Get_Par(2)
    for p in phases:
        if p['x1'] + p['x2'] == -2:
            events = p['max_events']
            steps = 0
        else:
            steps1 = abs(p['x1'] - x1)
            steps2 = abs(p['x2'] - x2)
            if p['x1'] == -1:
                index = 1
                x2 = p['x2']
            elif p['x2'] == -1:
                index = 0
                x1 = p['x1']
            else:
                index = np.argmax([steps1, steps2])
                x1 = p['x1']
                x2 = p['x2']
            steps, gain = [steps1, steps2][index], [X1GAIN, X2GAIN][index]
        if p['ramp']:
            events = steps
        else:
            events = 1
        if events > p['max_events'] and p['max_events'] > 0:
            broadcast(f"WARNING: {events} events are necessary to finish the sweep" +
                      f", but the timeout set only allows for {p['max_events']}.")
        total_time += (events + 1) * p['event_time']
        total_length += gain * steps * ADW_MAX_INPUT / ADW_STEPS
    return total_time, total_length


def plot(ax, mode, df):
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    ykeys, xkey = mode.split('/')
    ykeys = ykeys.split('-')
    u1 = None
    u0 = UNITS[ykeys[0]]
    hs = []
    for i, ykey in enumerate(ykeys):
        if UNITS[ykey] == u0:
            h_ = ax.scatter(df[xkey], df[ykey], s=5, edgecolors=None, c=cols[i], label=f'{ykey} [{UNITS[ykey]}]')
        else:
            u1 = UNITS[ykey]
            ax1 = ax.twinx()
            h_ = ax1.scatter(df[xkey], df[ykey], s=5, edgecolors=None, c=cols[i], label=f'{ykey} [{UNITS[ykey]}]')
        hs.append(h_)
    ax.set_xlabel(f'{xkey} [{UNITS[xkey]}]')
    ax.tick_params(axis='y')
    ax.legend(handles=hs)
    ax.set_ylabel(f'[{u0}]')
    if u1 is not None:
        ax1.tick_params(axis='y')
        ax1.set_ylabel(f'[{u1}]')
        ax1.set_ylim([1.2 * y for y in ax1.get_ylim()])


def update_plots(modes, df):
    if df.shape[0] == 0:
        return
    for j, mode in enumerate(modes):
        fig = plt.figure(j)
        ax = plt.gca()
        ax.clear()
        plot(ax, mode, df)
        plt.draw()
        fig.canvas.flush_events()


def get_data(df, event_time=None):
    global adw, ADW_CLOCK_TIME, ADW_MAX_ARRAY_SIZE

    length = adw.Get_Par(7)
    length_last = df.shape[0]
    if length_last == 0:
        t_last = 0
    else:
        t_last = df['t'].iloc[-1]
    temp_all = get_temperatures()
    if event_time is None:
        event_time = ADW_CLOCK_TIME * adw.Get_Par(3) * adw.Get_Processdelay(1)

    if length - length > ADW_MAX_ARRAY_SIZE:
        raise RuntimeError('The Data retrieved in a single update is too much. Increase the averaging or the delay.')

    while length_last < length:
        index = length_last % ADW_MAX_ARRAY_SIZE
        count = min(ADW_MAX_ARRAY_SIZE - index, length - length_last)

        data1 = adw.GetData_Long(1, index, count)
        data2 = adw.GetData_Long(2, index, count)
        data3 = adw.GetData_Float(3, index, count)

        df0 = pd.DataFrame({
            'x1': X1GAIN * (np.array(data1) - ADW_STEPS) * ADW_MAX_INPUT / ADW_STEPS,
            'y': YGAIN * (np.array(data3) - ADW_STEPS) * ADW_MAX_INPUT / ADW_STEPS,
            'x2': X2GAIN * (np.array(data2) - ADW_STEPS) * ADW_MAX_INPUT / ADW_STEPS,
            'temp': temp_all[0],
            't': t_last + np.arange(1, count + 1) * event_time,
            'temp1': temp_all[2],
            'temp2': temp_all[3],
        })
        df = df.merge(df0, how='outer')
        length_last += count
        t_last = df['t'].iloc[-1]

    return df


def run(phases, plots=[], plot_update_time=1, clip_error=False, clip_thresh=None, handle_interrupt=True):
    global adw

    e = Event()

    def sig_handler(sig, frame):
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        if ask_yn('Do you really want to shut down the measurement?'):
            e.set()
            adw.Stop_Process(1)
            broadcast('[{}] ADwin process stopped({})'.format(time.ctime(), adw.Process_Status(1)))
            reset_inputs()
        else:
            signal.signal(signal.SIGINT, sig_handler)

    if not isinstance(phases, list):
        phases = [phases]
    if plots is None:
        plots = []

    df = pd.DataFrame({
        'x1': [],
        'y': [],
        'x2': [],
        'temp': [],
        't': [],
        'temp1': [],
        'temp2': [],
    })

    adw.Set_Par(7, 0)
    t0 = time.monotonic()
    steps1_max = None
    steps2_max = None
    if handle_interrupt:
        signal.signal(signal.SIGINT, sig_handler)
    for i, p in enumerate(phases):
        # set parameters
        adw.Set_Processdelay(1, p['delay'])
        adw.Set_Par(3, p['averaging'])
        adw.Set_Par(8, p['max_events'])
        if steps1_max is not None and abs(p['x1'] - ADW_STEPS) > steps1_max:
            x1 = ADW_STEPS + int(np.sign(p['x1'] - ADW_STEPS) * steps1_max)
        else:
            x1 = p['x1']
        if steps2_max is not None and abs(p['x2'] - ADW_STEPS) > steps2_max:
            x2 = ADW_STEPS + int(np.sign(p['x2'] - ADW_STEPS) * steps2_max)
        else:
            x2 = p['x2']
        adw.Set_Par(4, x1)
        adw.Set_Par(5, x2)
        if not p['ramp']:
            if p['x1'] > -1:
                adw.Set_Par(1, x1)
            if p['x2'] > -1:
                adw.Set_Par(2, x2)

        # start process
        adw.Start_Process(1)
        while adw.Process_Status(1) > 0:
            df = get_data(df, event_time=p['event_time'])

            if clip_thresh is not None and df['y'].max() > clip_thresh:
                broadcast('Output too high, risk of clipping.')
                adw.Stop_Process(1)
                if clip_error:
                    update_plots(plots, df)
                    raise RuntimeError('Output too high, risk of clipping.')
                else:
                    if p['x1'] > -1:
                        steps1_max = adw.Get_Par(1) - ADW_STEPS
                    if p['x2'] > -1:
                        steps2_max = adw.Get_Par(2) - ADW_STEPS
                    clip_thresh = None
                    broadcast('Continuing measurement...', end='')
                    break

            t = time.monotonic()
            if (t - t0) // plot_update_time:
                update_plots(plots, df)
                t = t0
        else:
            # get leftover points
            df = get_data(df, event_time=p['event_time'])
            update_plots(plots, df)

        if e.is_set():
            break
    signal.signal(signal.SIGINT, signal.SIG_DFL)
    return df, e.is_set()


def set_input(x1=None, x2=None, ramp=True, stab=True, plots=[], plot_update_time=1, clip_error=False, clip_thresh=None, **kwargs):
    todo = [f"input {i+1} to {x}" for i, x in enumerate([x1, x2]) if x is not None]
    if len(todo) > 0:
        action = "Ramping " if ramp else "Setting "
        if stab:
            action += "and stabilizing "
        info = r"[{}] " + action + " and ".join(todo) + r" (will take {})..."
    else:
        raise ValueError("Specify at least one input.")

    phases = [get_phase(x1=x1, x2=x2, ramp=ramp, **kwargs)]
    meas_time, meas_length = get_measurement_length(phases)
    if meas_length == 0:
        return None, False
    if stab:
        stab_time = max(1*meas_length, 2*meas_length - meas_time)
        meas_time += stab_time
        phases.append(get_phase(timeout=stab_time))

    broadcast(info.format(time.ctime(), human_time(meas_time)), end='')
    df, s = run(phases, plots=plots, plot_update_time=plot_update_time, clip_error=clip_error,
                clip_thresh=clip_thresh, handle_interrupt=(stab or ramp))
    if not s:
        broadcast('Done({})'.format(adw.Process_Status(1)))
    return df, s


def take_iv(x1=None, x2=None, plots=[], plot_update_time=1, clip_error=False, clip_thresh=None, **kwargs):
    args = [(i+1, x) for i, x in enumerate([x1, x2]) if x is not None]
    if len(args) > 0:
        info = r"[{}] Taking iv on " + \
               " and ".join([f"input {i} to {x}" for i, x in args]) + \
               r" (will take {})..."
    else:
        raise ValueError("Specify at least one input.")

    reset_arg = args[0][0] if len(args) == 1 else 'all'
    reset_inputs(reset_arg, ask=False, plots=plots)

    phases = [
        get_phase(x1=x1, x2=x2, **kwargs),
        get_phase(x1=-x1 if x1 is not None else None, x2=-x2 if x2 is not None else None, **kwargs),
        get_phase(x1=0 if x1 is not None else None, x2=0 if x2 is not None else None, **kwargs),
    ]

    broadcast(info.format(time.ctime(), human_time(get_measurement_length(phases)[0])), end='')
    df, s = run(phases, plots=plots, plot_update_time=plot_update_time, clip_error=clip_error, clip_thresh=clip_thresh)
    if not s:
        broadcast('Done({})'.format(adw.Process_Status(1)))
    return df, s


def reset_inputs(num='all', ask=True, stab=False, **kwargs):
    """Reset inputs to zero.
        - num: which inputs? can be 'all'(default), 'x1' (or 1), 'x2' (or 2).
        - ask: wheter to ask for confirmation.
    """
    input_dict = {
        'all': (0, 0),
        'x1': (0, None),
        '1': (0, None),
        'x2': (None, 0),
        '2': (None, 0),
    }
    x1, x2 = [x if adw.Get_Par(i + 1) != ADW_STEPS else None for i, x in enumerate(input_dict[str(num)])]
    todo = [str(i + 1) for i, x in enumerate([x1, x2]) if x is not None]
    if len(todo) == 0:
        return
    if ask:
        info = 'input' + [' ', 's '][len(todo) - 1] + ' and '.join(todo)
        if not ask_yn(f"Do you want to reset {info} to zero?"):
            return

    if set_input(x1=x1, x2=x2, ramp=False, stab=stab, **kwargs)[1]:
        raise KeyboardInterrupt


# utilities
def save_data(path_name, df, modes=['y/x1', 'x1-x2-y/t']):
    broadcast('Saving...')
    if df.empty:
        raise ValueError('Data is empty! Try to restart kernel.')
    df.to_csv(path_name + '.csv', index=False)

    name = os.path.basename(path_name)
    io_keys = ['x1', 'x2', 'y']
    vmax = {k: np.max(df[k]) for k in (io_keys + ['t'])}
    temp = np.mean(df['temp'])
    main_input = 'x2' if vmax['x1'] == 0 else 'x1'

    info = "Temperature: {0:.3g} K\n".format(temp) + \
        "Max input 1: {0:.3g} {1}\n".format(vmax['x1'], UNITS['x1']) + \
        "Max input 2: {0:.3g} {1}\n".format(vmax['x2'], UNITS['x2']) + \
        "Max output: {0:.3g} {1}\n".format(vmax['y'], UNITS['y']) + \
        "Measurement took {0:.3g} s ({1})\n".format(vmax['t'], time.strftime('%Hh %Mm %Ss', time.gmtime(round(vmax['t'])))) + \
        "Averaging is {}\n".format(AVERAGING) + \
        "Process Delay is  {}".format(DELAY)
    with open(path_name + '.txt', 'w+') as output_file:
        print(f'{info}\nSaved in {name}.csv with column order: input 1, output, input 2 (gate), sample temperature' +
              ', time, arm temperature, cold head temperature', file=output_file)
    broadcast(' ---------------- ')
    broadcast(info)
    broadcast(' ---------------- ')

    modes = [f'y/{main_input}', 'x1-x2-y/t']
    n_sp = len(modes)
    fig, axs = plt.subplots(n_sp, 1, figsize=(7, n_sp*4))
    fig.suptitle(f'{name} ({temp:.2f}K)')
    if n_sp == 1:
        axs = [axs]
    for ax, mode in zip(axs, modes):
        plot(ax, mode, df)
    fig.tight_layout()
    fig.savefig(path_name + '.png', dpi=100)
    plt.close()
    if LAB_ASSISTANT_ENABLED:
        lab_assistant.send_image(path_name + '.png')


def check_name(path_name):
    global NUM, OVERWRITE_ALL
    while True:
        path_name = os.path.join(DATA_DIR, f'{CHIP}_{NUM}')
        broadcast(f'Selected file name is {path_name}')
        if OVERWRITE_ALL:
            break
        elif os.path.isfile(path_name + '.xlsx') or os.path.isfile(path_name + '.csv'):
            ans = input('File already exists. next available/overwrite/overwrite all/specify/cancel?[N/o/all/s/c] ')
        else:
            break
        if ans.lower() == 'c':
            reset_inputs()
            raise SystemExit(0)
        elif ans.lower() == 'n' or ans.lower() == '':
            while os.path.isfile(path_name + '.xlsx') or os.path.isfile(path_name + '.csv'):
                NUM += 1
                path_name = os.path.join(DATA_DIR, f'{CHIP}_{NUM}')
            broadcast(f"Using file name {path_name}.")
            break
        elif ans.lower() == 's':
            NUM = int(input('Specify number: '))
            path_name = os.path.join(DATA_DIR, f'{CHIP}_{NUM}')
        elif ans.lower() == 'o':
            break
        elif ans.lower() == 'all':
            OVERWRITE_ALL = True
            break
    return path_name


def human_time(t):
    return time.strftime('%Hh %Mm %Ss', time.gmtime(round(t)))


def ask_yn(question):
    while True:
        ans = input(f'{question} [Y/n] ')
        if ans.lower() == 'n':
            return False
        elif ans.lower() == 'y' or ans.lower() == '':
            return True
        else:
            broadcast(f"Invalid answer '{ans}'.")


if __name__ == '__main__':
    # get temperature controller
    if TEMP_FORCE_SET is None:
        try:
            temp_controller
        except NameError:
            if SETUP == 'cactus':
                from lakeshore331 import LakeShore331
                temp_controller = LakeShore331('GPIB::12')
            elif SETUP == 'probe-station':
                from lakeshore import Model336, Model336HeaterRange
                temp_controller = Model336()
            else:
                raise NotImplementedError(f"Cannot handle setup '{SETUP}'.")

    # get adw instance
    try:
        adw
    except NameError:
        adw = ADwin.ADwin(DEVICENUMBER, RAISE_EXCEPTIONS)
    adw.Boot(adw.ADwindir + '\\ADwin9.btl')
    adw.Load_Process(PROCESS)
    reset_inputs(ask=False)

    if not isinstance(TEMP, list):
        TEMP = [TEMP]
    if not isinstance(X1, list):
        X1 = [X1 for temp in TEMP]
    if not isinstance(X2, list):
        X2 = [[X2] for temp in TEMP]
    elif not isinstance(X2[0], list):
        X2 = [X2 for temp in TEMP]
    OVERWRITE_ALL = False

    if MODE == 'set_vg-take_iv':
        for i, (temp, x1, x2_list) in enumerate(zip(TEMP, X1, X2)):
            for j, x2 in enumerate(x2_list):
                path_name = check_name(os.path.join(DATA_DIR, f'{CHIP}_{NUM}'))

                if j == 0:
                    set_temperature(temp)
                if x2 is not None:
                    if set_input(x2=x2, plots=['y/x2', 'x2/t', 'y/t'], clip_thresh=CLIP_THRESH)[1]:
                        raise KeyboardInterrupt

                df, s = take_iv(x1=x1, delay=DELAY, averaging=AVERAGING, plots=['y/x1', 'x1/t', 'y/t'], clip_thresh=CLIP_THRESH)
                if s:
                    if ask_yn('Do you want to save the incomplete measurement?'):
                        save_data(path_name, df)
                    raise KeyboardInterrupt
                save_data(path_name, df)
                NUM += 1

    # test gate
    elif MODE == 'test_gate':
        for i, (temp, x2) in enumerate(zip(TEMP, X2)):
            path_name = check_name(os.path.join(DATA_DIR, f'{CHIP}_{NUM}'))
            df, s = set_input(x2=x2, plots=['y/x2', 'x2/t', 'y/t'], delay=DELAY, averaging=AVERAGING, clip_thresh=CLIP_THRESH)
            if s:
                if ask_yn('Do you want to save the incomplete measurement?'):
                    save_data(path_name, df)
                raise KeyboardInterrupt
            save_data(path_name, df)
            NUM += 1

    reset_inputs()
