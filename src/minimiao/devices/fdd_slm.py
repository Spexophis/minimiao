# -*- coding: utf-8 -*-
# Copyright (c) 2020 Peter Kner & Ruizhe Lin
# Licensed under the MIT License.


import ctypes as ct
import subprocess

from minimiao import logger

class Dev(ct.Structure):
    pass


Dev._fields_ = [("id", ct.c_char_p), ("next", ct.POINTER(Dev))]

NULL = ct.POINTER(ct.c_int)()
RS485_DEV_TIMEOUT = ct.c_uint16(1000)
RS485_BAUDRATE = ct.c_uint32(256000)
RS232_BAUDRATE = ct.c_uint32(115200)


class QXGA:
    def __init__(self, logg=None, config=None):
        self.logg = logg or logger.setup_logging()
        self.config = config or self.load_configs()
        self.r11 = self._initiate()
        self.get_temperature()
        self.ord_dict = self.get_order_list()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass

    @staticmethod
    def load_configs():
        import json
        config_file = input("Enter configuration file directory: ")
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg

    def _initiate(self):
        r11_lib = self.config["Spatial Light Modulator"]["Forth Dimension Displays"]["ControlLibrary"]
        r11 = ct.windll.LoadLibrary(r11_lib)
        ver = ct.create_string_buffer(16)
        max_len = ct.c_uint8(16)
        res = r11.R11_LibGetVersion(ver, max_len)
        if res == 0:
            self.logg.info(r"QXGA R11 Software version: %s" % ver.value)
            guid = ct.c_char_p(b"54ED7AC9-CC23-4165-BE32-79016BAFB950")
            dev_count = ct.c_uint16(0)
            devlist = ct.POINTER(Dev)()
            res = r11.FDD_DevEnumerateWinUSB(guid, ct.pointer(devlist), ct.byref(dev_count))
            if res == 0:
                port = devlist.contents.id.decode()
                self.logg.info('Dev port: %s' % port)
                po = ct.c_char_p(b'\\\\?\\usb#vid_19ec&pid_0503#0175000787#{54ed7ac9-cc23-4165-be32-79016bafb950}')
                res = r11.FDD_DevOpenWinUSB(po, RS485_DEV_TIMEOUT)
                if res == 0:
                    self.logg.info('Open Dev port successfully')
                    return r11
                else:
                    raise RuntimeError("Fail to open the port")
            else:
                raise RuntimeError("Fail to find the port")
        else:
            raise RuntimeError("Fail to open the port")

    def close(self):
        re = self.r11.FDD_DevClose()
        if re == 0:
            self.logg.info('Port closed successfully')
        else:
            raise RuntimeError('Fail to close QXGA')

    def send_rep(self, fns):
        sn = self.config["Spatial Light Modulator"]["Forth Dimension Displays"]["Serial"]
        x = subprocess.run([r"C:/Program Files/MetroCon-3.3/RepTools/RepSender.exe", '-z', fns, '-d', sn],
                           stdout=subprocess.PIPE)
        if x.returncode == 0:
            self.logg.info('load the repz11 file successfully')
        else:
            self.logg.error('fail to load the repz11 file')

    def get_temperature(self):
        disp_temp = ct.c_uint16(0)
        res = self.r11.R11_RpcSysGetDisplayTemp(ct.byref(disp_temp))
        if res == 0:
            self.logg.info('Display temperature: %s' % disp_temp.value)
        else:
            raise RuntimeError('Fail to get the display temperature')

    def get_order_num(self):
        ord_count = ct.c_uint16(0)
        res = self.r11.R11_RpcRoGetCount(ct.byref(ord_count))
        if res == 0:
            return ord_count.value
        else:
            raise RuntimeError('Fail to get the order number')

    def get_order_name(self, n):
        ord_count = ct.c_uint16(n)
        ord_name = ct.create_string_buffer(128)
        max_len = ct.c_uint8(128)
        res = self.r11.R11_RpcRoGetName(ord_count, ord_name, max_len)
        if res == 0:
            return ord_name.value
        else:
            raise RuntimeError('Fail to get the order name')

    def get_order_list(self):
        ord_dict = {}
        odn = self.get_order_num()
        for i in range(odn):
            ord_name = self.get_order_name(i)
            ord_dict[ord_name.decode('utf-8')] = i
        return ord_dict

    def get_selected_order(self):
        index = ct.c_uint16(0)
        res = self.r11.R11_RpcRoGetSelected(ct.byref(index))
        if res == 0:
            o = self.get_order_name(index.value)
            return index.value, o.decode('utf-8')
        else:
            raise RuntimeError('Fail to Get Selected Order')

    def select_order(self, n):
        ord_index = ct.c_uint16(n)
        res = self.r11.R11_RpcRoSetSelected(ord_index)
        if res == 0:
            index = ct.c_uint16(0)
            res = self.r11.R11_RpcRoGetSelected(ct.byref(index))
            if (res == 0) & (index.value == n):
                self.logg.info('Order is set to #%s' % n)
                o = self.get_order_name(index.value)
                t_total, t_end, t_on = self.get_sequence_parameters(o.decode('utf-8'))
                return t_total, t_end, t_on
            else:
                raise RuntimeError('Order set is wrong')
        else:
            raise RuntimeError('Fail to set the order')

    def activate(self):
        res = self.r11.R11_RpcRoActivate(ct.c_void_p())
        if res == 0:
            self.logg.info('Activate QXGA successfully')
        else:
            raise RuntimeError('Fail to activate QXGA')

    def deactivate(self):
        res = self.r11.R11_RpcRoDeactivate(ct.c_void_p())
        if res == 0:
            self.logg.info('Deactivate QXGA successfully')
        else:
            raise RuntimeError('Fail to deactivate')

    def get_sequence_parameters(self, slm_seq="10ms_lit_pair"):
        if "200us_lit_balanced" in slm_seq:
            self.logg.info("200us_lit_balanced")
            t_total = 576.533e-6
            t_on = 199.893e-6
            t_end = 520.853e-6
        elif "400us_lit_balanced" in slm_seq:
            self.logg.info("400us_lit_balanced")
            t_total = 776.64e-6
            t_on = 400e-6
            t_end = 720.96e-6
        elif "600us_lit_balanced" in slm_seq:
            self.logg.info("600us_lit_balanced")
            t_total = 976.747e-6
            t_on = 600.107e-6
            t_end = 921.067e-6
        elif "500us_lit_pair" in slm_seq:
            self.logg.info("500us_lit_pair")
            t_total = 2 * 810.667e-6
            t_on = 2 * 499.947e-6 + 310.72e-6
            t_end = 770.133e-6 + t_total / 2
        elif "1ms_lit_pair" in slm_seq:
            self.logg.info("500us_lit_pair")
            t_total = 2 * 1310.72e-6
            t_on = 2 * 1000e-6 + 310.72e-6
            t_end = 1270.187e-6 + t_total / 2
        elif "5ms_lit_pair" in slm_seq:
            self.logg.info("5ms_lit_pair")
            t_total = 2 * 5310.72e-6
            t_on = 2 * 5000.0e-6 + 310.72e-6
            t_end = 5270.187e-6 + t_total / 2
        elif "10ms_lit_pair" in slm_seq:
            self.logg.info("10ms_lit_pair")
            t_total = 2 * 10310.72e-6
            t_on = 2 * 10000.0e-6 + 310.72e-6
            t_end = 10270.187e-6 + t_total / 2
        elif "20ms_lit_pair" in slm_seq:
            self.logg.info("20ms_lit_pair")
            t_total = 2 * 20310.72e-6
            t_on = 2 * 20000.0e-6 + 310.72e-6
            t_end = 20270.187e-6 + t_total / 2
        elif "5ms_dark_pair" in slm_seq:
            self.logg.info("5ms_dark_pair")
            t_total = 5310.72e-6
            t_on = 5000.0e-6
            t_end = 5270.187e-6
        elif "10ms_dark_pair" in slm_seq:
            self.logg.info("10ms_dark_pair")
            t_total = 10310.72e-6
            t_on = 10000.0e-6
            t_end = 10270.187e-6
        elif "20ms_dark_pair" in slm_seq:
            self.logg.info("20ms_dark_pair")
            t_total = 20310.72e-6
            t_on = 20000.0e-6
            t_end = 20270.187e-6
        else:
            self.logg.error("SLM sequence error.")
            raise ValueError("SLM sequence is wrong.")
        return t_total, t_end, t_on

"""
Generate SLM run-time sequences for a phase/angle sweep.
"""
import re
import textwrap

# ---------------------------------------------------------------- settings --
SEQ_TYPE = "linear"    # "nonlinear" or "linear"
BASE = 113             # index of angle 0, phase 0
N_ANGLES = 6
N_PHASES = 10          # resolft needs this even; the pi partner is N_PHASES // 2 away
OFF_GPO = 6            # resolft: sync port for the pi-shifted pattern
ON_GPO = 2             # resolft: sync port for the readout pattern
LIT_GPO = 2            # lit_pair: the single sync port
OFF_DESIG = "A"        # pattern designator for the pi-shifted load
ON_DESIG = "A"         # pattern designator for the readout load
ANGLE_MAJOR = True     # True: all phases of one angle, then the next angle
OFF_FIRST = True       # resolft: pi-shifted pattern first in each frame
PORTS = ("GPO3", "GPO2", "GPO1")
WIDTH = None           # None keeps the body on one line; an int wraps to that width
NAME = None            # None auto-names from the settings; "" omits the name line
EXPOSURE = "1ms"       # used only in the auto-generated name
SPACING_PIX = 20       # used only in the auto-generated name
OUTPUT = r"C:\Users\ruizhe.lin\Documents\Data\slm_files\QXGA\sequence.txt"
# -----------------------------------------------------------------------------


def frame_index(base, angle, phase, n_phases):
    return base + angle * n_phases + (phase % n_phases)


def load(gpo, designator, index):
    """One pattern load. gpo=None emits a bare load with no GPO prefix."""
    body = f"t({designator}+,{index}) ({designator}-,{index})"
    return body if gpo is None else f"GPO={gpo} {body}"


def sweep_order(n_angles=N_ANGLES, n_phases=N_PHASES, angle_major=ANGLE_MAJOR):
    return ([(a, p) for a in range(n_angles) for p in range(n_phases)] if angle_major
            else [(a, p) for p in range(n_phases) for a in range(n_angles)])


def build_linear_pair(base=BASE, n_angles=N_ANGLES, n_phases=N_PHASES,
                      gpo=LIT_GPO, desig=ON_DESIG, angle_major=ANGLE_MAJOR):
    """Single GPO trigger followed by every pattern in sweep order."""
    idx = [frame_index(base, a, p, n_phases)
           for a, p in sweep_order(n_angles, n_phases, angle_major)]
    return [load(gpo if k == 0 else None, desig, i) for k, i in enumerate(idx)]


def build_nonlinear_frames(base=BASE, n_angles=N_ANGLES, n_phases=N_PHASES,
                           off_gpo=OFF_GPO, on_gpo=ON_GPO, off_desig=OFF_DESIG, on_desig=ON_DESIG,
                           angle_major=ANGLE_MAJOR, off_first=OFF_FIRST):
    if n_phases % 2:
        raise ValueError(f"n_phases={n_phases} is odd; no exact pi partner exists")
    half = n_phases // 2
    frames = []
    for angle, phase in sweep_order(n_angles, n_phases, angle_major):
        readout = frame_index(base, angle, phase, n_phases)
        shifted = frame_index(base, angle, phase + half, n_phases)
        if off_first:
            pair = [(off_gpo, off_desig, shifted), (on_gpo, on_desig, readout)]
        else:
            pair = [(on_gpo, on_desig, readout), (off_gpo, off_desig, shifted)]
        frames.append(" ".join(load(*p) for p in pair))
    return frames


LOAD_RE = re.compile(r"(?:GPO=(\d+) )?t\(([A-Z])\+,(\d+)\) \(([A-Z])-,(\d+)\)")


def _rewrite(sequence, pick):
    """Walk the loads, tracking the GPO in force, and let pick(gpo, desig) rename."""
    state = {"gpo": None}

    def sub(m):
        gpo, desig, i1, _, i2 = m.groups()
        if gpo is not None:
            state["gpo"] = gpo
        d = pick(state["gpo"], desig) or desig
        prefix = "" if gpo is None else f"GPO={gpo} "
        return f"{prefix}t({d}+,{i1}) ({d}-,{i2})"
    return LOAD_RE.sub(sub, sequence)


def set_designators(sequence, off_desig=None, on_desig=None,
                    off_gpo=OFF_GPO, on_gpo=ON_GPO):
    """Rewrite pattern designators in an existing sequence string.

    Loads under off_gpo get off_desig, loads under on_gpo get on_desig.
    Pass None for either to leave that one alone. Bare loads inherit the
    GPO most recently set, so lit_pair sequences are handled too.
    """
    table = {str(off_gpo): off_desig, str(on_gpo): on_desig}
    return _rewrite(sequence, lambda gpo, desig: table.get(gpo))


def swap_ab(sequence):
    """Swap every A designator with B and vice versa."""
    return _rewrite(sequence, lambda gpo, desig: {"A": "B", "B": "A"}.get(desig))


def sequence_name(seq_type=None, exposure=EXPOSURE, spacing_pix=SPACING_PIX,
                  n_phases=N_PHASES, n_angles=N_ANGLES):
    kind = "lit_pair" if (seq_type or SEQ_TYPE) == "lit_pair" else "resolft"
    return (f"{exposure}_{kind}_sim_{spacing_pix}_pix_{n_phases}_phs_{n_angles}_ang")


def build_sequence(frames, width=WIDTH, ports=PORTS, name=NAME):
    body = "<" + " ".join(frames) + " >"
    lines = ([body] if width is None else
             textwrap.wrap(body, width=width, break_long_words=False,
                           break_on_hyphens=False))
    head = [f'"{name}"'] if name else []
    return "\n".join(head + ["[s", " " + " ".join(ports)] + [" " + line for line in lines] + ["]"])


if __name__ == "__main__":

    if SEQ_TYPE == "linear":
        frames = build_linear_pair()
        n_loads = len(frames)
    elif SEQ_TYPE == "nonlinear":
        frames = build_nonlinear_frames()
        n_loads = 2 * len(frames)
    else:
        raise ValueError(f"unknown SEQ_TYPE {SEQ_TYPE!r}")

    sequence = build_sequence(frames, name=sequence_name() if NAME is None else NAME)
    lo, hi = BASE, BASE + N_ANGLES * N_PHASES - 1
    print(f"# {SEQ_TYPE}: {len(frames)} frames, {n_loads} pattern loads, indices {lo}-{hi}")
    if OUTPUT:
        with open(OUTPUT, "w") as fh:
            fh.write(sequence + "\n")
        print(f"# written to {OUTPUT}")
    else:
        print(sequence)
