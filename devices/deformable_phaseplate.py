import os
import time
from pathlib import Path
import pickle

from dpp_ctrl import api_dpp

cal_file = Path(r"C:\Users\ruizhe.lin\Desktop\phaseform\Calibration_D7-22-00039-A.json").expanduser()

zms = [(-1, 1), (1, 1), (0, 2), (-2, 2), (2, 2), (-1, 3), (1, 3), (-3, 3), (3, 3), (0, 4), (-2, 4), (2, 4), (-4, 4), (4, 4)]
phases = {}
for zm in zms:
    phases[zm] = 0.0

class DPP:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.dpp, self.opened_flag = self._initialize()

    def __del__(self):
        pass

    def close(self):
        self.dpp.close()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def _initialize(self):
        dpp = api_dpp.initialize()
        if dpp.connect_device(port_name="COM9"):
            if dpp.load_infl_matrix(str(cal_file), operation_mode='v'):
                return dpp, True
            else:
                self.logg.error("Cannot connect DPP!")
                return False, False
        else:
            self.logg.error("Cannot connect DPP!")
            return False, False

phases = {}
for zm in zms:
    phases[zm] = 0.0
amps = [-0.1, -0.05, 0, 0.05, 0.1]
a = []
data = []
for i, zm in enumerate(zms):
    image = []
    mts = []
    for amp in amps:
        phase_temp = phases.copy()
        phase_temp[zm] = amp
        dpp.apply_phases(phase_temp)
        time.sleep(0.1)
        daq.run_triggers()
        time.sleep(0.02)
        re = cam.get_image(True)
        print(re[1])
        image.append(re[0])
        daq.stop_triggers(_close=False)
    data.append(image)
    for img in image:
        mt = np.max(img)
        mts.append(mt)
    v = ipr.peak_find(amps, mts)
    print(zm, v)
    if isinstance(v, np.floating):
        val = round(v.item(), ndigits=4)
        a.append(val)
        phases[zm] += val
    else:
        a.append(v)


fn = f"_test"
fd = os.path.join(data_folder, time.strftime('%Y%m%d%H%M%S') + fn)
tf.imwrite(str(fd + r".tif"), np.asarray(data))


# Save to file
with open(str(os.path.join(data_folder, time.strftime('%Y%m%d%H%M%S') + fn) + ".pkl"), "wb") as f:
    pickle.dump(data, f)


# Read from file
with open("data.pkl", "rb") as f:
    loaded = pickle.load(f)


# from dpp_ctrl import gui_dpp
# from multiprocessing import Queue
# mpQueue = Queue() # define Queue
# ctrl_dpp_prc = gui_dpp.IndPrcLauncher(mpQueue)
# ctrl_dpp_prc.start() # start GUI process
#
# mpQueue.put_nowait("EXIT")