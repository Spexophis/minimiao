from minimiao import logger
import serial
import time
import threading

BAUDRATE = 19200
COMMAND_TERMINATOR = "\r"

class LeicaDMI:
    def __init__(self, port=None, mid=None, logg=None, config=None):
        self.logg = logg or logger.setup_logging()
        self.config = config or self.load_configs()
        self.port = port or self.config["Microscope Stand"]["LeicaDMI"]["Port"]
        self.mid = mid or self.config["Microscope Stand"]["LeicaDMI"]["MotorID"]
        self.cmds = self.config["Microscope Stand"]["LeicaDMI"]["CMD"]
        self.rs232 = self.open_serial()
        self.corr_max = 9387
        self.corr_min = 0
        self.current_corr = 0
        self.current_z = 0
        self.corr_init_thread = None
        self.corr_ready = threading.Event()
        self.initialize()

    @staticmethod
    def load_configs():
        import json
        config_file = input("Enter configuration file directory: ")
        with open(config_file, 'r') as f:
            cfg = json.load(f)
        return cfg

    def open_serial(self):
        ser = serial.Serial(port=self.port,
                            baudrate=BAUDRATE,
                            bytesize=serial.EIGHTBITS,
                            parity=serial.PARITY_NONE,
                            stopbits=serial.STOPBITS_ONE,
                            xonxoff=True,
                            rtscts=False,
                            dsrdtr=False,
                            timeout=1.0,
                            write_timeout=2.0)
        return ser

    def send_command(self, command):
        message = command + COMMAND_TERMINATOR
        self.rs232.write(message.encode("ascii"))

    def read_response(self, timeout=16.0):
        start_time = time.monotonic()
        response = b""

        while time.monotonic() - start_time < timeout:
            waiting = self.rs232.in_waiting

            if waiting:
                response += self.rs232.read(waiting)

                try:
                    response_str = response.decode("ascii", errors="replace")
                    self.logg.info(f"Received: {response_str}")
                    return response_str
                except Exception:
                    self.logg.error(f"Received: {response}")
                    return response

            time.sleep(0.1)

        self.logg.error(f"No response received within {timeout} seconds")
        return None

    def initialize(self):
        self.send_command(self.cmds["GET_MODULE_Z_DRIVE"])
        r = self.read_response()
        self.send_command(self.cmds["GET_VERSION_Z_DRIVE"])
        r = self.read_response()
        self.send_command(self.cmds["INIT_Z"])
        r = self.read_response()
        if r.split()[0] == self.cmds["INIT_Z"]:
            self.get_z()
        self.send_command(self.cmds["GET_MODULE_MOT_CORR"])
        r = self.read_response()
        self.send_command(self.cmds["GET_VERSION_MOT_CORR"])
        r = self.read_response()
        if r.split()[0] == self.cmds["GET_VERSION_MOT_CORR"]:
            self.corr_init_thread = threading.Thread(target=self._initialize_corr, daemon=True)
            self.corr_init_thread.start()

    def _initialize_corr(self):
        try:
            self.send_command(self.cmds["INIT_MOT_CORR"] + self.mid + " 0")
            r = self.read_response(64)
            self.send_command(self.cmds["GET_MAX_POS_MOT_CORR"] + self.mid)
            r = self.read_response()
            self.corr_max = int(r.split()[-1])
            self.send_command(self.cmds["GET_MIN_POS_MOT_CORR"] + self.mid)
            r = self.read_response()
            self.corr_min = int(r.split()[-1])
        except Exception as e:
            self.logg.exception(f"Error during background corr motor initialization: {e}")
        finally:
            self.corr_ready.set()
            self.logg.info("Background corr motor initialization finished.")

    def wait_for_corr_init(self, timeout=None):
        return self.corr_ready.wait(timeout)

    def mov_corr(self, pos: int):
        self.send_command(self.cmds["SET_POS_MOT_CORR"] + self.mid + f" {pos}")
        r = self.read_response(32)
        if r.split()[0] == self.cmds["SET_POS_MOT_CORR"]:
            self.get_corr()

    def get_corr(self):
        self.send_command(self.cmds["GET_POS_MOT_CORR"] + self.mid)
        r = self.read_response(8)
        if r.split()[0] == self.cmds["GET_POS_MOT_CORR"]:
            self.current_corr = int(r.split()[-1])

    def mov_z(self, pos: int):
        self.send_command(self.cmds["POS_ABS_Z"] + f" {pos}")
        r = self.read_response(8)
        if r.split()[0] == self.cmds["POS_ABS_Z"]:
            self.get_z()

    def get_z(self):
        self.send_command(self.cmds["GET_POS_Z"])
        r = self.read_response(8)
        if r.split()[0] == self.cmds["GET_POS_Z"]:
            self.current_z = int(r.split()[-1])
