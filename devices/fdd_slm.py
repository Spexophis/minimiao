import ctypes as ct
import subprocess


class Dev(ct.Structure):
    pass


Dev._fields_ = [("id", ct.c_char_p), ("next", ct.POINTER(Dev))]

NULL = ct.POINTER(ct.c_int)()
RS485_DEV_TIMEOUT = ct.c_uint16(1000)
RS485_BAUDRATE = ct.c_uint32(256000)
RS232_BAUDRATE = ct.c_uint32(115200)


class QXGA:
    def __init__(self, logg=None, config=None):
        self.logg = logg or self.setup_logging()
        self.config = config or self.load_configs()
        self.r11 = self._initiate()
        self.get_temperature()
        self.ord_dict = self.get_order_list()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @staticmethod
    def load_configs():
        config_file = input("Enter configuration file directory: ")
        from miao.utilities import configurations
        cfg = configurations.MicroscopeConfiguration(fd=config_file)
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
            ord_dict[i] = ord_name.decode('utf-8')
        return ord_dict

    def select_order(self, n):
        ord_index = ct.c_uint16(n)
        res = self.r11.R11_RpcRoSetSelected(ord_index)
        if res == 0:
            index = ct.c_uint16(0)
            res = self.r11.R11_RpcRoGetSelected(ct.byref(index))
            if (res == 0) & (index.value == n):
                self.logg.info('Order is set to #%s' % n)
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
