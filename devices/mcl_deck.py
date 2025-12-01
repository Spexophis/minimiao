import ctypes as ct
import os
import sys
import threading

sys.path.append(r'C:\Program Files\Mad City Labs\MicroDrive')
micro_dll_path = os.path.join('C:', os.sep, 'Program Files', 'Mad City Labs', 'MicroDrive', 'MicroDrive.dll')


class MCLMicroDrive:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.mcl_deck = ct.CDLL(micro_dll_path)
        self.handle = self.initialize_deck()
        self.encoder_resolution, self.step_size, self.velocity_max, self.max_velocity_axis_two, \
            self.max_velocity_axis_three, self.velocity_min = self.get_device_info()
        self.error_dictionary = {0: 'MCL_SUCCESS',
                                 -1: 'MCL_GENERAL_ERROR',
                                 -2: 'MCL_DEV_ERROR',
                                 -3: 'MCL_DEV_NOT_ATTACHED',
                                 -4: 'MCL_USAGE_ERROR',
                                 -5: 'MCL_DEV_NOT_READY',
                                 -6: 'MCL_ARGUMENT_ERROR',
                                 -7: 'MCL_INVALID_AXIS',
                                 -8: 'MCL_INVALID_HANDLE'}
        # Dictionary to know the axis limit returns.
        # Dictionary saves [axis, forward (1) or backward (-1), description]
        self.motor_limits = [[1, -1, 'Axis 1 reverse limit'],  # 126 <-> '1111110' <-> position 0
                             [1, 1, 'Axis 1 forward limit'],  # 125 <-> '1111101' <-> position 1
                             [2, -1, 'Axis 2 reverse limit'],  # 123 <-> '1111011' <-> position 2
                             [2, 1, 'Axis 2 forward limit'],  # 119 <-> '1110111' <-> position 3
                             [3, -1, 'Axis 3 reverse limit'],  # 111 <-> '1101111' <-> position 4
                             [3, 1, 'Axis 3 forward limit']]  # 095 <-> '1011111' <-> position 5
        self.total_range = 23  # mm
        self.position = 0
        self.move_thread = None

    def __del__(self):
        pass

    def close(self):
        """
        Closes the connection by releasing the handle.
        """
        self.stop_moving()
        self.mcl_deck.MCL_ReleaseHandle(self.handle)
        self.logg.info('Deck Handle released.')

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def initialize_deck(self):
        self.mcl_deck.MCL_ReleaseAllHandles()
        handle = self.mcl_deck.MCL_InitHandle()
        if handle > 0:
            n = self.mcl_deck.MCL_DeviceAttached(ct.c_uint(500), handle)
            sn = self.mcl_deck.MCL_GetSerialNumber(handle)
            pid = ct.pointer(ct.c_ushort())
            self.mcl_deck.MCL_GetProductID(pid, handle)
            self.logg.info(f'Connected to {n} MadDeck With handle: {handle}\nSN: {sn}\nProductID: {pid.contents.value}')
        else:
            self.logg.error('MadDeck Connection failed.')
        return handle

    def get_device_info(self):
        encoder_resolution = ct.pointer(ct.c_double())
        step_size = ct.pointer(ct.c_double())
        max_velocity = ct.pointer(ct.c_double())
        max_velocity_axis_two = ct.pointer(ct.c_double())
        max_velocity_axis_three = ct.pointer(ct.c_double())
        min_velocity = ct.pointer(ct.c_double())
        self.mcl_deck.MCL_MDInformation(encoder_resolution, step_size, max_velocity, max_velocity_axis_two,
                                        max_velocity_axis_three, min_velocity, self.handle)
        info = encoder_resolution.contents.value, step_size.contents.value, max_velocity.contents.value, \
            max_velocity_axis_two.contents.value, max_velocity_axis_three.contents.value, min_velocity.contents.value
        return info

    def _get_status(self):
        """
        Internal function to get the error number
        """
        try:
            status_temp = ct.pointer(ct.c_ushort())
            self.mcl_deck.MCL_MDStatus(status_temp, self.handle)
            return status_temp.contents.value
        except Exception as e:
            self.logg.error(f"Error in getting status: {e}")
            return None  # or handle this case as needed

    def get_status(self):
        """
        Returns a list of motors that are out of bounds (reverse of forward limit)
        [axis, forward (1) / reverse (-1), description]
        [1,-1,'Axis 1 reverse limit']
        [1, 1,'Axis 1 forward limit']
        [2,-1,'Axis 2 reverse limit']
        [2, 1,'Axis 2 forward limit']
        [3,-1,'Axis 3 reverse limit']
        [3, 1,'Axis 3 forward limit']
        """
        status = self._get_status()
        errors_limit = []
        for i, b in enumerate(bin(status)[:1:-1]):
            if b == '0':
                errors_limit.append(self.motor_limits[i])
        if not errors_limit:
            errors_limit.append([0, 0, 'All ok'])
        return errors_limit

    def wait(self):
        """
        This function takes approximately 10ms if the motors are not moving.
        """
        error_number = self.mcl_deck.MCL_MicroDriveWait(self.handle)
        if error_number != 0:
            self.logg.error('Error while waiting: ' + self.error_dictionary[error_number])

    def _move_relative(self, axis, distance, velocity=1.5):
        error_code = self.mcl_deck.MCL_MDMove(ct.c_uint(axis), ct.c_double(velocity), ct.c_double(distance),
                                              self.handle)
        self.wait()
        return error_code

    def move_relative(self, axis, distance, velocity=1.5):
        """
        Moves a single axis by distance with velocity.
        """
        if velocity > self.velocity_max:
            self.logg.error('Given velocity is too high. Velocity is set to maximum value.')
            velocity = self.velocity_max
        elif velocity < self.velocity_min:
            self.logg.error('Given velocity is too low. Velocity is set to minimum value.')
            velocity = self.velocity_min
        error_number = self._move_relative(axis, distance, velocity)
        if error_number == 0:
            self.position = self.get_position_steps_taken(3)
        else:
            raise RuntimeError('Error while moving axis ' + str(axis) + ': ' + self.error_dictionary[error_number])

    def get_position_steps_taken(self, axis):
        micro_steps = ct.pointer(ct.c_int())
        error_number = self.mcl_deck.MCL_MDCurrentPositionM(ct.c_int(axis), micro_steps, self.handle)
        if error_number == 0:
            return micro_steps.contents.value * self.step_size
        else:
            self.logg.error(
                'Error reading the position of axis' + str(axis) + ': ' + self.error_dictionary[error_number])

    def move_deck(self, direction, velocity):
        self.move_thread = MoveThread(self, direction, velocity)
        self.move_thread.start()

    def stop_deck(self):
        if self.move_thread is not None:
            self.move_thread.stop()
            self.move_thread.join()
            self.move_thread = None

    def is_moving(self):
        """
        Checks if motors are moving.
        This function takes approximately 20ms.
        """
        _is_moving = ct.pointer(ct.c_int())
        self.mcl_deck.MCL_MicroDriveMoveStatus(_is_moving, self.handle)
        return _is_moving.contents.value

    def stop_moving(self):
        """
        Stops motors from moving.
        """
        status = ct.pointer(ct.c_ushort())
        error_number = self.mcl_deck.MCL_MDStop(status, self.handle)
        if error_number != 0:
            self.logg.error("Error while stopping device: " + self.error_dictionary[error_number])


class MoveThread(threading.Thread):
    running = False
    lock = threading.Lock()

    def __init__(self, mdk, direction, velocity):
        threading.Thread.__init__(self)
        self.mdk = mdk
        self.d = direction
        self.v = velocity

    def run(self):
        self.running = True
        while self.running:
            if self.mdk.is_moving():
                pass
            else:
                with self.lock:
                    try:
                        self.mdk.move_relative(3, self.d * 0.000762, velocity=self.v)
                    except Exception as e:
                        self.mdk.logg.error(f"MadDeck Error: {e}")
                        self.stop()

    def stop(self):
        self.running = False
