import numpy as np


class ImgRecon:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()
        self.point_scan_gate_mask = None
        self.point_scan_n_pixels = 0
        self.point_scan_n_lines = 0
        self.point_scan_dwell_samples = 0

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def point_scan_img_recon(self, photon_counts, bi_direction=False):
        gate = self.point_scan_gate_mask.astype(bool)
        per_on = photon_counts[gate]
        expected = self.point_scan_n_lines * self.point_scan_n_pixels * self.point_scan_dwell_samples
        if per_on.size != expected:
            raise ValueError(f"Gate-on samples = {per_on.size}, expected {expected}. "
                             f"(check dwell_s / mask / flyback)")
        img = per_on.reshape(self.point_scan_n_lines, self.point_scan_n_pixels, self.point_scan_dwell_samples).sum(axis=2)
        if bi_direction:
            img[1::2] = img[1::2, ::-1]
        return img
