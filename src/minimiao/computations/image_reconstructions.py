import numpy as np


class ImgRecon:

    def __init__(self, logg=None):
        self.logg = logg or self.setup_logging()

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    def point_scan_img_recon(self, photon_counts, gate_mask, n_lines, n_pixels, dwell_samples, bi_direction=False):
        gate = gate_mask.astype(bool)
        per_on = photon_counts[gate]
        expected = n_lines * n_pixels * dwell_samples
        if per_on.size != expected:
            raise ValueError(f"Gate-on samples = {per_on.size}, expected {expected}. "
                             f"(check dwell_s / mask / flyback)")
        img = per_on.reshape(n_lines, n_pixels, dwell_samples).sum(axis=2)
        if bi_direction:
            img[1::2] = img[1::2, ::-1]
        return img

