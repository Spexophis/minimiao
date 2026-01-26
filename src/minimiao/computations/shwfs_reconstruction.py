# -*- coding: utf-8 -*-
# Copyright (c) 2025 Peter Kner, Ruizhe Lin
# Licensed under the MIT License.


import os
import time

import numpy as np
import tifffile as tf
from scipy.signal import fftconvolve as corr
from skimage.filters import threshold_otsu

from minimiao.utilities import zernike_generator as tz
from minimiao.utilities import image_processor as ipr

fft2 = np.fft.fft2
ifft2 = np.fft.ifft2
fftshift = np.fft.fftshift
pi = np.pi


class WavefrontSensing:

    def __init__(self, logg=None):
        self.logg = logg or logger.setup_logging()
        self.n_lenslets_x = 18
        self.n_lenslets_y = 18
        self.n_lenslets = self.n_lenslets_x * self.n_lenslets_y
        self.x_center_base = 1231
        self.y_center_base = 926
        self.x_center_offset = 1231
        self.y_center_offset = 926
        self.lenslet_spacing = 23  # spacing between each lenslet
        self.hsp = 16  # size of subimage is 2 * hsp
        self.bg = 0.1
        self.pixel_size = .0065  # mm
        self.calfactor = (self.pixel_size / 5.2) * 150  # pixel size * focalLength * pitch
        self.method = 'correlation'
        self.mag = 1
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        section_corr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(section_corr.argmax(), section_corr.shape)
        self._ref = None
        self._meas = None
        self.wf = None
        self.im = None

    @staticmethod
    def setup_logging():
        import logging
        logging.basicConfig(format='%(levelname)s: %(message)s', level=logging.INFO)
        return logging

    @property
    def ref(self):
        return self._ref

    @ref.setter
    def ref(self, new_ref):
        self._ref = new_ref

    @property
    def meas(self):
        return self._meas

    @meas.setter
    def meas(self, new_meas):
        self._meas = new_meas

    def update_parameters(self, parameters):
        self.x_center_base = parameters[0]
        self.y_center_base = parameters[1]
        self.x_center_offset = parameters[2]
        self.y_center_offset = parameters[3]
        self.n_lenslets_x = parameters[4]
        self.n_lenslets_y = parameters[5]
        self.n_lenslets = self.n_lenslets_x * self.n_lenslets_y
        self.lenslet_spacing = parameters[6]
        self.hsp = parameters[7]
        section = np.ones((2 * self.hsp, 2 * self.hsp))
        section_corr = corr(1.0 * section, 1.0 * section[::-1, ::-1], mode='full')
        self.CorrCenter = np.unravel_index(section_corr.argmax(), section_corr.shape)
        self.bg = parameters[8]
        self.calfactor = (self.pixel_size / 5.2) * 150

    def wavefront_reconstruction(self, md='correlation', rt=False):
        (gradx, grady) = self.get_gradient_xy(mtd=md)
        self.wf = self.gradient_to_wavefront(gradx, grady)
        if rt:
            return self.wf

    def gradient_to_wavefront(self, gradx, grady):
        gradx = np.pad(gradx, ((1, 1), (1, 1)), 'constant')
        grady = np.pad(grady, ((1, 1), (1, 1)), 'constant')
        extx, exty = self._hudgins_extend_mask(gradx, grady)
        phi = self._reconstruction_hudgins(extx, exty)
        phicorr = self._remove_global_waffle(phi)
        msk = self._elliptical_mask((self.n_lenslets_y / 2, self.n_lenslets_x / 2),
                                    (self.n_lenslets_y + 2, self.n_lenslets_x + 2))
        phicorr = phicorr * msk
        return phicorr[1:1 + self.n_lenslets_y, 1:1 + self.n_lenslets_x]

    def get_gradient_xy(self, mtd='correlation'):
        """ Determines Gradients by Correlating each section with its base reference section"""
        nx = self.n_lenslets_x
        ny = self.n_lenslets_y
        hsp = self.hsp
        rx = int(nx / 2.)
        ry = int(ny / 2.)
        bot_base = self.y_center_base - ry * self.lenslet_spacing
        left_base = self.x_center_base - rx * self.lenslet_spacing
        bot_offset = self.y_center_offset - ry * self.lenslet_spacing
        left_offset = self.x_center_offset - rx * self.lenslet_spacing
        base = self._sub_back(self.ref, self.bg)
        offset = self._sub_back(self.meas, self.bg)
        self.im = np.zeros((2, 2 * self.hsp * ny, 2 * self.hsp * nx))
        gradx = np.zeros((ny, nx))
        grady = np.zeros((ny, nx))
        for iy in range(ny):
            for ix in range(nx):
                vert_base = int(bot_base + iy * self.lenslet_spacing)
                horiz_base = int(left_base + ix * self.lenslet_spacing)
                vert_offset = int(bot_offset + iy * self.lenslet_spacing)
                horiz_offset = int(left_offset + ix * self.lenslet_spacing)
                secbase = base[(vert_base - hsp):(vert_base + hsp), (horiz_base - hsp):(horiz_base + hsp)]
                sec = offset[(vert_offset - hsp):(vert_offset + hsp), (horiz_offset - hsp):(horiz_offset + hsp)]
                self.im[0, iy * 2 * hsp: (iy + 1) * 2 * hsp, ix * 2 * hsp: (ix + 1) * 2 * hsp] = secbase
                self.im[1, iy * 2 * hsp: (iy + 1) * 2 * hsp, ix * 2 * hsp: (ix + 1) * 2 * hsp] = sec
                if mtd == 'correlation':
                    seccorr = corr(1.0 * secbase, 1.0 * sec[::-1, ::-1], mode='full')
                    py, px = self._parabolic_fit(seccorr)
                    gradx[iy, ix] = (self.CorrCenter[1] - px) * self.calfactor
                    grady[iy, ix] = (self.CorrCenter[0] - py) * self.calfactor
                elif mtd == 'centerofmass':
                    sy, sx = ipr.find_center_of_mass(secbase)
                    py, px = ipr.find_center_of_mass(sec)
                    gradx[iy, ix] = (px - sx) * self.calfactor
                    grady[iy, ix] = (py - sy) * self.calfactor
        return gradx, grady

    def save_wfs_results(self, file_name, dm):
        try:
            tf.imwrite(file_name + f'_{dm.dm_serial}_wfs_base_raw.tif', self.ref)
        except Exception as e:
            self.logg.error(f"Error saving wfs base: {e}")
        try:
            tf.imwrite(file_name + f'_{dm.dm_serial}_wfs_offset_raw.tif', self.meas)
        except Exception as e:
            self.logg.error(f"Error saving wfs offset: {e}")
        try:
            tf.imwrite(file_name + f'_{dm.dm_serial}_wfs_processed.tif', self.im)
        except Exception as e:
            self.logg.error(f"Error saving wfs processed images: {e}")
        try:
            tf.imwrite(file_name + f'_{dm.dm_serial}_recon_wf.tif', self.wf)
        except Exception as e:
            self.logg.error(f"Error saving wfs wavefront: {e}")

    @staticmethod
    def _hudgins_extend_mask(gradx, grady):
        """ extension technique Poyneer 2002 """
        nx, ny = gradx.shape
        if nx % 2 == 0:  # even
            mx = nx / 2
        else:  # odd
            mx = (nx + 1) / 2
        if ny % 2 == 0:  # even
            my = ny / 2
        else:  # odd
            my = (ny + 1) / 2
        for jj in range(int(nx)):
            for ii in range(int(my), int(ny)):
                if grady[jj, ii] == 0.0:
                    grady[jj, ii] = grady[jj, ii - 1]
            for ii in range(int(my), -1, -1):
                if grady[jj, ii] == 0.0:
                    grady[jj, ii] = grady[jj, ii + 1]
        for jj in range(int(ny)):
            for ii in range(int(mx), int(nx)):
                if gradx[ii, jj] == 0.0:
                    gradx[ii, jj] = gradx[ii - 1, jj]
            for ii in range(int(mx), -1, -1):
                if gradx[ii, jj] == 0.0:
                    gradx[ii, jj] = gradx[ii + 1, jj]
        gradxe = gradx.copy()
        gradye = grady.copy()
        gradxe[:, ny - 1] = -1.0 * gradx[:, :(ny - 1)].sum(1)
        gradye[nx - 1, :] = -1.0 * grady[:(nx - 1), :].sum(0)
        return gradxe, gradye

    @staticmethod
    def _reconstruction_hudgins(gradx, grady):
        """ wavefront reconstruction from gradients Hudgins Geometry, Poyneer 2002 """
        sx = fft2(gradx)
        sy = fft2(grady)
        ny, nx = gradx.shape
        ky, kx = np.meshgrid(np.arange(ny), np.arange(nx), indexing='ij')
        numx = (np.exp(-2j * pi * kx / nx) - 1)
        numy = (np.exp(-2j * pi * ky / ny) - 1)
        den = 4 * (np.sin(pi * kx / nx) ** 2 + np.sin(pi * ky / ny) ** 2)
        # sw = (numx * sx + numy * sy) / den
        sw = np.divide((numx * sx + numy * sy), den, where=den != 0)
        sw[0, 0] = 0.0
        return (ifft2(sw)).real

    @staticmethod
    def _remove_global_waffle(phi):
        ny, nx = phi.shape
        wmode = np.zeros((ny, nx))
        constant_num = 0
        constant_den = 0
        # a waffle-mode vector of +-1 for a given pixel of the Wavefront
        for x in range(nx):
            for y in range(ny):
                if (x + y) / 2 - np.round((x + y) / 2) == 0:
                    wmode[y, x] = 1
                else:
                    wmode[y, x] = -1
        for i in range(ny):
            for k in range(nx):
                temp = phi[i, k] * wmode[i, k]
                temp2 = wmode[i, k] * wmode[i, k]
                constant_num = constant_num + temp
                constant_den = constant_den + temp2
        constant = constant_num / constant_den
        return phi - constant * wmode

    def _parabolic_fit(self, sec):
        try:
            init_max_loc = np.unravel_index(sec.argmax(), sec.shape)
            sec_zoom = sec[(init_max_loc[0] - 1):(init_max_loc[0] + 2), (init_max_loc[1] - 1):(init_max_loc[1] + 2)]
            gradx = init_max_loc[1] + 0.5 * (1.0 * sec_zoom[1, 0] - 1.0 * sec_zoom[1, 2]) / (
                    1.0 * sec_zoom[1, 0] + 1.0 * sec_zoom[1, 2] - 2.0 * sec_zoom[1, 1])
            grady = init_max_loc[0] + 0.5 * (1.0 * sec_zoom[0, 1] - 1.0 * sec_zoom[2, 1]) / (
                    1.0 * sec_zoom[0, 1] + 1.0 * sec_zoom[2, 1] - 2.0 * sec_zoom[1, 1])
        except:  # IndexError
            gradx = self.CorrCenter[0]
            grady = self.CorrCenter[1]
        return grady, gradx

    @staticmethod
    def _sub_back(img, factor):
        thresh = factor * threshold_otsu(img)
        binary = img > thresh
        return (img - thresh) * binary

    @staticmethod
    def _elliptical_mask(radius, size):
        coord_x = np.arange(0.5, size[0], 1.0)
        coord_y = np.arange(0.5, size[1], 1.0)
        y, x = np.meshgrid(coord_y, coord_x)
        x -= size[0] / 2.
        y -= size[1] / 2.
        return (x * x / (radius[0] * radius[0])) + (y * y / (radius[1] * radius[1])) <= 1

    def generate_influence_matrices(self, data_folder, dm, sv=None, verbose=False):
        n_actuators, amp = dm.n_actuator, dm.amp
        dm.nly, dm.nlx = self.n_lenslets_y, self.n_lenslets_x
        dm.nls = self.n_lenslets_y * self.n_lenslets_x
        dm.zernike = tz.zernike_polynomials(size=[self.n_lenslets_y, self.n_lenslets_x])
        dm.zslopes = tz.zernike_derivatives(size=[self.n_lenslets_y, self.n_lenslets_x])
        influence_matrix_phase = np.zeros((self.n_lenslets, n_actuators))
        wfs_phase = np.zeros((n_actuators, self.n_lenslets_y, self.n_lenslets_x))
        influence_matrix_zonal = np.zeros((2 * self.n_lenslets, n_actuators))
        influence_matrix_modal = np.zeros((dm.n_zernike, n_actuators))
        for filename in os.listdir(data_folder):
            if filename.endswith(".tif") & filename.startswith("actuator"):
                ind = int(filename.split("_")[1])
                if verbose:
                    self.logg.info(filename.split("_")[1])
                data_stack = tf.imread(os.path.join(data_folder, filename))
                n, x, y = data_stack.shape
                if n != 4:
                    raise "The image number has to be 4"
                self.ref, self.meas = data_stack[0], data_stack[1]
                gdxp, gdyp = self.get_gradient_xy()
                wfp = self.gradient_to_wavefront(gdxp, gdyp)
                self.ref, self.meas = data_stack[2], data_stack[3]
                gdxn, gdyn = self.get_gradient_xy()
                wfn = self.gradient_to_wavefront(gdxn, gdyn)
                # phase
                msk = (wfp != 0.0).astype(np.float32)
                mn = wfp.sum() / msk.sum()
                wfp = msk * (wfp - mn)
                msk = (wfn != 0.0).astype(np.float32)
                mn = wfn.sum() / msk.sum()
                wfn = msk * (wfn - mn)
                wfg = (wfp - wfn) / (2 * amp)
                wfs_phase[ind] = wfg
                influence_matrix_phase[:, ind] = wfg.reshape(self.n_lenslets)
                # zonal
                influence_matrix_zonal[:self.n_lenslets, ind] = ((gdxp - gdxn) / (2 * amp)).reshape(self.n_lenslets)
                influence_matrix_zonal[self.n_lenslets:, ind] = ((gdyp - gdyn) / (2 * amp)).reshape(self.n_lenslets)
                # modal
                a1 = ipr.get_eigen_coefficients(np.concatenate((gdxp.flatten(), gdyp.flatten())), dm.zslopes, 14)
                a2 = ipr.get_eigen_coefficients(np.concatenate((gdxn.flatten(), gdyn.flatten())), dm.zslopes, 14)
                influence_matrix_modal[:, ind] = ((a1 - a2) / (2 * amp)).flatten()
        control_matrix_phase = ipr.pseudo_inverse(influence_matrix_phase, n=14)
        control_matrix_zonal = ipr.pseudo_inverse(influence_matrix_zonal, n=14)
        control_matrix_modal = ipr.pseudo_inverse(influence_matrix_modal, n=14)
        if sv is not None:
            fd = sv.configs["Adaptive Optics"]["Deformable Mirrors"][dm.dm_name]["Calibration File Folder"]
            t = time.strftime("%Y_%m_%d_%H_%M")
            fn = os.path.join(fd, f"influence_function_phase_{t}.tif")
            tf.imwrite(fn, influence_matrix_phase)
            fn = os.path.join(fd, f"control_matrix_phase_{t}.tif")
            tf.imwrite(fn, control_matrix_phase)
            dm.control_matrix_phase = control_matrix_phase
            sv.configs["Adaptive Optics"]["Deformable Mirrors"][dm.dm_name]["Phase Control Matrix"] = fn
            fn = os.path.join(fd, f"influence_function_images_{t}.tif")
            tf.imwrite(fn, wfs_phase)
            sv.configs["Adaptive Optics"]["Deformable Mirrors"][dm.dm_name]["Influence Function Images"] = fn
            fn = os.path.join(fd, f"influence_function_zonal_{t}.tif")
            tf.imwrite(fn, influence_matrix_zonal)
            fn = os.path.join(fd, f"control_matrix_zonal_{t}.tif")
            tf.imwrite(fn, control_matrix_zonal)
            dm.control_matrix_zonal = control_matrix_zonal
            sv.configs["Adaptive Optics"]["Deformable Mirrors"][dm.dm_name]["Zonal Control Matrix"] = fn
            fn = os.path.join(fd, f"influence_function_modal_{t}.tif")
            tf.imwrite(fn, influence_matrix_modal)
            fn = os.path.join(fd, f"control_matrix_modal_{t}.tif")
            tf.imwrite(fn, control_matrix_modal)
            dm.control_matrix_modal = control_matrix_modal
            sv.configs["Adaptive Optics"]["Deformable Mirrors"][dm.dm_name]["Modal Control Matrix"] = fn
            sv.write_config(sv.configs, sv.cfd)
