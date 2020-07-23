# CREDIT: DEEPVOG https://github.com/pydsgz/DeepVOG

import numpy
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

import numpy as np
import matplotlib as mpl
from skimage.measure import label, regionprops
from skimage.morphology import closing, square
# from .bwperim import bwperim
# from .ellipses import LSqEllipse #The code is pulled frm https://github.com/bdhammel/least-squares-ellipse-fitting
from skimage.draw import ellipse_perimeter


"""Demonstration of least-squares fitting of ellipses
    __author__ = "Ben Hammel, Nick Sullivan-Molina"
    __credits__ = ["Ben Hammel", "Nick Sullivan-Molina"]
    __maintainer__ = "Ben Hammel"
    __email__ = "bdhammel@gmail.com"
    __status__ = "Development"
    Requirements 
    ------------
    Python 2.X or 3.X
    numpy
    matplotlib
    References
    ----------
    (*) Halir, R., Flusser, J.: 'Numerically Stable Direct Least Squares 
        Fitting of Ellipses'
    (**) http://mathworld.wolfram.com/Ellipse.html
    (***) White, A. McHale, B. 'Faraday rotation data analysis with least-squares 
        elliptical fitting'
"""


class LSqEllipse:

    def fit(self, data):
        """Lest Squares fitting algorithm
        Theory taken from (*)
        Solving equation Sa=lCa. with a = |a b c d f g> and a1 = |a b c>
            a2 = |d f g>
        Args
        ----
        data (list:list:float): list of two lists containing the x and y data of the
            ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
        Returns
        ------
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        """
        x, y = numpy.asarray(data, dtype=float)

        # Quadratic part of design matrix [eqn. 15] from (*)
        D1 = numpy.mat(numpy.vstack([x ** 2, x * y, y ** 2])).T
        # Linear part of design matrix [eqn. 16] from (*)
        D2 = numpy.mat(numpy.vstack([x, y, numpy.ones(len(x))])).T

        # forming scatter matrix [eqn. 17] from (*)
        S1 = D1.T * D1
        S2 = D1.T * D2
        S3 = D2.T * D2

        # Constraint matrix [eqn. 18]
        C1 = numpy.mat('0. 0. 2.; 0. -1. 0.; 2. 0. 0.')

        # Reduced scatter matrix [eqn. 29]
        M = C1.I * (S1 - S2 * S3.I * S2.T)

        # M*|a b c >=l|a b c >. Find eigenvalues and eigenvectors from this equation [eqn. 28]
        eval, evec = numpy.linalg.eig(M)

        # eigenvector must meet constraint 4ac - b^2 to be valid.
        cond = 4 * numpy.multiply(evec[0, :], evec[2, :]) - numpy.power(evec[1, :], 2)
        a1 = evec[:, numpy.nonzero(cond.A > 0)[1]]

        # |d f g> = -S3^(-1)*S2^(T)*|a b c> [eqn. 24]
        a2 = -S3.I * S2.T * a1

        # eigenvectors |a b c d f g>
        self.coef = numpy.vstack([a1, a2])
        self._save_parameters()

    def _save_parameters(self):
        """finds the important parameters of the fitted ellipse

        Theory taken form http://mathworld.wolfram
        Args
        -----
        coef (list): list of the coefficients describing an ellipse
           [a,b,c,d,f,g] corresponding to ax**2+2bxy+cy**2+2dx+2fy+g
        Returns
        _______
        center (List): of the form [x0, y0]
        width (float): major axis
        height (float): minor axis
        phi (float): rotation of major axis form the x-axis in radians
        """

        # eigenvectors are the coefficients of an ellipse in general form
        # a*x^2 + 2*b*x*y + c*y^2 + 2*d*x + 2*f*y + g = 0 [eqn. 15) from (**) or (***)
        a = self.coef[0, 0]
        b = self.coef[1, 0] / 2.
        c = self.coef[2, 0]
        d = self.coef[3, 0] / 2.
        f = self.coef[4, 0] / 2.
        g = self.coef[5, 0]

        # finding center of ellipse [eqn.19 and 20] from (**)
        x0 = (c * d - b * f) / (b ** 2. - a * c)
        y0 = (a * f - b * d) / (b ** 2. - a * c)

        # Find the semi-axes lengths [eqn. 21 and 22] from (**)
        numerator = 2 * (a * f * f + c * d * d + g * b * b - 2 * b * d * f - a * c * g)
        denominator1 = (b * b - a * c) * ((c - a) * numpy.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        denominator2 = (b * b - a * c) * ((a - c) * numpy.sqrt(1 + 4 * b * b / ((a - c) * (a - c))) - (c + a))
        width = numpy.sqrt(numerator / denominator1)
        height = numpy.sqrt(numerator / denominator2)

        # angle of counterclockwise rotation of major-axis of ellipse to x-axis [eqn. 23] from (**)
        # or [eqn. 26] from (***).
        phi = .5 * numpy.arctan((2. * b) / (a - c))

        self._center = [x0, y0]
        self._width = width
        self._height = height
        self._phi = phi

    @property
    def center(self):
        return self._center

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def phi(self):
        """angle of counterclockwise rotation of major-axis of ellipse to x-axis
        [eqn. 23] from (**)
        """
        return self._phi

    def parameters(self):
        return self._center, self._width, self._height, self._phi


def make_test_ellipse(center=[1, 1], width=1, height=.6, phi=3.14 / 5):
    """Generate Elliptical data with noise

    Args
    ----
    center (list:float): (<x_location>, <y_location>)
    width (float): semimajor axis. Horizontal dimension of the ellipse (**)
    height (float): semiminor axis. Vertical dimension of the ellipse (**)
    phi (float:radians): tilt of the ellipse, the angle the semimajor axis
        makes with the x-axis
    Returns
    -------
    data (list:list:float): list of two lists containing the x and y data of the
        ellipse. of the form [[x1, x2, ..., xi],[y1, y2, ..., yi]]
    """
    t = numpy.linspace(0, 2 * numpy.pi, 1000)
    x_noise, y_noise = numpy.random.rand(2, len(t))

    ellipse_x = center[0] + width * numpy.cos(t) * numpy.cos(phi) - height * numpy.sin(t) * numpy.sin(
        phi) + x_noise / 2.
    ellipse_y = center[1] + width * numpy.cos(t) * numpy.sin(phi) + height * numpy.sin(t) * numpy.cos(
        phi) + y_noise / 2.

    return [ellipse_x, ellipse_y]

def isolate_islands(prediction, threshold):
    bw = closing(prediction > threshold, square(3))
    labelled = label(bw)
    regions_properties = regionprops(labelled)
    max_region_area = 0
    select_region = 0
    for region in regions_properties:
        if region.area > max_region_area:
            max_region_area = region.area
            select_region = region
    output = np.zeros(labelled.shape)
    if select_region == 0:
        return output
    else:
        output[labelled == select_region.label] = 1
        return output


# input: output from bwperim -- 2D image with perimeter of the ellipse = 1
def gen_ellipse_contour_perim(perim, color="r"):
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0] < 6):
        return None
    else:
        try:
            vertices = np.array([input_points[0], input_points[1]]).T
            # Contour
            fitted = LSqEllipse()
            fitted.fit([vertices[:, 1], vertices[:, 0]])
            center, w, h, radian = fitted.parameters()
            ell = mpl.patches.Ellipse(xy=[center[0], center[1]], width=w * 2, height=h * 2, angle=np.rad2deg(radian),
                                      fill=False, color=color)
            # Because of the np indexing of y-axis, orientation needs to be minus
            rr, cc = ellipse_perimeter(int(np.round(center[0])), int(np.round(center[1])), int(np.round(w)),
                                       int(np.round(h)), -radian)
            return (rr, cc, center, w, h, radian, ell)
        except:
            return None


def gen_ellipse_contour_perim_compact(perim):
    # Vertices
    input_points = np.where(perim == 1)
    if (np.unique(input_points[0]).shape[0]) < 6 or (np.unique(input_points[1]).shape[0] < 6):
        return None
    else:
        try:
            vertices = np.array([input_points[0], input_points[1]]).T
            # Contour
            fitted = LSqEllipse()
            fitted.fit([vertices[:, 1], vertices[:, 0]])
            center, w, h, radian = fitted.parameters()
            # Because of the np indexing of y-axis, orientation needs to be minus
            return (center, w, h, radian)
        except:
            return None


def fit_ellipse(img, threshold=0.5, color="r", mask=None):
    isolated_pred = isolate_islands(img, threshold=threshold)
    perim_pred = bwperim(isolated_pred)

    # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
    if mask is not None:
        mask_bool = mask < 0.5
        perim_pred[mask_bool] = 0

    # masking bwperim_output on the img boundaries as 0
    perim_pred[0, :] = 0
    perim_pred[perim_pred.shape[0] - 1, :] = 0
    perim_pred[:, 0] = 0
    perim_pred[:, perim_pred.shape[1] - 1] = 0
    ellipse_info = gen_ellipse_contour_perim(perim_pred, color)

    return ellipse_info


def fit_ellipse_compact(img, threshold=0.5, mask=None):
    """Fitting an ellipse to the thresholded pixels which form the largest connected area.
    Args:
        img (2D numpy array): Prediction from the DeepVOG network (240, 320), float [0,1]
        threshold (scalar): thresholding pixels for fitting an ellipse
        mask (2D numpy array): Prediction from DeepVOG-3D network for eyelid region (240, 320), float [0,1].
                                intended for masking away the eyelid such as the fitting is better
    Returns:
        ellipse_info (tuple): A tuple of (center, w, h, radian), center is a list [x-coordinate, y-coordinate] of the ellipse centre.
                                None is returned if no ellipse can be found.
    """
    isolated_pred = isolate_islands(img, threshold=threshold)
    perim_pred = bwperim(isolated_pred)

    # masking eyelid away from bwperim_output. Currently not available in DeepVOG (But will be used in DeepVOG-3D)
    if mask is not None:
        mask_bool = mask < 0.5
        perim_pred[mask_bool] = 0

    # masking bwperim_output on the img boundaries as 0
    perim_pred[0, :] = 0
    perim_pred[perim_pred.shape[0] - 1, :] = 0
    perim_pred[:, 0] = 0
    perim_pred[:, perim_pred.shape[1] - 1] = 0

    ellipse_info = gen_ellipse_contour_perim_compact(perim_pred)
    return ellipse_info

# CREDIT: DEEPVOG https://github.com/pydsgz/DeepVOG

# vim: set ts=4 sts=4 sw=4 expandtab smartindent:
#
# This file was originally part of the octave-forge project
# Ported to python by Luis Pedro Coelho <luis@luispedro.org> (February 2008)
# Copyright (C) 2006       Soren Hauberg
# Copyright (C) 2008-2010  Luis Pedro Coelho (Python port)
#
# This program is free software; you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation; either version 2, or (at your option)
# any later version.
#
# This program is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this file.  If not, see <http://www.gnu.org/licenses/>.

import numpy as np

__all__ = ['bwperim']

def bwperim(bw, n=4):
    """
    perim = bwperim(bw, n=4)
    Find the perimeter of objects in binary images.
    A pixel is part of an object perimeter if its value is one and there
    is at least one zero-valued pixel in its neighborhood.
    By default the neighborhood of a pixel is 4 nearest pixels, but
    if `n` is set to 8 the 8 nearest pixels will be considered.
    Parameters
    ----------
      bw : A black-and-white image
      n : Connectivity. Must be 4 or 8 (default: 8)
    Returns
    -------
      perim : A boolean image
    """

    if n not in (4,8):
        raise ValueError('mahotas.bwperim: n must be 4 or 8')
    rows,cols = bw.shape

    # Translate image by one pixel in all directions
    north = np.zeros((rows,cols))
    south = np.zeros((rows,cols))
    west = np.zeros((rows,cols))
    east = np.zeros((rows,cols))

    north[:-1,:] = bw[1:,:]
    south[1:,:]  = bw[:-1,:]
    west[:,:-1]  = bw[:,1:]
    east[:,1:]   = bw[:,:-1]
    idx = (north == bw) & \
          (south == bw) & \
          (west  == bw) & \
          (east  == bw)
    if n == 8:
        north_east = np.zeros((rows, cols))
        north_west = np.zeros((rows, cols))
        south_east = np.zeros((rows, cols))
        south_west = np.zeros((rows, cols))
        north_east[:-1, 1:]   = bw[1:, :-1]
        north_west[:-1, :-1] = bw[1:, 1:]
        south_east[1:, 1:]     = bw[:-1, :-1]
        south_west[1:, :-1]   = bw[:-1, 1:]
        idx &= (north_east == bw) & \
               (south_east == bw) & \
               (south_west == bw) & \
               (north_west == bw)
    return ~idx * bw