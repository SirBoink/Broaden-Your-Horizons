"""Random utilities for the project"""

# import ipdb
from xml.dom import minidom
from itertools import product

import numpy as np


def coords_from_svg(reference=None, size_multiplier=None, offset=None,
                    filename=None):
    """Parses the svg and extracts the coordinates of all relevant elements"""
    if filename is None:
        filename = './experiment_sheet.svg'
    doc = minidom.parse(filename)
    circles = doc.getElementsByTagName('circle')
    rects = [rect for rect in doc.getElementsByTagName('rect') if
             rect.getAttribute('id').startswith('mk_')]
    elements = {}
    for circle in circles:
        center = np.array((circle.getAttribute('cx'),
                           circle.getAttribute('cy')), dtype=float)
        center[1] *= -1
        radius = float(circle.getAttribute('r'))
        label = circle.getAttribute('id')
        fill_color = None
        stroke_color = None
        for style in circle.getAttribute('style').split(';'):
            if style.startswith('fill:'):
                fill_color = style[5:]
            if style.startswith('stroke:'):
                stroke_color = style[7:]
        elements[label] = Circle(center=center, radius=radius,
                                 fill_color=fill_color,
                                 stroke_color=stroke_color)

    for rect in rects:
        anchor = np.array((rect.getAttribute('x'), rect.getAttribute('y')),
                          dtype=float)
        sides = np.array((float(rect.getAttribute('width')),
                          float(rect.getAttribute('height'))))
        center = anchor + sides / 2
        center[1] *= -1
        label = rect.getAttribute('id')
        elements[label] = Circle(center=center, radius=sides[0])

    docu = doc.getElementsByTagName('svg')[0]
    paper = np.array([float(docu.getAttribute('width')[:-2]),
                      float(docu.getAttribute('height')[:-2])])
    return Elements(elements, reference=reference, paper_size=paper,
                    size_multiplier=size_multiplier, offset=offset)


class Circle(object):
    """Class for detecting whether coordinates are inside circle."""

    _radius_mm = 1  # Radius in mm
    _radius_rel = None  # Radius in psychopy's (-1, 1) relative coordinates
    _center_mm = np.zeros(2)
    _center_rel = None
    fill_color = '#FFFFFF'
    stroke_color = '#000000'

    def __init__(self, center, radius, units='mm', fill_color=None,
                 stroke_color=None):
        """Create a circle."""
        self.units = units
        self.radius = radius
        self.center = center
        if not (fill_color is None):
            self.fill_color = fill_color
        if not (stroke_color is None):
            self.stroke_color = stroke_color

    def __call__(self, xy):
        """Return True if (x, y) is within the circle.

        Otherwise, that's a paddlin'
        """
        disty = np.linalg.norm(xy - self.center[None, :], axis=1)
        return disty <= self.radius + 1  # HACK!

    @property
    def radius(self):
        """Radius of the circle, in mm."""
        return self._radius

    @radius.setter
    def radius(self, new_radius):
        if new_radius <= 0:
            raise ValueError(f'Radius must be > 0, was {new_radius}')
        self._radius = new_radius

    @property
    def units(self):
        """Units of the measurements. Defaults to mm."""
        return self._units

    @units.setter
    def units(self, units):
        if units in ['mm', 'rel']:
            self._units = units
        else:
            raise ValueError(f'Units {units} not in ["mm", "rel"]')

    @property
    def center(self):
        """Center of the circle."""
        return self._center_mm

    @center.setter
    def center(self, new_center):
        """Set the new center.

        Parameters
        ----------
        new_center : ndarray
        New center in absolute units (mm).

        """
        if len(self._center_mm) != len(new_center):
            raise ValueError('New center does not have the same size as old')
        self._center_mm = new_center

    @property
    def ncenter(self, ):
        """Center in normalized (psychopy) units of -1 to 1."""
        if self._center_rel is None:
            raise ValueError('Center not yet set in relative coordinates')
        return self._center_rel

    @ncenter.setter
    def ncenter(self, new_ncenter):
        self._center_rel = new_ncenter

    @property
    def nradius(self, ):
        """Radius of the circle in normalized units."""
        if self._radius_rel is None:
            raise ValueError('Radius not yet set in relative coordinates')
        return self._radius_rel

    @nradius.setter
    def nradius(self, new_value):
        self._radius_rel = new_value


class Elements(object):
    """Class for the collection of svg elements relevant to the experiment. The
    sole intention of this class is to be able to handle conversion to relative
    units, both in the mm space and psychopy's (-1, 1)x(-1, 1) space.

    """

    def __init__(self, elements, reference, paper_size, size_multiplier=None,
                 offset=None):
        self._elements = elements
        self.keys = elements.keys
        if size_multiplier is None:
            self.size_multiplier = 1
        else:
            self.size_multiplier = size_multiplier
        if offset is None:
            self.offset = np.zeros(2)
        else:
            self.offset = offset
        self.paper_size = paper_size
        self.recenter(reference)
        self.relative_units(reference)

    def __len__(self, ):
        return self._elements.__len__()

    def values(self):
        return self._elements.values()

    def keys(self):
        return self._elements.keys()

    def items(self):
        return self._elements.items()

    def __getitem__(self, index):
        """Returns the svg element called --mastr--."""
        if isinstance(index, int):
            key = list(self._elements.keys())[index]
        else:
            key = index
        return self._elements[key]

    def __setitem__(self, *args, **kwargs):
        """Setting items is not allowed after instanciation."""
        raise ValueError(f'Adding items to class {"Uh..."} is not allowed')

    def recenter(self, reference):
        # ipdb.set_trace()
        elements = self._elements
        if reference is None:
            zero = np.zeros(2)
        if isinstance(reference, str):
            zero = elements[reference].center
        elif isinstance(reference, (np.ndarray, list, tuple)):
            zero = np.array(reference)
        for element in elements.values():
            element.center = element.center - zero

    def relative_units(self, reference):
        self.paper_center = (self['mk_bl'].center + self['mk_tr'].center) / 2
        self.sizes = self.paper_size / 2 * self.size_multiplier
        ncoords = {}
        for key in self.keys():
            ncoords[key] = (self[key].center - self.paper_center)
        all_coords = np.zeros((len(self), 2))
        for idx, coord in enumerate(self.values()):
            all_coords[idx, :] = coord.center
        for key in self.keys():
            self._elements[key].ncenter = ncoords[key] / self.sizes + self.offset
            self._elements[key].nradius = self[key].radius / self.sizes.min()

    def normalize(self, coords_local):
        """Normalize coordinates."""
        return (coords_local + self[self.reference].center -
                self.paper_center) / self.sizes + self.offset

    def denormalize(self, coords_norm):
        """Return coordinates to non-normalized."""
        return self.sizes * coords_norm - self[self.reference].center + self.paper_center


def calc_subplots(num_plots):
    """Calculate a good arrangement for the subplots given the number."""
    if num_plots == 2 or num_plots == 3:
        return num_plots, 1

    sqrtns = np.sqrt(num_plots)
    if abs(sqrtns - np.ceil(sqrtns)) < 0.001:
        a1 = a2 = np.ceil(sqrtns)
    else:
        divs = num_plots % np.arange(2, num_plots)
        divs = np.arange(2, num_plots)[divs == 0]
        if divs.size == 0:
            return calc_subplots(num_plots + 1)
        else:
            a1 = divs[np.ceil(len(divs) / 2).astype(int)]
            a2 = num_plots / a1
    return int(a1), int(a2)


def get_labels():
    """Return the labels on the paper, divided by pt and st."""
    basket = coords_from_svg()
    labels = basket.keys()
    pt = sorted([key for key in labels
                  if key.startswith('pt_') and key != 'pt_start'])
    st = sorted([key for key in labels
                  if key.startswith('st_')])
    single_seqs = list(product(['pt_start'], pt))
    seq_seqs = list(product(['pt_start'], pt, st))
    return pt, st, single_seqs, seq_seqs
