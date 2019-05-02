from numpy.linalg import lstsq
from numpy import ones,vstack
from typing import List, Tuple


def get_line_equation(points: List) -> Tuple:
	x_coords, y_coords = zip(*points)
	A = vstack([x_coords,ones(len(x_coords))]).T
	m, c = lstsq(A, y_coords, rcond=None)[0]
	return (m, c)