import math
import numpy as np
from timeit import timeit
from matplotlib import pyplot as plt
from gtda.homology import VietorisRipsPersistence, WeakAlphaPersistence
from gtda.plotting import plot_diagram
import gudhi as gd


# Function to generate points from a circle with noise

def _randomPointCircle(sigma=0.05):
    t = 2 * math.pi * np.random.rand()
    return [math.cos(t) + (sigma * np.random.randn()), math.sin(t) + (sigma * np.random.randn())]

def randCircle(n=100, sigma=0.05):
    return np.array([_randomPointCircle(sigma) for _ in range(n)])

X = randCircle(n=500, sigma=0.2)
plt.scatter([X[:,0]], [X[:,1]])
plt.show()


# Define Persistence Diagram Computing Methods

# 1. Using GTDA

VR1 = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], collapse_edges=False, max_edge_length=3)
ECVR1 = VietorisRipsPersistence(homology_dimensions=[0, 1, 2], collapse_edges=True, max_edge_length=3)
Alpha1 = WeakAlphaPersistence(homology_dimensions=[0, 1, 2], max_edge_length=3)

# 2. Using Gudhi

def VR2(X): 
    return gd.RipsComplex(points=X, max_edge_length=3.0).create_simplex_tree(max_dimension=2).persistence()

# Edge Collapse in this Library seems to be broken!
def ECVR2(X): 
    return gd.RipsComplex(points=X, max_edge_length=3.0).create_simplex_tree(max_dimension=2).collapse_edges().persistence()

def Alpha2(X): 
    return gd.AlphaComplex(points=X).create_simplex_tree().persistence()


# Compute Persistence Diagrams

# 1. Runtime Comparisons
timeit("VR1.fit_transform([X])", globals=globals(), number=1)
timeit("VR2(X)", globals=globals(), number=1)

timeit("ECVR1.fit_transform([X])", globals=globals(), number=1)

timeit("Alpha1.fit_transform([X])", globals=globals(), number=1)
timeit("Alpha2(X)", globals=globals(), number=1)


# 2. GTDA persistence diagrams are plotted using Plotly backend

D11 = VR1.fit_transform([X])
D12 = ECVR1.fit_transform([X])
D13 = Alpha1.fit_transform([X])

plot_diagram(D11[0])
plot_diagram(D12[0])
plot_diagram(D13[0])

# 3. Gudhi Persistence Diagrams use Matplotlib backend

D21 = VR2(X)
D23 = Alpha2(X)

gd.plot_persistence_diagram(D21)
gd.plot_persistence_diagram(D23)





