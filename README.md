This is a Python package for working with coordinate transforms in three dimensions. It provides the following:
 - Conversion between common spatial rotation representations
 - The `SE3` class for applying rigid translations and rotations to points and vectors
 - `TransformTree` and `TransformForest` types for handling named coordinate frames and frame hierarchies

## Dependencies
Numpy is the only dependency. It is installed automatically with this package.

## Installation
Clone this repo and do `pip install -e /path/to/spatial-effects`.

Verify the installation by running the tests:

    python -m unittest discover

## Usage of SE(3) type:
```
    x = [1, 0, 0]              # Define a point/translation
    r = [0, 0, pi/2]           # Define an orientation/rotation
    T1 = SE3()                 # Create identity transformation
    T2 = SE3(x, r)             # Lists, tuples, or ndarrays ok here
    callable(T1)               # True
    T2(x)                      # array([1., 1., 0.])
    T2.R @ x + T2.t            # same
    T2.inverse(T2(x))          # x
    T1.matrix                  # np.eye(4)
    T2.vec                     # array([1, 0, 0, 0, 0, 1.571])
    (T1 + T2.vec).vec          # same (in this case)
    T2 - T1                    # same (in this case)
    T1 + T2                    # ValueError! Cannot ⊞ two manifolds
    T1 + (T2 - T1) == T2       # always True
    (T1 + v) - T1 == v         # always True (v is any 6 DOF vector)
```


## Conventions
Points, vectors, quaternions, and matrices are all represented as Numpy arrays that follow Numpy's row-major data layout. For example, five 3D points or vectors would be passed to the library functions as a 5x3 array.

Unit quaternions are always expressed in (w, x, y, z) order, where w is the real or scalar component. The Hamilton convention for quaternion muliplication (as opposed to JPL) is followed consistently.
