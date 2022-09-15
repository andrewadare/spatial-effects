# spatial-effects
This is a lightweight Python package for working with coordinate transforms in three dimensions. It provides the following:
 - Conversion between common spatial rotation representations
 - The `SE3` class for applying rigid translations and rotations to points and vectors
 - [TODO] `Transform` and `TransformTree` types for handling named coordinate frames and frame hierarchies

## Dependencies
Numpy is the only dependency. It is installed automatically with this package.

## Installation
Clone this repo and do `pip install -e /path/to/spatial-effects`.

## Usage
TODO: examples. For now, see unit tests

## Conventions
Points, vectors, quaternions, and matrices are all represented as Numpy arrays that follow Numpy's row-major data layout. For example, five 3D points or vectors would be passed to the library functions as a 5x3 array.

Unit quaternions are always expressed in (w, x, y, z) order, where w is the real or scalar component. The Hamilton convention for quaternion muliplication (as opposed to JPL) is followed consistently.
