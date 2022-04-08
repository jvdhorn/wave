# Wave

Wave is a tool to optimize an arrangement of vectors on a (periodic) N-D grid, by randomly swapping their positions.

# Example

Visualization of the process, which was generated with `python wave.py -n 50 -p 1337 -r 512 -a`.
![Example output animation](https://github.com/jvdhorn/wave/blob/main/data/animation.gif)

# Dependencies

* numpy
* matplotlib
* scipy (optional: only required when using target image)
