#!/usr/bin/python

from __future__ import division, print_function
import matplotlib.pyplot as plt
import numpy as np
import sys, os, time, argparse

def optimize(points, shifts, extent=1, target=None, periodic=True, rng=None):

  '''
  Optimize vector arrangement on (periodic) grid.
  Requires as input a N*D list of vector origins (points) and
                    a N*D list of vectors (shifts).
  Returns a tuple containing the optimized shifts,
                             a list of valid swaps and
                             the total number of swaps.
  '''

  if not rng: rng = np.random
  shifts = shifts.copy()
  L,D    = points.shape
  N = int(min([0,2,6,12,24,40,72,126,240,272][D], L-1)) # Kissing number in N-D
  # Find N nearest neighbours
  if periodic:
    stack = points[None,].repeat(3**D,0) + ~np.indices((3,)*D).T.reshape(-1,1,D)%-3+1
  else:
    stack = points[None,]
  indices = np.array([np.square(point-stack).sum(axis=2).min(axis=0).argsort()[1:N+1]
                      for point in stack[0]])
  del stack
  # Start swappin'
  pairs = rng.permutation(np.tile(np.triu_indices(L,1), extent).T)
  valid = []
  if target is None or D != 2:
    for pair in pairs:
      if np.square(
           shifts[np.array((pair,pair[::-1]))]
         - shifts[indices[pair]].sum(axis=1)/N
         ).sum(axis=(1,2)).argmin():
        shifts[pair] = shifts[pair[::-1]]; valid.append(pair)
  # Loop separated to improve speed if target is not used
  else:
    for pair in pairs:
      if (
          np.square(
            shifts[np.array((pair,pair[::-1]))]
          - shifts[indices[pair]].sum(axis=1)/N
          ).sum(axis=(1,2)).argmin()
        and
          ((target[np.array((pair,pair[::-1]))]
          / np.sqrt(np.square(shifts[pair]).sum(axis=1))
          ).sum(axis=1) * (1,0.95)).argmin()
      ):
        shifts[pair] = shifts[pair[::-1]]; valid.append(pair)
  return shifts, valid, pairs.shape[0]

def target_from_img(filename, pts, contour=None):

  '''Read image and interpolate intensities on grid points'''

  from PIL import Image
  from scipy import interpolate, ndimage
  try:
    img = Image.open(filename)
  except FileNotFoundError:
    return
  arr = np.rot90(img, 3)
  avg = arr[...,:3].mean(axis=2)
  val = (avg-avg.min())/(avg.max()-avg.min())
  # Apply contour filter if contrast is too low
  if contour or (contour == None and abs(val - 0.5).mean() < 0.4):
    print('Contrast too low: applying contour filter')
    val = ndimage.sobel(ndimage.gaussian_filter(val,1))
  grd = np.array(
          np.meshgrid(*map(np.arange, avg.shape))
        ).reshape(2,-1)/np.array(avg.shape)[...,None]
  inp = interpolate.griddata(grd.T, val.T.ravel(), pts.T)
  return inp / inp.max()

def plot2d(ax, *args, **kwargs):

  scatter = kwargs.pop('scatter', False)
  if scatter:
    kwargs.pop('scale_units', None)
    args      = list(args)
    args[2:4] = [(np.square(args[2]) + np.square(args[3])) * 1e4]
    return ax.scatter(*args, **kwargs)
  else:
    return ax.quiver(*args, **kwargs)

def gaussian(x, integral=1, stretch=1):
  
  '''Calculate the Gaussian probabilty on a point x'''

  return np.exp(-(x*stretch)**2/2)/(2*np.pi)**.5 * stretch * integral

def inverse_gaussian(y, integral=1, stretch=1):
  
  '''Calculate the positive x-value for a certain Gaussian probabilty'''

  return (-np.log(y*(2*np.pi)**.5/(stretch*integral))*2)**.5/stretch

if __name__ == '__main__':

  start_time = time.time()

  # Parse input
  parser = argparse.ArgumentParser(
             description = 'Generate a vector field from random vectors.',
             epilog = 'Run with "-a -l=1 -f=1" to save only the final result.',
           )
  parser.add_argument('--size', '-n',            dest='size',       
                      metavar='N',   type=int,   default=25, 
                      help='Number of points per dimension [%(default)s]')
  parser.add_argument('--dimensions', '-d',      dest='dimensions',
                      metavar='D',   type=int,   default=2, 
                      help='Number of dimensions [%(default)s]')
  parser.add_argument('--not-periodic', '-np',   dest='periodic',  
                      action='store_const',      default=True, const=False,
                      help='Disable periodic boundary conditions')
  parser.add_argument('--extent', '-x',          dest='extent',    
                      metavar='EXT', type=int,   default=1, 
                      help='Try each possible swap this many times [%(default)s]')
  parser.add_argument('--seed', '-p',            dest='seed',      
                      metavar='SEED',type=int,   default=-1, 
                      help='Random seed')
  parser.add_argument('--scatter', '-t',         dest='scatter',   
                      action='store_const',      default=False, const=True,  
                      help='Plot circles instead of arrows')
  parser.add_argument('--show', '-s',            dest='show',      
                      action='store_const',      default=False, const=True,  
                      help='Show interactive plot')
  parser.add_argument('--input', '-i',           dest='input',     
                      metavar='INP', type=str,   default=None, 
                      help='Target image file name')
  parser.add_argument('--contour', '-c',         dest='contour',   
                      action='store_const',      default=None, const=True,  
                      help='Force contour filter on (priority)')
  parser.add_argument('--no-contour', '-nc',     dest='no_contour',
                      action='store_const',      default=None, const=False, 
                      help='Force contour filter off')
  parser.add_argument('--animation', '-a',       dest='animation', 
                      action='store_const',      default=False, const=True,  
                      help='Enable animation')
  parser.add_argument('--writer', '-w',          dest='writer',     
                      metavar='WRT', type=str,   default='ffmpeg', 
                      help='Animation backend [%(default)s]')
  parser.add_argument('--duration', '-l',        dest='duration',  
                      metavar='DUR', type=int,   default=10, 
                      help='Animation duration in seconds [%(default)s]')
  parser.add_argument('--fps', '-f',             dest='fps',       
                      metavar='FPS', type=float, default=30, 
                      help='Animation frame rate in 1/seconds [%(default)s]')
  parser.add_argument('--resolution', '-r',      dest='resolution',
                      metavar='RES', type=float, default=1080, 
                      help='Vertical resolution in pixels [%(default)s]')
  parser.add_argument('--aspect-ratio', '-ar',   dest='aspect_ratio',    
                      metavar='ASP', type=float, default=1.0, 
                      help='Aspect ratio of the output (hor/ver) [%(default)s]')
  parser.add_argument('--output', '-o',          dest='output',    
                      metavar='OUT', type=str,   default='animation.gif', 
                      help='Animation output file name, can include input\
                            arguments like animation_{size}x{size}.gif')
  args = parser.parse_args()

  # Seed rng
  if not 0 <= args.seed < 2**32: args.seed = np.random.randint(2**32)
  if hasattr(np.random, 'default_rng'):
    rng = np.random.default_rng(args.seed)
  else:
    np.random.seed(args.seed)
    rng = np.random
  print('Seed:', args.seed)
  
  # Construct random vectors
  N      = args.size
  D      = args.dimensions
  points = np.stack(np.meshgrid(*(np.arange(N),)*D)).reshape(D,-1).T/float(N)
  points[:,0] += points[:,1]*0.5
  points %= 1
  s_orig = rng.normal(size=(N**D,D)) / N

  # Open target image if applicable
  if D == 2 and args.input != None:
    contour = args.contour or args.no_contour
    target = target_from_img(args.input, points.T, contour=contour)
  else:
    target = None
  
  # Optimize vector arrangement
  print('Optimizing vector arrangement')
  try:
    shifts, valid, n_swaps = optimize(points=points, shifts=s_orig,
                                      target=target, periodic=args.periodic,
                                      extent=args.extent, rng=rng)
    print('Accepted swaps: {}/{} ({:2.3f}%)'.format(
          len(valid), n_swaps, n_swaps and len(valid)/n_swaps*100))
    print('Unchanged vectors:', (shifts==s_orig).all(axis=1).sum())
  except KeyboardInterrupt:
    args.animation = False

  nice = D == 2

  # Plot 2D
  if D == 2:

    X,Y    = points.T
    Uo, Vo = s_orig.T
    Un, Vn = shifts.T
    if nice:
      plt.style.use('dark_background')
      Co     = np.arctan2(Uo, Vo)
      Cn     = np.arctan2(Un, Vn)
      cmap = 'hsv'
    else:
      Co = Cn = []
      cmap = 'black'
    if args.show:
      fig    = plt.figure()
      ax0    = fig.add_subplot(1,2,1)
      plot2d(ax0, X,Y,Uo,Vo,Co, cmap=cmap, scale_units='xy', scatter=args.scatter)
      ax1    = fig.add_subplot(1,2,2)
      plot2d(ax1, X,Y,Un,Vn,Cn, cmap=cmap, scale_units='xy', scatter=args.scatter)
      fig.tight_layout()
      plt.ion()
      plt.show()
      plt.ioff()
      plt.pause(.001)
    
    # Build animation
    if args.animation:
      print('Animating')
      filename = args.output.format(**dict(args._get_kwargs()))
      from matplotlib import animation
      anim = plt.figure(figsize=(6 * args.aspect_ratio, 6))
      ax = anim.add_subplot(1,1,1)
      ax.axis('square')
      ax.set_aspect(1/args.aspect_ratio)
      ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
      anim.tight_layout()
      colors = np.arctan2(*s_orig.T)
      swaps = [[0,0]] + list(map(list,valid)) 
      swapper = iter(swaps)
      def animate(n=0):
        for _ in range(n):
          try:
            pair = next(swapper)
            s_orig[pair]=s_orig[pair[::-1]]
            colors[pair]=colors[pair[::-1]]
          except: pass
        ax.cla()
        U,V = s_orig.T
        return plot2d(ax, X,Y,U,V, colors, cmap=cmap, scatter=args.scatter),
      # Distribute swaps over n_frames following a Gaussian
      n_frames = args.duration * args.fps
      assert n_frames
      R = np.arange(n_frames) - (n_frames-1)/2.
      n_iter = len(swaps)
      stretch = inverse_gaussian(1.0, n_iter) / (n_frames/2) if n_iter > 2 else 1
      frames = np.array(0)
      # Gradually increase n_iter until all swaps are accounted for
      while frames.sum() < len(swaps):
        frames = gaussian(R, n_iter, stretch).astype(int)
        n_iter += 1
      # Start animating
      animation.FuncAnimation(anim, animate, frames=frames, blit=True, repeat=False
        ).save(filename, writer=args.writer, dpi=args.resolution/6, fps=args.fps)
      plt.close(anim)
      print('Wrote', filename)

    # Keep interactive plot open if still active
    # Silence stderr because there's nothing to draw
    stderr, sys.stderr = sys.stderr, open(os.devnull,'w')
    plt.show(block=True)
    sys.stderr = stderr

  # Plot 3D
  elif D == 3:
    from mpl_toolkits import mplot3d
    if nice:
      plt.style.use('dark_background')
      colors_old = abs(s_orig/np.sqrt(np.square(s_orig).sum(axis=1,keepdims=True)))
      colors_old = np.concatenate((colors_old, colors_old.repeat(2,axis=0)))
      colors_new = abs(shifts/np.sqrt(np.square(shifts).sum(axis=1,keepdims=True)))
      colors_new = np.concatenate((colors_new, colors_new.repeat(2,axis=0)))
    else:
      colors_old = colors_new = None
    X, Y, Z    = points.T
    Uo, Vo, Wo = s_orig.T
    Un, Vn, Wn = shifts.T
    fig    = plt.figure()
    ax0    = fig.add_subplot(1,2,1, projection='3d', proj_type='ortho')
    ax0.quiver3D(X,Y,Z,Uo,Vo,Wo, colors = colors_old, length=0.1, linewidth=2)
    ax1    = fig.add_subplot(1,2,2, projection='3d', proj_type='ortho')
    ax1.quiver3D(X,Y,Z,Un,Vn,Wn, colors = colors_old, length=0.1, linewidth=2)
    plt.tight_layout()
    plt.show(block=True)
  
  time_spent = time.time() - start_time
  print('Time spent: {:02.0f}:{:02.0f}'.format(*divmod(time_spent,60)))

