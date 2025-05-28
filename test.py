import logging
import xarray as xr
import numpy as np
import sys
#from utils import lat_lon_deg_to_cartesian
"""
def main():
    logging.info('started')
    log_something()
    logging.info("log complete")s
"""

if __name__ == "__main__":
    print(sys.path)
    """
    #main()
    nx = 3
    lev = 1

    a = np.arange(5, 25)
    b = np.arange(100, 125)
    mg = np.meshgrid(a, b, indexing="ij")  # Use 'ij' indexing for (Nx,Ny)
    latlon = np.stack((mg[0], mg[1]), axis=-1)
    print(latlon.shape)
    print(latlon[:,:,0])
    
    #xy = lat_lon_deg_to_cartesian(node_lat = latlon[:,:,0], node_lon = latlon[:,:,1])
    #print(xy.shape)
    #print(xy.shape[0] - 1)
    # print(xy[:,:,1])
    #print(xy[:,:,1][1:-1:3, 1:-1:3].shape)

    
    nlev = int(np.log(max(xy.shape[:2])) / np.log(nx))
    nleaf = nx**nlev
    n = int(nleaf / (nx**lev))
    xm, xM = np.amin(xy[:, :, 0][:, 0]), np.amax(xy[:, :, 0][:, 0])
    ym, yM = np.amin(xy[:, :, 1][0, :]), np.amax(xy[:, :, 1][0, :])

    # avoid nodes on border
    dx = (xM - xm) / n
    dy = (yM - ym) / n
    lx = np.linspace(xm + dx / 2, xM - dx / 2, n)
    ly = np.linspace(ym + dy / 2, yM - dy / 2, n)

    mg = np.meshgrid(lx, ly, indexing="ij")  # Use 'ij' indexing for (Nx,Ny)
    print(mg)
    print(mg[0].shape, mg[1].shape)
    """
