#!/usr/bin/env python3

import h5py
import numpy as np

import sys


def printSteps(fname):
    """ Display contents of HDF5 file: step, iteration and time """
    ifile = h5py.File(fname, "r")
    print(fname, "contains the following steps:")
    print("hdf5 step number".rjust(15), "sph iteration".rjust(15), "time".rjust(15))
    for i in range(len(list(ifile["/"]))):
        h5step = ifile["Step#%d" % i]
        print("%5d".rjust(14) % i, "%5d".rjust(14) % h5step.attrs["iteration"][0],
              "%5f".rjust(14) % h5step.attrs["time"][0])


def readStep(fname, step):
    ifile = h5py.File(fname, "r")
    try:
        h5step = ifile["Step#%s" % step]
        return h5step
    except KeyError:
        print(fname, "step %s not found" % step)
        printSteps(fname)
        sys.exit(1)

def plotSlice(fname, step, txtname):
    """ Plot a 2D xy-cross section with particles e.g. abs(z) < 0.1, using density as color """

    h5step = readStep(fname, step)

    # x = np.array(h5step["x"])
    # y = np.array(h5step["y"])
    # z = np.array(h5step["z"])
    # h = np.array(h5step["h"])
    # vx = np.array(h5step["vx"])
    # vy = np.array(h5step["vy"])
    # vz = np.array(h5step["vz"])

    # rho = np.array(h5step["rho"])
    # divv = np.array(h5step["divv"])
    # curlv = np.array(h5step["curlv"])

    f = open(txtname, "w")

    chunkNo = 10
    chunkSize = int(len(h5step["z"])/chunkNo)
    for ind in range(chunkNo):
        x = np.array(h5step["x"][ind*chunkSize:(ind+1)*chunkSize])
        y = np.array(h5step["y"][ind*chunkSize:(ind+1)*chunkSize])
        z = np.array(h5step["z"][ind*chunkSize:(ind+1)*chunkSize])
        vx = np.array(h5step["vx"][ind*chunkSize:(ind+1)*chunkSize])
        vy = np.array(h5step["vy"][ind*chunkSize:(ind+1)*chunkSize])
        vz = np.array(h5step["vz"][ind*chunkSize:(ind+1)*chunkSize])

        # Calculate the modulus of velocity
        # for index in range(chunkSize):
        #     v = np.sqrt (vx[index]*vx[index] + vy[index]*vy[index] + vz[index]*vz[index])
        #     str = "%lf %lf %lf %lf\n" % (x[index], y[index], z[index], v)
        #     f.write(str)
        
        # Write the velocities to the text file
        for index in range(chunkSize):
            str = "%lf %lf %lf %lf %lf %lf\n" % (x[index], y[index], z[index], vx[index], vy[index], vz[index])
            f.write(str)

        print("chunk = ", ind)

    f.close()


if __name__ == "__main__":
    # first cmdline argument: hdf5 file name to plot
    fname = sys.argv[1]

    # second cmdline argument: hdf5 step number to plot or print (-p) and exit
    step = sys.argv[2]
    if step == "-p":
        printSteps(fname)
        sys.exit(1)
    
    # first cmdline argument: hdf5 file name to plot
    txtname = sys.argv[3]

    plotSlice(fname, step, txtname)


