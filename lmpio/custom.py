#!/bin/env python3

import os,sys
from typing import List

try:
    import numpy as np
except ModuleNotFoundError:
    print("numpy not installed. Please install numpy")
    sys.exit("terminated")

CUSTOM_NPY_EXT = '_custom.npy'

def load_custom(filename: str,savenpy=True,loadnpy=True,sortbyid=True, splitargs=False) -> dict:
    if not os.path.isfile(filename):
        raise FileNotFoundError( f"No such file or directory: '{filename}'" )
    fnpy = os.path.splitext(filename)[0]+CUSTOM_NPY_EXT
    if os.path.isfile(fnpy) and loadnpy and os.path.getmtime(fnpy) >= os.path.getmtime(filename):
        specs = read_specs(filename)
        print(f"loading data from '{fnpy}'")
        specs['data'] = np.load(fnpy)
    else:
        specs = read_custom(filename,sortbyid=sortbyid, splitargs=False)
        if savenpy:
            _save_custom_binary(fnpy,specs['data'])
    if splitargs:
        specs = customdata_splitargs(specs,remove_data=True)
    return specs

def read_specs(filename: str) -> dict:
    specs = dict()
    with open(filename) as f:
        for i, line in enumerate(f):
            if i == 3:
                specs['num_atoms'] = int(line.strip())
            if i == 4:
                specs['boundary'] = line.strip().split(' ')[3:]
            if i == 5:
                xlim = [float(entry) for entry in line.strip().split(' ')]
            if i == 6:
                ylim = [float(entry) for entry in line.strip().split(' ')]
            if i == 7:
                zlim = [float(entry) for entry in line.strip().split(' ')]
                simbox = np.array([xlim,ylim,zlim])
                specs['simulation_box'] = simbox
            if i == 8:
                specs['args'] = line.strip().split(' ')[2:]
            if i == 9:
                break
    return specs

def nskip(filehandler,n):
    line = filehandler.readline()
    if line == '':
        return False
    [filehandler.readline() for i in range(n-1)]
    return True

def customdata_splitargs(specs: dict,remove_data=True) -> dict:
    args = specs['args']
    data = specs['data']

    cpargs = [str(arg) for arg in args]
    if ('x' in cpargs and 'y' in cpargs and 'z' in cpargs):
        cpargs.append('position')
        cpargs.remove('x')
        cpargs.remove('y')
        cpargs.remove('z')
    if ('vx' in args and 'vy' in args and 'vz' in args):
        cpargs.append('velocity')
        cpargs.remove('vx')
        cpargs.remove('vy')
        cpargs.remove('vz')

    for arg in cpargs:
        if arg == 'position':
            specs['position'] = data[:,:,[args.index(d) for d in ['x','y','z']]]
            continue
        if arg == 'velocity':
            specs['velocity'] = data[:,:,[args.index(d) for d in ['vx','vy','vz']]]
            continue
        specs[arg] = data[:,:,args.index(arg)]
        if arg == 'id':
            specs[arg] = specs[arg].astype(np.int)
    if remove_data:
        del specs['data']

    return specs

def read_custom(filename: str, sortbyid=True, splitargs=False) -> dict:
    # print(f"reading '{filename}'")

    specs = read_specs(filename)
    N     = specs['num_atoms']
    args  = specs['args']
    nargs = len(args)

    data = list()
    with open(filename) as f:
        while nskip(f,9):
            snap = np.empty((N,nargs))
            for i in range(N):
                line = f.readline()
                snap[i] = [float(elem) for elem in line.strip().split(' ')]
            if sortbyid and 'id' in args:
                snap = snap[np.argsort(snap[:,args.index('id')])]
            data.append(snap)
    specs['data'] = np.array(data)
    if splitargs:
        specs = customdata_splitargs(specs,remove_data=True)
    return specs

def _save_custom_binary(outname: str,data: np.ndarray) -> None:
    if os.path.splitext(outname)[-1] == '.npy':
        outname += '.npy'
    np.save(outname,data)    


def write_custom(outfn: str, data: dict, extension='custom', append=False) -> None:
    """
    Writes configuration to custom file
    
    Parameters
    ----------
    outfn : string 
        name of custom file
    
    """
    # handle extension
    extension = extension.lower()
    if extension[0] != '.':
        extension = '.' + extension
    if extension not in outfn.lower():
        outfn += extension

    # select data entries
    datalists = list()
    for arg in data['args']:
        if arg == 'x' and 'x' not in data.keys():
            if 'position' not in data.keys():
                raise KeyError(f'Neither x nore position was provided in data dictionary.')
            dat = [np.array(psnap)[:,0] for psnap in data['position']]
            datalists.append(dat)
            continue
        if arg == 'y' and 'y' not in data.keys():
            if 'position' not in data.keys():
                raise KeyError(f'Neither y nore position was provided in data dictionary.')
            dat = [np.array(psnap)[:,1] for psnap in data['position']]
            datalists.append(dat)
            continue
        if arg == 'z' and 'z' not in data.keys():
            if 'position' not in data.keys():
                raise KeyError(f'Neither z nore position was provided in data dictionary.')
            dat = [np.array(psnap)[:,2] for psnap in data['position']]
            datalists.append(dat)
            continue
        if arg == 'vx' and 'vx' not in data.keys():
            if 'velocity' not in data.keys():
                raise KeyError(f'Neither vx nore velocity was provided in data dictionary.')
            dat = [np.array(vsnap)[:,0] for vsnap in data['velocity']]
            datalists.append(dat)
            continue
        if arg == 'vy' and 'vy' not in data.keys():
            if 'velocity' not in data.keys():
                raise KeyError(f'Neither vy nore velocity was provided in data dictionary.')
            dat = [np.array(vsnap)[:,1] for vsnap in data['velocity']]
            datalists.append(dat)
            continue
        if arg == 'vz' and 'vz' not in data.keys():
            if 'velocity' not in data.keys():
                raise KeyError(f'Neither vz nore velocity was provided in data dictionary.')
            dat = [np.array(vsnap)[:,2] for vsnap in data['velocity']]
            datalists.append(dat)
            continue
        if arg not in data.keys():
            raise KeyError(f'The key {arg} was not provided in data dictionary.')  
        datalists.append(data[arg])

    # print(len(datalists))
    # print(len(datalists[0]))
    # print(len(datalists[0][0]))
    # sys.exit()

    # set boundary string
    if 'boundary' not in data.keys():
        boundarystr == 'ITEM: BOX BOUNDS ff ff ff\n'
    else:
        boundarystr = f"ITEM: BOX BOUNDS {data['boundary'][0]} {data['boundary'][1]} {data['boundary'][2]}\n"

    # check if timesteps were provided
    timesteps_provided = False
    if 'timesteps' in data.keys():
        timesteps_provided = True
        timesteps = data['timesteps']

    # find number of atoms and check consistency
    nbp   = len(datalists[0][0])
    nsnap = len(datalists[0])
    for datalist in datalists:
        if len(datalist) != nsnap:
            raise ValueError(f'Inconsistent number of snapshots')
        for snap in datalist:
            if len(snap) != nbp:
                raise ValueError(f'Inconsistent number of atoms')
    

    # configure mode
    if append:
        mode = 'a'
    else:
        mode = 'w'

    with open(outfn,mode) as f:
        for snap in range(nsnap):
            # write header
            f.write('ITEM: TIMESTEP\n')
            if timesteps_provided:
                f.write(f'{timesteps[snap]}\n')  
            else:
                f.write(f'0\n')  
            f.write(f'ITEM: NUMBER OF ATOMS\n')
            f.write(f'{nbp}\n')
            f.write(boundarystr)
            f.write(f"{data['simulation_box'][0,0]} {data['simulation_box'][0,1]}\n")
            f.write(f"{data['simulation_box'][1,0]} {data['simulation_box'][1,1]}\n")
            f.write(f"{data['simulation_box'][2,0]} {data['simulation_box'][2,1]}\n")
            f.write(f"ITEM: ATOMS {' '.join(data['args'])}\n")

            for iatom in range(nbp):
                line = ''
                for iarg in range(len(datalists)):
                    line += f'{datalists[iarg][snap][iatom]} '
                line = line[:-1] + '\n'
                f.write(line)
                    



# if __name__ == "__main__":
    
#     if len(sys.argv) < 3:
#         print("usage: python %s fin fout"%sys.argv[0])
#         sys.exit(0)
#     fin  = sys.argv[1]
#     fout = sys.argv[2]
#     xyz  = load_xyz(fin)
#     # ~ fnpy = fin[:-4]+XYZ_NPY_EXT+'.npy'
#     # ~ save_xyz_binary(fnpy,xyz)
    
#     types = read_xyz_atomtypes(fin)
#     print(f'number of atoms = {len(types)}')
    
#     # ~ print(xyz.keys())
#     # ~ write_xyz(fout,xyz)
    
