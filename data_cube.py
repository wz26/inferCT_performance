# class for the data loader and operations (cubify and stitch)
# sequential data loader

import sys, os, h5py
import numpy as np
import torch
from torch.utils.data import Dataset 
import time, logging

# tiffs is new class, not same as tiff
import tiffs  

def cubify(volume, cube_size):

    D, H, W = volume.shape

    # Compute how much each dimension needs to be padded
    def next_multiple(x, size):
        return ((x + size - 1) // size) * size  # Round up to nearest multiple

    D_pad = next_multiple(D, cube_size)
    H_pad = next_multiple(H, cube_size)
    W_pad = next_multiple(W, cube_size)

    # Amount of padding needed for each dimension
    pad_d = D_pad - D
    pad_h = H_pad - H
    pad_w = W_pad - W

    # Pad the volume using 'edge' values
    # Format: ((beforeD, afterD), (beforeH, afterH), (beforeW, afterW))

    #we may need to change the 'edge'
    volume_padded = np.pad(
        volume,
        pad_width=((0, pad_d), (0, pad_h), (0, pad_w)),
        mode='edge'
    )

    # Now the shape is (D_pad, H_pad, W_pad)
    # We can split normally as in the original function
    nD = D_pad // cube_size
    nH = H_pad // cube_size
    nW = W_pad // cube_size

    # Reshape to group data into sub-blocks
    reshaped = volume_padded.reshape(nD, cube_size, nH, cube_size, nW, cube_size)

    # Transpose to make sub-block dimensions contiguous
    transposed = reshaped.transpose(0, 2, 4, 1, 3, 5)

    # Flatten the block indices into a single dimension
    cubes = transposed.reshape(-1, cube_size, cube_size, cube_size)

    return cubes, (D_pad, H_pad, W_pad)

def stitch(cubes, original_shape, padded_shape, cube_size):

    D, H, W = original_shape
    D_pad, H_pad, W_pad = padded_shape

    # Number of cubes along each dimension in the padded volume
    nD = D_pad // cube_size
    nH = H_pad // cube_size
    nW = W_pad // cube_size

    # Reshape from (N, c, c, c) -> (nD, nH, nW, c, c, c)
    reshaped = cubes.reshape(nD, nH, nW, cube_size, cube_size, cube_size)

    # Undo the transpose performed during splitting
    # Forward transpose was (0, 2, 4, 1, 3, 5), so the inverse is (0, 3, 1, 4, 2, 5)
    transposed = reshaped.transpose(0, 3, 1, 4, 2, 5)

    # Reshape to the padded 3D volume shape (D_pad, H_pad, W_pad)
    padded_volume = transposed.reshape(D_pad, H_pad, W_pad)

    # Slice off the padding to return to the original shape
    volume = padded_volume[:D, :H, :W]

    return volume


class TomoDataset3DCube(Dataset):
    def __init__(self, ih5, cube_size):
        super(TomoDataset3DCube, self).__init__()

        with h5py.File(ih5, 'r') as fp:
            self.split0 = fp['split0'][:].astype(np.float32)[:]
            self.split1 = fp['split1'][:].astype(np.float32)[:]

        self.original_shape = self.split0.shape
        self.cube_size = cube_size


        self.split0, _ = cubify(self.split0, cube_size=cube_size)
        self.split1, _ = cubify(self.split1, cube_size=cube_size)

        self.samples = self.__len__()

    def __getitem__(self, idx):


        view1, view2 = self.split0[idx, np.newaxis], self.split1[idx, np.newaxis]

        return view1, view2
    
    def __len__(self):
        return self.split0.shape[0]
    

# sequential h5 data loader used in the code
class TomoInferDatasetFoam3DCube_h5(Dataset):
    def __init__(self, ih5, cube_size):
        super(TomoInferDatasetFoam3DCube_h5, self).__init__()

        # measure this time
        load_time, cubify_time = 0, 0
        start_time = time.time()

        with h5py.File(ih5, 'r') as fp:
            self.noise = np.asarray(fp['noise'], dtype=np.float32)
            self.target = np.asarray(fp['ground_truth'], dtype=np.float32)

        load_time += time.time()-start_time
        logging.info(f"\nLoading H5 from Disk Time is: {load_time:.4f} seconds\n")

        self.original_shape = self.noise.shape
        self.cube_size = cube_size

        start_time = time.time()
        self.noise, self.padded_shape = cubify(self.noise, cube_size=cube_size)
        self.target, _ = cubify(self.target, cube_size=cube_size)
        cubify_time += time.time()-start_time
        logging.info(f"\nCubify time is: {cubify_time:.4f} seconds\n")

        self.samples = self.__len__()

    def stitch(self, cubes):

        stitched_volume = stitch(cubes, self.original_shape, self.padded_shape, cube_size=self.cube_size)

        return stitched_volume
    
    def __getitem__(self, idx):


        inp = self.noise[idx, np.newaxis]
        tgt = self.target[idx, np.newaxis]

        return inp, tgt
    
    def __len__(self):
        return self.noise.shape[0]



# sequential tiff data loader used in the code
class TomoInferDatasetFoam3DCube_tiff(Dataset):
    def __init__(self, path_to_tiffs_dir, cube_size, mean, std):
        super(TomoInferDatasetFoam3DCube_tiff, self).__init__()

        # path_to_tiffs_dir = path_to_tiffs_dir #+ '/'
        # tiffs_collection = tiffs.glob(path_to_tiffs_dir)
        # self.original = tiffs.load_stack(tiffs_collection)
        # self.original = self.original.astype(np.float32)

         # measure this time
        logging.info(f"\nstart to load \n")
        load_time, cubify_time = 0, 0
        start_time = time.time()

        path_to_tiffs_dir = path_to_tiffs_dir #+ '/'
        all_files = tiffs.glob(path_to_tiffs_dir)
        total_files = len(all_files)

        self.original = tiffs.load_stack(all_files)
        self.original = self.original.astype(np.float32)

        load_time += time.time()-start_time
        logging.info(f"\nLoading TIFF from Disk Time is: {load_time:.4f} seconds\n")

        self.original_shape = self.original.shape
        self.cube_size = cube_size

        start_time = time.time()

        self.original = ((self.original - mean) / (std)).astype(np.float32)
        # logging.info(f"\nData is scaled with provided mean: {mean}, std: {std}")
        self.original, self.padded_shape = cubify(self.original, cube_size=cube_size)

        cubify_time += time.time()-start_time
        logging.info(f"\nCubify time is: {cubify_time:.4f} seconds\n")

        self.samples = self.__len__()

    def stitch(self, cubes):

        stitched_volume = stitch(cubes, self.original_shape, self.padded_shape, cube_size=self.cube_size)

        return stitched_volume
    
    def __getitem__(self, idx):
    
        inp = self.original[idx, np.newaxis]

        return inp
    
    def __len__(self):
        return self.original.shape[0]