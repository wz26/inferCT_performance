# class for the data loader and operations (cubify and stitch)
# support parallel loader

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

# need to add num_processes to handle the case when rank 0 only work on the stiching
def stitch(cubes, original_shape, padded_shape, cube_size, num_processes=1):

    D, H, W = original_shape
    D_pad, H_pad, W_pad = padded_shape
    D_pad *= num_processes        # this is to handle the case when rank 0 only work on the stiching

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


# parallel h5 data loader used in the code
class TomoInferDatasetFoam3DCube_h5(Dataset):
    def __init__(self, ih5, cube_size, rank, world_size):
        super(TomoInferDatasetFoam3DCube_h5, self).__init__()

        # measure this time
        load_time, cubify_time = 0, 0
        start_time = time.time()

        # parallel read of the h5 file 
        with h5py.File(ih5, 'r') as fp:
            noise_ds  = fp['noise']
            target_ds = fp['ground_truth']

            total_slices = noise_ds.shape[0]
            chunk        = total_slices // world_size

            start = rank * chunk
            # last rank picks up any leftovers
            end   = total_slices if rank == world_size - 1 else (start + chunk)

            # read only this slab
            self.noise  = noise_ds[start:end,  ...].astype(np.float32)
            self.target = target_ds[start:end, ...].astype(np.float32)

        load_time += time.time()-start_time
        logging.info(f"\nLoading H5 from Disk Time is: {load_time:.4f} seconds\n")

        self.original_shape = self.noise.shape
        self.cube_size = cube_size

        start_time = time.time()
        self.noise, self.padded_shape = cubify(self.noise, cube_size=cube_size)
        self.target, _ = cubify(self.target, cube_size=cube_size)
        self.num_processes = world_size
        self.rank = rank
        cubify_time += time.time()-start_time
        logging.info(f"\nCubify time is: {cubify_time:.4f} seconds\n")

        self.samples = self.__len__()

    def stitch(self, cubes):

        stitched_volume = stitch(cubes, self.original_shape, self.padded_shape, cube_size=self.cube_size, num_processes=self.num_processes)

        return stitched_volume
    
    def __getitem__(self, idx):


        inp = self.noise[idx, np.newaxis]
        tgt = self.target[idx, np.newaxis]

        return inp, tgt
    
    def __len__(self):
        return self.noise.shape[0]


# parallel tiff data loader used in the code
class TomoInferDatasetFoam3DCube_tiff(Dataset):
    def __init__(self, path_to_tiffs_dir, cube_size, mean, std, rank, world_size):
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

        # Split file list into chunks for each process
        chunk_size = (total_files + world_size - 1) // world_size
        chunks = [all_files[i:i + chunk_size] for i in range(0, total_files, chunk_size)]
        chunks = chunks[rank]

        self.original = tiffs.load_stack(chunks)
        self.original = self.original.astype(np.float32)

        load_time += time.time()-start_time
        logging.info(f"\nLoading TIFF from Disk Time is: {load_time:.4f} seconds\n")

        self.original_shape = self.original.shape
        self.cube_size = cube_size

        start_time = time.time()

        self.original = ((self.original - mean) / (std)).astype(np.float32)
        # logging.info(f"\nData is scaled with provided mean: {mean}, std: {std}")
        self.original, self.padded_shape = cubify(self.original, cube_size=cube_size)

        self.num_processes = world_size
        self.rank = rank

        cubify_time += time.time()-start_time
        logging.info(f"\nCubify time is: {cubify_time:.4f} seconds\n")

        self.samples = self.__len__()

    def stitch(self, cubes):

        stitched_volume = stitch(cubes, self.original_shape, self.padded_shape, cube_size=self.cube_size, num_processes=self.num_processes)

        return stitched_volume
    
    def __getitem__(self, idx):
    
        inp = self.original[idx, np.newaxis]

        return inp
    
    def __len__(self):
        return self.original.shape[0]


class TomoTiffTrain(Dataset):
    def __init__(self, split0_dir, split1_dir, cube_size, split0_mean, split0_std, split1_mean, split1_std, rank, world_size):
        super(TomoTiffTrain, self).__init__()

        tiffs_collection = tiffs.glob(split0_dir)
        self.split0 = tiffs.load_stack(tiffs_collection)
        self.split0 = self.split0.astype(np.float32)

        tiffs_collection = tiffs.glob(split1_dir)
        self.split1 = tiffs.load_stack(tiffs_collection)
        self.split1 = self.split1.astype(np.float32)


        self.split0_shape = self.split0.shape
        self.cube_size = cube_size

        split0_mean = self.split0.mean()
        split0_std= self.split0.std()
        split1_mean = self.split1.mean()
        split1_std= self.split1.std()

        self.split0 = ((self.split0 - split0_mean) / (split0_std)).astype(np.float32)
        logging.info(f"\nData is scaled with provided mean: {split0_mean}, std: {split0_std}")

        self.split1 = ((self.split1 - split1_mean) / (split1_std)).astype(np.float32)
        logging.info(f"\nData is scaled with provided mean: {split1_mean}, std: {split1_std}")

        self.split0, self.padded_shape = cubify(self.split0, cube_size=cube_size)

        self.split1, self.padded_shape = cubify(self.split1, cube_size=cube_size)

        self.samples = self.__len__()

    def stitch(self, cubes):

        stitched_volume = stitch(cubes, self.original_shape, self.padded_shape, cube_size=self.cube_size, num_processes=self.num_processes)

        return stitched_volume
    
    def __getitem__(self, idx):


        view1, view2 = self.split0[idx, np.newaxis], self.split1[idx, np.newaxis]

        return view1, view2
    
    def __len__(self):
        return self.split0.shape[0]






