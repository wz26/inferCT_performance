
import torch, argparse, os, time, sys, shutil, logging


from model import unet
from data_cube import TomoInferDatasetFoam3DCube_h5, TomoInferDatasetFoam3DCube_tiff
import numpy as np
from torch.utils.data import DataLoader
from matplotlib import pyplot as plt
import tifffile
import pandas as pd
#from utils import calc_std, sobel_sharpness, canny_sharpness, calc_inertia, calc_SSIM, calc_PSNR
from utils import save2img
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr


import torch.multiprocessing as mp  # for multiple processing


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# used to estimate the memory usage for the specific batch size, 
# TODO: create a batch size predictor
def mem_estimator(args, model):
    dummy_input1 = torch.randn(1, 1, 64, 64, 64).cuda()  # Modify as needed
    # Reset peak memory stats
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_input1)
        peak_memory_bytes = torch.cuda.max_memory_allocated()
        peak_memory_mb1 = peak_memory_bytes / (1024 ** 2)
    logging.info(f"Peak GPU memory usage during dummy 1 inference: {peak_memory_mb1:.2f} MB")
    
    dummy_input2 = torch.randn(1, 1, 128, 128, 128).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_input2)
        peak_memory_mb2 = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logging.info(f"Peak GPU memory usage during dummy 2 inference: {peak_memory_mb2:.2f} MB")

    dummy_input3 = torch.randn(2, 1, 64, 64, 64).cuda()
    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        _ = model(dummy_input3)
        peak_memory_mb3 = torch.cuda.max_memory_allocated() / (1024 ** 2)
    logging.info(f"Peak GPU memory usage during dummy 3 inference: {peak_memory_mb3:.2f} MB")

    
    ratio_cubic = args.cube_size * args.cube_size * args.cube_size / 262144
    ratio_cubic = ratio_cubic/8.0
    ratio_cubic = ratio_cubic * peak_memory_mb2/peak_memory_mb1

    ratio_batch = args.mbsz / 2.0
    ratio_batch = ratio_batch * peak_memory_mb3/peak_memory_mb1 
    
    logging.info(f"Peak GPU memory estimated is: {peak_memory_mb1*ratio_cubic*ratio_batch:.2f} MB")

    return peak_memory_mb1*ratio_cubic*ratio_batch

def evaluation(args, gts, preds, odir):
    ssim_vals, psnr_vals, rmse_vals = [], [], []
    #Calculate Test Metrics
    for i in range(gts.shape[0]):
        #SSIM Calc
        img1 = preds[i]
        img1 = (img1-img1.min()) / (img1.max()-img1.min()+1e-7)
        img2 = gts[i]
        img2 = (img2-img2.min()) / (img2.max()-img2.min()+1e-7)

        #Crop out region outside sample
        #l_x, l_y = img2.shape[0], img2.shape[1]
        #X, Y = np.ogrid[:l_x, :l_y]
        #outer_disk_mask = (X - l_x / 2) ** 2 + (Y - l_y / 2) ** 2 > (l_x / 2) ** 2
        #[~outer_disk_mask]


        dr = img2.max() - img2.min()
        ssim_vals.append(ssim(img1, img2, data_range=dr))

        #PSNR Calc
        dr = img2.max() - img2.min()+1e-7
        psnr_vals.append(psnr(img1, img2, data_range=dr))

        #RMSE Calc
        rmse = np.sqrt(((img1-img2)**2.).mean())
        rmse_vals.append(rmse)

        #Save a single image (runs faster)
        if i == 525:
            save2img(img1, f'{odir}/tiffs/pred/{i:05d}.tiff')

        #Save output
        #save2img(img1, f'{odir}/tiffs/pred/{i:05d}.tiff')
        #save2img(img2, f'{odir}/tiffs/gt/{i:05d}.tiff')
        #save2img(img1, f'{odir}/pngs/pred/{i:05d}.png')
        #save2img(img2, f'{odir}/pngs/gt/{i:05d}.png')


    ssim_vals = np.array(ssim_vals)
    psnr_vals = np.array(psnr_vals)
    rmse_vals = np.array(rmse_vals)

    out_path = '/home/beams/WZHENG/3DN2I/CSVs/3D'
    dosage_level = '50A_500I_Cube'
    
    results_df = pd.DataFrame({
        'SSIM': ssim_vals,
        'PSNR': psnr_vals,
        'RMSE': rmse_vals,
    })

    return results_df

# mains inference function
def inference(args, model, odir, rank, num_processes, preds, gts, temp_X_buffer, temp_Y_buffer, barrier):

    total_time = 0

    start_time = time.time()
    if rank == 0:
        if os.path.isdir(f'{odir}/tiffs'):
            shutil.rmtree(f'{odir}/tiffs')
            shutil.rmtree(f'{odir}/pngs')
        os.mkdir(f'{odir}/tiffs')
        os.mkdir(f'{odir}/tiffs/pred')
        os.mkdir(f'{odir}/tiffs/noise')
        os.mkdir(f'{odir}/tiffs/gt')
        os.mkdir(f'{odir}/pngs')
        os.mkdir(f'{odir}/pngs/pred')
        os.mkdir(f'{odir}/pngs/noise')
        os.mkdir(f'{odir}/pngs/gt')
    total_time += time.time() - start_time
    logging.info(f"\nFolder creating time is: {time.time()-start_time:.4f} seconds\n")
    logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")

    # create the dataset according to the rank 
    start_time = time.time()
    if args.ih5 != 'false':
        ds_test = TomoInferDatasetFoam3DCube_h5(ih5=args.ih5, cube_size=args.cube_size)
    else:
        ds_test = TomoInferDatasetFoam3DCube_tiff(path_to_tiffs_dir=args.tiff_dir, cube_size=args.cube_size, mean=args.s0_mean, std=args.s0_std)
    
    total_time += time.time() - start_time
    logging.info(f"\nData load Time phase 0: {time.time()-start_time:.4f} seconds\n")
    logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")


    start_time = time.time()
    dl_test = DataLoader(dataset=ds_test, batch_size=args.mbsz, shuffle=False, num_workers=0, drop_last=False, pin_memory=True)
    total_time += time.time() - start_time
    logging.info(f"\nData load Time phase 1: {time.time()-start_time:.4f} seconds\n")
    logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")

    
    start_time = time.perf_counter()

    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus  # dynamically assign GPU based on rank
    torch.cuda.set_device(gpu_id)
    model.eval()
    model.to(gpu_id)
    num_cubes = 0

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    setup_time = elapsed_time 
    total_time += setup_time
    logging.info(f"\nModel setup Time: {setup_time:.4f} seconds\n")
    logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")

    # below is an estimator of the memory usage, will have a batch size predictor for it

    # Example: adjust the dimensions to match your actual input
    start_time = time.perf_counter()

    mem_estimator(args, model)

    torch.cuda.synchronize()
    elapsed_time = time.perf_counter() - start_time
    estimator_time = elapsed_time 
    total_time += estimator_time

    logging.info(f"\nEstimator Time: {estimator_time:.4f} seconds\n")
    logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")

    torch.cuda.synchronize()
    inference_time_start = time.perf_counter()
   
    with torch.no_grad():
        actual_compute_time = 0
        two_device_time     = 0
        two_host_time       = 0
        post_process_time   = 0
        data_loading_time   = 0
        barrier_wait_time   = 0
        skipped_time        = 0
        unsq_time_time      = 0
        data_loader_create_time = 0
        local_buffer_time       = 0

        bytes_2host, bytes_2dev = 0, 0
        cube_index = 0

        torch.cuda.synchronize()
        start_time = time.perf_counter()
        data_iter = iter(dl_test)
        torch.cuda.synchronize()
        data_loader_create_time = time.perf_counter() - start_time

        for idx in range(len(dl_test)):
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            if args.ih5 != 'false':
                X, y = next(data_iter)
            else:
                X = next(data_iter)
            torch.cuda.synchronize()
            data_loading_time += time.perf_counter() - start_time

            skip_start = time.perf_counter()
            if idx % num_processes != rank:
                torch.cuda.synchronize()
                skipped_time += time.perf_counter() - skip_start
                continue

            skipped_time += time.perf_counter() - skip_start

            cube_index += 1
            bytes_2dev += X.nelement() * X.element_size()

            # Device transfer
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            X = X.to(gpu_id)
            torch.cuda.synchronize()
            two_device_time += time.perf_counter() - start_time 

            # Compute
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            output = model(X)
            torch.cuda.synchronize()
            actual_compute_time += time.perf_counter() - start_time 
            peak_memory_bytes = torch.cuda.max_memory_allocated()

            # Host transfer
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            output = output.cpu()
            torch.cuda.synchronize()
            two_host_time += time.perf_counter() - start_time 
            # bytes_2host += output.nelement() * output.element_size()

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            temp_X_buffer[rank] = output.squeeze(dim=1).numpy()
            # temp_Y_buffer[rank] = y.squeeze(dim=1)
            torch.cuda.synchronize()
            unsq_time_time += time.perf_counter() - start_time 

            # Barrier wait 1
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            barrier.wait()
            torch.cuda.synchronize()
            barrier_wait_time += time.perf_counter() - start_time

            torch.cuda.synchronize()
            start_time = time.perf_counter()
            if rank == 0:
                preds += temp_X_buffer
                # gts += temp_Y_buffer
            torch.cuda.synchronize()
            local_buffer_time += time.perf_counter() - start_time

            # Barrier wait 2
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            barrier.wait()
            torch.cuda.synchronize()
            barrier_wait_time += time.perf_counter() - start_time

            # Post processing (reset buffers)
            torch.cuda.synchronize()
            start_time = time.perf_counter()
            temp_X_buffer[:] = [None] * len(temp_X_buffer)
            # temp_Y_buffer[:] = [None] * len(temp_Y_buffer)
            torch.cuda.synchronize()
            post_process_time += time.perf_counter() - start_time 
            # torch.cuda.synchronize()
            # total_loop_time = time.perf_counter() - inference_time_start
        
        total_loop_time = time.perf_counter() - inference_time_start

        # Add everything (skipped_time intentionally not added here)
        total_time = (
            data_loading_time +
            two_device_time +
            actual_compute_time +
            two_host_time +
            barrier_wait_time +
            post_process_time +
            unsq_time_time + 
            data_loader_create_time +
            local_buffer_time
        )

        logging.info(f"\n[Whole Loop Time]            : {total_loop_time:.4f} seconds")
        logging.info(f"[Data Loader Create Time]     : {data_loader_create_time:.4f} seconds")
        logging.info(f"[Data Loading Time]          : {data_loading_time:.4f} seconds")
        logging.info(f"[2Device Time]               : {two_device_time:.4f} seconds")
        logging.info(f"[Kernel Compute Time]        : {actual_compute_time:.4f} seconds")
        logging.info(f"[2Host Time]                 : {two_host_time:.4f} seconds")
        logging.info(f"[Barrier Wait Time]          : {barrier_wait_time:.4f} seconds")
        logging.info(f"[Local buffer Time]          : {local_buffer_time:.4f} seconds")
        logging.info(f"[Post Processing Time]       : {post_process_time:.4f} seconds")
        logging.info(f"[Unsqueeze Time]             : {unsq_time_time:.4f} seconds")
        logging.info(f"[Total Accounted Time]       : {total_time:.4f} seconds")
        logging.info(f"[Skipped Time]               : {skipped_time:.4f} seconds (if measured)")
        logging.info(f"[# Cubes Processed by Rank]  : {cube_index}")

        
    if rank == 0:
        start_time = time.time()
        preds = np.concatenate(preds, axis=0)
        # gts = np.concatenate(gts, axis=0)
        total_time += time.time() - start_time
        logging.info(f"\nConcatenate Time: {time.time()-start_time:.4f} seconds\n")
        logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")


        start_time = time.time()
        preds = ds_test.stitch(preds)
        # gts = ds_test.stitch(gts)
        total_time += time.time() - start_time
        logging.info(f"\nStitch Time: {time.time()-start_time:.4f} seconds\n")
        logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")

        # logging.info(f'\nStitched output Shape: {preds.shape}')
 
    if rank == 0:
        start_time = time.time()
        # results_df = evaluation(args, gts, preds, odir)  
        # logging.info(results_df.describe())
        #results_df.to_csv(f'{out_path}/{dosage_level}_test_results_cubify.csv', columns=results_df.columns, index=False)
        total_time += time.time() - start_time
        logging.info(f"\nEvaluation Time: {time.time()-start_time:.4f} seconds\n")
        logging.info(f"\nInference function Time now is: {total_time:.4f} seconds\n")

    logging.info(f"\nTotal inference Time: {total_time:.4f} seconds\n")


def main(args, rank, num_processes, preds, gts, temp_X_buffer, temp_Y_buffer, barrier):

    total_time = 0
    init_time = time.time()

    num_gpus = torch.cuda.device_count()
    gpu_id = rank % num_gpus  # dynamically assign GPU based on rank

    # Set the current device for this process
    torch.cuda.set_device(gpu_id)

    itr_out_dir = args.expName + '-itrOut' 
    logging.basicConfig(filename=os.path.join(itr_out_dir, f'TomoGAN_Inference_b{args.mbsz}_process{num_processes}_rank{rank}.log'), \
                        level=logging.DEBUG,format='%(asctime)s %(levelname)s %(module)s: %(message)s')

    start_time = time.time()

    logging.info(f'Process {rank} running on GPU {gpu_id}')

    start_time = time.time()
    checkpoint = torch.load(args.mdl, map_location=torch.device('cpu'))
    total_time += time.time() - start_time
    logging.info(f"\nModel load Phase 0 Time: {time.time()-start_time:.4f} seconds\n")

    # #model = NestedUNet()
    start_time = time.time()
    model = unet(start_filter_size=8)
    total_time += time.time() - start_time
    logging.info(f"\nModel load Phase 1 Time: {time.time()-start_time:.4f} seconds\n")
    print(f"Number of model parameters: {count_parameters(model):,}")

    start_time = time.time()
    model.load_state_dict(checkpoint['model_state_dict'])
    total_time += time.time() - start_time
    logging.info(f"\nModel load Phase 2 Time: {time.time()-start_time:.4f} seconds\n")

    start_time = time.time()
    inference(args, model, itr_out_dir, rank, num_processes, preds, gts, temp_X_buffer, temp_Y_buffer, barrier)
    total_time += time.time() - start_time
    logging.info(f"\nInference Function Time: {time.time()-start_time:.4f} seconds\n")

    inference_time = time.time() - init_time
    logging.info(f"\nTotal Time: {total_time:.4f} seconds\n")
    logging.info(f"\nWall clock Time: {inference_time:.4f} seconds\n")


if __name__ == '__main__':

    #FOAM (Small 1024x1024x1024)
    #tiff_dir = '/home/beams/AYUNKER/APS/SC/3D/data/foam_tiffs'
    #s0_mean = 102.48
    #s0_std = 28.996351
    #mdl = '/home/beams/AYUNKER/APS/SC/3D/debug3DCube-itrOut/best_abs_model.pth'
    test_h5 = '/home/beams/WZHENG/3DN2I/data/FOAM/foam_1800P_50A_500I.h5'

    #Foam (Large 4096x4096x4096)
    tiff_dir = '/home/beams/AYUNKER/APS/3DN2I/reconstructions_1800/noisy'
    tiff_dir = '/vast/users/wzheng/noisy'
    s0_mean = 125.8205795288086
    s0_std = 21.531658172607422
    # mdl = '/home/beams/AYUNKER/APS/SC/3D/MODELS/FOAM/foam_val_4096_2.pth'
    mdl = '/home/beams/WZHENG/3DN2I/MODELS/FOAM/3D/unet_128_cubify.pth'
    mdl = '/vast/users/wzheng/3DN2I/MODELS/FOAM/3D/unet_128_cubify.pth'

    #run: python foam_infer_cube.py -cube_size=128 -mbsz=8

    parser = argparse.ArgumentParser(description='Inference with N2I')
    parser.add_argument('-nproc',type=int, default=2, help='Number of processes')
    parser.add_argument('-expName',type=str, default="test", help='Experiment name, will write log file and output to here')

    parser.add_argument('-ih5',type=str, default="false", help='input directory for h5 file')
    parser.add_argument('-tiff_dir',type=str, default="false", help='input directory for tiff file')

    parser.add_argument('-staging', type=bool, default=False, help='Use staging for the data loader')
    parser.add_argument('-staged_path', type=str, default='/local/wzheng/stage_data/tmp.h5', help='Path to staged dataset')
    
    parser.add_argument('-mdl',type=str, default=mdl, help='model path')
    parser.add_argument('-num_workers',type=int, default=0, help='number of workers to load data, default set to 0 since we have pipelined execution')

    # parser.add_argument('-input_size',type=int, default=1024, help='inference input size')
    parser.add_argument('-cube_size',type=int, default=128, help='inference cube size')
    parser.add_argument('-mbsz',type=int, default=8, help='inference batch size')

    parser.add_argument('-verbose',type=int, default=1, help='1:print to terminal; 0: redirect to file')

    # parameters for the tff version
    parser.add_argument('-s0_mean',type=int, default=s0_mean, help='split 0 mean')
    parser.add_argument('-s0_std',type=int, default=s0_std, help='split 0 std')

    args, unparsed = parser.parse_known_args()

    if len(unparsed) > 0:
        print('Unrecognized argument(s): \n%s \nProgram exiting ... ... ' % '\n'.join(unparsed))
        exit(0)

    itr_out_dir = args.expName + '-itrOut'
    if os.path.isdir(itr_out_dir): 
        shutil.rmtree(itr_out_dir)
    os.mkdir(itr_out_dir) # to save temp output

    # logging.basicConfig(filename=os.path.join(itr_out_dir, f'TomoGAN_Inference_b{args.mbsz}.log'), level=logging.DEBUG,\
    #                     format='%(asctime)s %(levelname)s %(module)s: %(message)s')
    
    logging.getLogger('matplotlib.font_manager').disabled = True
    logging.getLogger('matplotlib').setLevel(level=logging.CRITICAL)
    if args.verbose:
        logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))

    mp.set_start_method('spawn', force=True)

    num_processes = args.nproc
    barrier = mp.Barrier(num_processes)

    processes = []
    queue = mp.Queue()

    start_time = time.time()

    # create preds and gts to share across devices
    manager = mp.Manager()
    preds = manager.list()
    gts = manager.list()

    # two empty lists that are used to collect all processes' results during forward
    temp_X_buffer = manager.list([None] * num_processes)
    temp_Y_buffer = manager.list([None] * num_processes)
 
    # barrier used to keep track of all processes to finish the inference job
    barrier = mp.Barrier(num_processes)

    # Spawn processes
    for rank in range(num_processes):
        p = mp.Process(target=main, args=(args, rank, num_processes, preds, gts, temp_X_buffer, temp_Y_buffer, barrier))
        p.start()
        processes.append(p)

    # Join processes
    for p in processes:
        p.join()

    inference_time = time.time() - start_time
    logging.info(f"\nWall clock Time: {inference_time:.4f} seconds\n")
