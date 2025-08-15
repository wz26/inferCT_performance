# inferCT_performance

## Commands to test the code, please change the data and model path as needed

### Original code: 
```shell
# 1,024 synthetic case:  
python fast_foam_infer_ori_bench.py -nproc 1 -ih5 /vast/users/wzheng/3DN2I_models_data/data/foam_1800P_50A_500I.h5\
       -mdl /vast/users/wzheng/3DN2I_models_data/models/unet_4096.pth
# 4,096 synthetic case:  
python fast_foam_infer_ori_bench.py -nproc 1 -tiff_dir /vast/users/wzheng/3DN2I_models_data/data/4096_noisy\
       -mdl /vast/users/wzheng/3DN2I_models_data/models/unet_4096.pth 
```

### Pipeline only code: 
```shell
# 1,024 synthetic case:  
python fast_foam_infer_pipeline_only.py -nproc 1 -ih5 /vast/users/wzheng/3DN2I_models_data/data/foam_1800P_50A_500I.h5\
       -mdl /vast/users/wzheng/3DN2I_models_data/models/unet_4096.pth
# 4,096 synthetic case:  
python fast_foam_infer_pipeline_only.py -nproc 1 -tiff_dir /vast/users/wzheng/3DN2I_models_data/data/4096_noisy\
       -mdl /vast/users/wzheng/3DN2I_models_data/models/unet_4096.pth 
```


### Optimized code: 
```shell
# 1,024 synthetic case:  
python fast_foam_infer_optimized.py -nproc 1 -ih5 /vast/users/wzheng/3DN2I_models_data/data/foam_1800P_50A_500I.h5\
       -mdl /vast/users/wzheng/3DN2I_models_data/models/unet_4096.pth  -input_size 1024
# 4,096 synthetic case:  
python fast_foam_infer_optimized.py -nproc 1 -tiff_dir /vast/users/wzheng/3DN2I_models_data/data/4096_noisy\
       -mdl /vast/users/wzheng/3DN2I_models_data/models/unet_4096.pth  -input_size 4096
```
