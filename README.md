# Scale-Net
This is the code repo of the implementation for the AAAI 2026 paper:

*Scale-Net: A Hierarchical U-Net Framework for Cross-Scale Generalization in Multi-Task Vehicle Routing Problems*
![VD (1)](https://github.com/lsyysl9711/Scale-Net/blob/main/arch.png)


## How to Run the Codes

    Default Settings: --problem_size=50 --pomo_size=50 --gpu_id=2

1. To run Scale-POMO-MTL
   
       python train.py --problem=Train_ALL --model_type=MTL_ET
   
3. To run Scale-MVMoE
   
       python train.py --problem=Train_ALL --model_type=MOE
