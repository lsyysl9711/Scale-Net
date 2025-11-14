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


## Acknowledgments
We want to express our sincere thanks to the following works:

[POMO: Policy Optimization with Multiple Optima for Reinforcement Learning](https://github.com/yd-kwon/POMO)

[MVMoE: Multi-Task Vehicle Routing Solver with Mixture-of-Experts](https://github.com/RoyalSkye/Routing-MVMoE)

[Rethinking Light Decoder-based Solvers for Vehicle Routing Problems](https://github.com/ziweileonhuang/reld-nco)

## Citations

  <div style="position: relative">
    <button onclick="navigator.clipboard.writeText(document.getElementById('bibtex-cite').innerText)" style="position: absolute; top: 4px; right: 4px;"></button>
    <pre id="bibtex-cite"><code>
@inproceedings{liu2026scale,
  title     = {Scale-Net: A Hierarchical U-Net Framework for Cross-Scale Generalization in Multi-Task Vehicle Routing Problems},
  author    = {Suyu Liu and Zhiguang Cao and Nan Yin and Yew-Soon Ong},
  booktitle = {AAAI},
  year      = {2026}
}
    </code></pre>
  </div>

