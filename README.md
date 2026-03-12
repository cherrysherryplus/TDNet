# <img src="figs/TDNet_logo.png" alt="TDNet Logo" width="30" height="30" style="vertical-align: middle; margin-left: 10px;"> TDNet: Degradation-aware Comprehensive Task Decomposition for Joint Rain and Haze Removal

#### News
- **Mar 12, 2026:** Results and pretrained weights on five benchmarks are available; Codes for train/test/evaluation are updated.
- **Jan 23, 2026:** Initialization of the repository.

<hr />

> **Abstract:** *This repository provides the official implementation of **TDNet**, a physically grounded restoration framework designed to address the coupled nature of joint rain and haze degradations. At the core of our method is the Degradation-aware Comprehensive Task Decomposition (**DCTD**) strategy, which leverages physics-informed inductive biases to both decouple the restoration process into manageable subtasks and guide the architectural design of specialized modules: (1) **IND** (Implicit Neural Deraining): Exploits the inherent low-pass filtering bias of implicit neural representations to naturally isolate and suppress high-frequency rain components. (2) **PAD** (Prior-adaptive Dehazing): Integrates physical scattering priors into the feature space to estimate atmospheric parameters and eliminate the low-frequency global haze effect. (3) **SRM** (Scene Restoration Module): Reconstructs high-fidelity image content from refined intermediate features by capturing long-range dependencies. Extensive experiments are conducted to demonstrate the efficiency of designs. **Seven** benchmarks are included for comparisons or ablations, including <u>RainCityscapes</u>, <u>RainCityscapes-pp</u>, <u>RainhazeSynscapes</u>, <u>Rain200H</u>, <u>RW2AH</u>, <u>SemiSIRR</u>, and <u>REAL-RAIN</u>. PSNR, SSIM, and LPIPS scores are measured for **reference-based** IQA. For **no-reference** IQA, we employ NIQE and CLIP-IQA.*
>

<p align="center">
  <figure>
  <img width="800" src="figs/framework.jpg">
  <figcaption>Overall pipeline of the proposed TDNet. DRB and DRP are the acronyms of "Degradation Removal Block" and "Degradation Removal Path"</figcaption>
  </figure>
</p>

---

## Installation
> More details can be found in the TXT or YAML files from the `envs` folder.

## Our results and pretrained models
> Results on RainhazeSynscapes, Raincityscapes, Raincityscapes-pp, Rain200H, and RW2AH are available at [**saved_images**](https://pan.baidu.com/s/1mLGH_Zm9lvj11Jn0lZ2keA?pwd=eggd). Corresponding pretrained weights on these five datasets are available at [**pretrained_ckpt**](https://pan.baidu.com/s/10j1iRyl9HWJS935SvJWSww?pwd=qw4g).

## Train/Test/Evaluation
> More details can be found in the `Derainhaze/run.sh` file

## Visualization

<p align="center">
  <figure>
  <img width="800" src="figs/rh.jpg">
  <figcaption>Visualizations on the RainhazeSynscapes benchmark (<b>Rain streaks + Rainy haze</b>)</figcaption>
  </figure>
</p>

<p align="center">
  <figure>
  <img width="800" src="figs/rcpp.jpg">
  <figcaption>Visualizations on the RainCityscapes-pp benchmark (<b>Rain streaks + Rainy haze + Raindrops</b>)</figcaption>
  </figure>
</p>

<!-- <p align="center">
  <figure>
  <img width="800" src="figs/rain200h.jpg">
  <figcaption>Visualizations on the Rain200H benchmark (<b>Rain only</b>)</figcaption>
  </figure>
</p>

<p align="center">
  <figure>
  <img width="800" src="figs/rw2ah.jpg">
  <figcaption>Visualizations on the RW2AH benchmark (<b>Haze only</b>)</figcaption>
  </figure>
</p> -->

<p align="center">
  <figure>
  <img width="800" src="figs/realworld.jpg">
  <figcaption>Visualizations on the real-world SemiSIRR benchmark (<b>Rain streaks + Rainy haze</b>)</figcaption>
  </figure>
</p>

<p align="center">
  <figure>
  <img width="800" src="figs/applications.jpg">
  <figcaption>Instance segmentation based on YOLOv11 on the RainCityscapes-pp benchmark. Odd columns are original inputs, and even columns are restored inputs (<b>Rain streaks + Rainy haze + Raindrops</b>)</figcaption>
  </figure>
</p>

## Acknowledgement
Our work is built upon the codebase of [SFNet](https://github.com/c-yn/SFNet), [TransMamba](https://github.com/sunshangquan/TransMamba), [Restormer](https://github.com/swz30/Restormer), and [C2PNet](https://github.com/YuZheng9/C2PNet), and we sincerely thank them for their contributions.