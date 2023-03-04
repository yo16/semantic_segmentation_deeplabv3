# semantic_segmentation_deeplabv3
DeepLabv3やほかのモデルを使用したSemantic Segmentation

# 概要
[Open MM Lab](https://openmmlab.com/)がgithubで提供しているライブラリ（[OpenMMLab](https://github.com/open-mmlab)）によって、様々なモデルを切り替えて試すことができるので、そのコード。

# 環境
下記で動作確認済み。
- python=3.10
- cpuで動作（gpuではなく）
- condaで、以下をpip install（conda installではなく）
  - torch torchvision torchaudio pillow opencv-python mmcv-full mmsegmentation openmim matplotlib

