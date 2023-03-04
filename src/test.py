
import os
import torch
from torchvision import transforms
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mim.commands.download import download
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def main():
    img_path = './photos/IMG_0004_512x512.JPG'
    """
    # 正規化のための変換を作成
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
    preprocess = transforms.Compose([
        transforms.Resize(512),
        transforms.CenterCrop(512),
        transforms.ToTensor(),
        # normalize
    ])
    
    # 画像を読み込んで前処理
    img = Image.open('./photos/IMG_0004_512x512.JPG')
    img_tensor = preprocess(img)
    #img_tensor = torch.from_numpy(np.array([img]))
    """
    
    """
    # モデルを読み込む
    config_file = './models/deeplabv3plus_r50-d8_512x512_20k_voc12aug.py'
    checkpoint_file = './models/deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth'
    model = init_segmentor(config_file, checkpoint_file, device='cpu')

    # 推論
    #pred_mask = model.predict(img_tensor)
    pred_mask = inference_segmentor(model, img_path)
    """
    os.makedirs('models', exist_ok=True)
    #checkpoint_name = 'deeplabv3plus_r101-d8_512x512_40k_voc12aug'
    checkpoint_name = 'deeplabv3plus_r50-d8_512x512_20k_voc12aug'
    config_fname = checkpoint_name + '.py'
    #checkpoint = download(package="mmsegmentation", configs=[checkpoint_name], dest_root="models")[0]
    checkpoint = 'deeplabv3plus_r50-d8_512x512_20k_voc12aug_20200617_102323-aad58ef1.pth'

    model = init_segmentor(os.path.join('models', config_fname), os.path.join('models', checkpoint), device = 'cpu')

    #result = inference_segmentor(model, frame)
    result = inference_segmentor(model, img_path)
    
    # 表示
    show_result_pyplot(model, img_path, result, get_palette('voc12aug'))


if __name__=='__main__':
    main()
