
import os
from mmseg.apis import inference_segmentor, init_segmentor, show_result_pyplot
from mmseg.core.evaluation import get_palette
from mim.commands.download import download
from mmcv.cnn.utils.sync_bn import revert_sync_batchnorm

def main():
    img_path = './photos/IMG_0004_512x512.JPG'
    #img_path = './photos/IMG_0004.JPG'
    
    device = 'cpu'
    
    model_dir = 'models'
    os.makedirs(model_dir, exist_ok=True)
    
    # モデル
    #checkpoint_name = 'deeplabv3plus_r101-d8_512x512_20k_voc12aug'
    #checkpoint_name = 'deeplabv3plus_r101-d8_512x512_40k_voc12aug'
    #checkpoint_name = 'deeplabv3plus_r50-d8_512x512_80k_ade20k'
    #checkpoint_name = 'knet_s3_fcn_r50-d8_8x2_512x512_adamw_80k_ade20k'
    #checkpoint_name = 'segformer_mit-b0_512x512_160k_ade20k'
    checkpoint_name = 'segmenter_vit-b_mask_8x1_512x512_160k_ade20k'
    
    config_fname = checkpoint_name + '.py'
    checkpoint = download(package="mmsegmentation", configs=[checkpoint_name], dest_root=model_dir)[0]
    
    model = init_segmentor(os.path.join('models', config_fname), os.path.join(model_dir, checkpoint), device = 'cpu')
    if device=='cpu':
        model = revert_sync_batchnorm(model)

    result = inference_segmentor(model, img_path)
    
    # 表示
    #show_result_pyplot(model, img_path, result, get_palette('voc12aug'))
    show_result_pyplot(model, img_path, result, get_palette('ade20k'))





if __name__=='__main__':
    main()
