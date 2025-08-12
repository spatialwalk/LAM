# Copyright (c) 2024-2025, The Alibaba 3DAIGC Team Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import cv2
import sys
import base64
import subprocess

import numpy as np
from PIL import Image
import argparse
from omegaconf import OmegaConf

import torch
import zipfile
from glob import glob
import moviepy.editor as mpy
from lam.utils.ffmpeg_utils import images_to_video
from tools.flame_tracking_single_image import FlameTrackingSingleImage
# from gradio_gaussian_render import gaussian_render
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image
from tqdm import tqdm

try:
    import spaces
except:
    pass

h5_rendering = False  # True


def launch_env_not_compile_with_cuda():
    os.system('pip install chumpy')
    os.system('pip install numpy==1.23.0')
    os.system(
        'pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu121_pyt251/download.html'
    )


def assert_input_image(input_image):
    if input_image is None:
        raise gr.Error('No image selected or uploaded!')


def prepare_working_dir():
    import tempfile
    working_dir = tempfile.TemporaryDirectory()
    return working_dir


def init_preprocessor():
    from lam.utils.preprocess import Preprocessor
    global preprocessor
    preprocessor = Preprocessor()


def preprocess_fn(image_in: np.ndarray, remove_bg: bool, recenter: bool,
                  working_dir):
    image_raw = os.path.join(working_dir.name, 'raw.png')
    with Image.fromarray(image_in) as img:
        img.save(image_raw)
    image_out = os.path.join(working_dir.name, 'rembg.png')
    success = preprocessor.preprocess(image_path=image_raw,
                                      save_path=image_out,
                                      rmbg=remove_bg,
                                      recenter=recenter)
    assert success, f'Failed under preprocess_fn!'
    return image_out


def get_image_base64(path):
    with open(path, 'rb') as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode()
    return f'data:image/png;base64,{encoded_string}'


def doRender():
    print('do render')


def save_images2video(img_lst, v_pth, fps):
    from moviepy.editor import ImageSequenceClip
    # Ensure all images are in uint8 format
    images = [image.astype(np.uint8) for image in img_lst]

    # Create an ImageSequenceClip from the list of images
    clip = ImageSequenceClip(images, fps=fps)

    # Write the clip to a video file
    clip.write_videofile(v_pth, codec='libx264')

    print(f"Video saved successfully at {v_pth}")


def add_audio_to_video(video_path, out_path, audio_path):
    # Import necessary modules from moviepy
    from moviepy.editor import VideoFileClip, AudioFileClip

    # Load video file into VideoFileClip object
    video_clip = VideoFileClip(video_path)
    video_duration = video_clip.duration

    # Load audio file into AudioFileClip object
    audio_clip = AudioFileClip(audio_path)
    audio_duration = audio_clip.duration
    
    print(f"视频时长: {video_duration:.2f}秒, 音频时长: {audio_duration:.2f}秒")
    
    # 根据视频长度调整音频，确保视频长度不受音频影响
    if audio_duration > video_duration:
        # 如果音频比视频长，截断音频到视频长度
        audio_clip = audio_clip.subclip(0, video_duration)
        print(f"音频过长，截断到 {video_duration:.2f}秒")
    elif audio_duration < video_duration:
        # 如果音频比视频短，循环播放音频直到视频长度
        loops_needed = int(video_duration / audio_duration) + 1
        audio_clip = audio_clip.loop(duration=video_duration)
        print(f"音频过短，循环播放到 {video_duration:.2f}秒")
    
    # 确保音频长度完全匹配视频长度
    audio_clip = audio_clip.set_duration(video_duration)

    # Attach audio clip to video clip (replaces existing audio)
    video_clip_with_audio = video_clip.set_audio(audio_clip)

    # Export final video with audio using standard codecs
    # 显式设置duration确保输出视频长度与原视频一致
    video_clip_with_audio.write_videofile(
        out_path, 
        codec='libx264', 
        audio_codec='aac',
        temp_audiofile='temp-audio.m4a',
        remove_temp=True
    )

    print(f"✅ 音频添加成功，最终视频时长: {video_duration:.2f}秒，保存至: {out_path}")


def parse_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # parse from ENV
    if os.environ.get("APP_INFER") is not None:
        args.infer = os.environ.get("APP_INFER")
    if os.environ.get("APP_MODEL_NAME") is not None:
        cli_cfg.model_name = os.environ.get("APP_MODEL_NAME")

    args.config = args.infer if args.config is None else args.config

    if args.config is not None:
        cfg_train = OmegaConf.load(args.config)
        cfg.source_size = cfg_train.dataset.source_image_res
        try:
            cfg.src_head_size = cfg_train.dataset.src_head_size
        except:
            cfg.src_head_size = 112
        cfg.render_size = cfg_train.dataset.render_image.high
        _relative_path = os.path.join(
            cfg_train.experiment.parent,
            cfg_train.experiment.child,
            os.path.basename(cli_cfg.model_name).split("_")[-1],
        )

        cfg.save_tmp_dump = os.path.join("exps", "save_tmp", _relative_path)
        cfg.image_dump = os.path.join("exps", "images", _relative_path)
        cfg.video_dump = os.path.join("exps", "videos", _relative_path)  # output path

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)
        cfg.setdefault(
            "save_tmp_dump", os.path.join("exps", cli_cfg.model_name, "save_tmp")
        )
        cfg.setdefault("image_dump", os.path.join("exps", cli_cfg.model_name, "images"))
        cfg.setdefault(
            "video_dump", os.path.join("dumps", cli_cfg.model_name, "videos")
        )
        cfg.setdefault("mesh_dump", os.path.join("dumps", cli_cfg.model_name, "meshes"))

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)

    cfg.setdefault("logger", "INFO")

    assert cfg.model_name is not None, "model_name is required"

    return cfg, cfg_train


def create_zip_archive(output_zip='runtime_/h5_render_data.zip', base_vid="nice"):
    flame_params_pth = os.path.join("./assets/sample_motion/export", base_vid, "flame_params.json")
    file_lst = [
        'runtime_data/lbs_weight_20k.json', 'runtime_data/offset.ply', 'runtime_data/skin.glb',
        'runtime_data/vertex_order.json', 'runtime_data/bone_tree.json',
        flame_params_pth
    ]
    try:
        # Create a new ZIP file in write mode
        with zipfile.ZipFile(output_zip, 'w') as zipf:
            # List all files in the specified directory
            for file_path in file_lst:
                zipf.write(file_path, arcname=os.path.join("h5_render_data", os.path.basename(file_path)))
        print(f"Archive created successfully: {output_zip}")
    except Exception as e:
        print(f"An error occurred: {e}")


def make_yaw_rotation_matrix(yaw_deg, device='cpu', dtype=torch.float32):
    """
    绕世界坐标系Y轴旋转的4x4齐次矩阵，yaw>0为向右看
    """
    theta = np.deg2rad(yaw_deg)
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([
        [c, 0.0, s, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [-s, 0.0, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)


def make_pitch_rotation_matrix(pitch_deg, device='cpu', dtype=torch.float32):
    theta = np.deg2rad(pitch_deg)
    c, s = np.cos(theta), np.sin(theta)
    return torch.tensor([
        [1.0, 0.0, 0.0, 0.0],
        [0.0, c, -s, 0.0],
        [0.0, s, c, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ], device=device, dtype=dtype)


def demo_lam(flametracking, lam, cfg, yaw_deg=0.0, pitch_deg=0.0):
    def core_fn(image_path: str, video_params, working_dir):
        image_raw = os.path.join(working_dir.name, "raw.png")
        with Image.open(image_path).convert('RGB') as img:
            img.save(image_raw)

        base_vid = os.path.basename(video_params).split(".")[0]
        flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        base_iid = os.path.basename(image_path).split('.')[0]
        image_path = os.path.join("./assets/sample_input", base_iid, "images/00000_00.png")

        dump_video_path = os.path.join(working_dir.name, "output.mp4")
        dump_image_path = os.path.join(working_dir.name, "output.png")

        # prepare dump paths
        omit_prefix = os.path.dirname(image_raw)
        image_name = os.path.basename(image_raw)
        uid = image_name.split(".")[0]
        subdir_path = os.path.dirname(image_raw).replace(omit_prefix, "")
        subdir_path = (
            subdir_path[1:] if subdir_path.startswith("/") else subdir_path
        )
        print("subdir_path and uid:", subdir_path, uid)

        motion_seqs_dir = flame_params_dir

        dump_image_dir = os.path.dirname(dump_image_path)
        os.makedirs(dump_image_dir, exist_ok=True)

        print(image_raw, motion_seqs_dir, dump_image_dir, dump_video_path)

        dump_tmp_dir = dump_image_dir

        # if os.path.exists(dump_video_path):
        #     return dump_image_path, dump_video_path

        motion_img_need_mask = cfg.get("motion_img_need_mask", False)  # False
        vis_motion = cfg.get("vis_motion", False)  # False

        # preprocess input image: segmentation, flame params estimation
        return_code = flametracking.preprocess(image_raw)
        assert (return_code == 0), "flametracking preprocess failed!"
        return_code = flametracking.optimize()
        assert (return_code == 0), "flametracking optimize failed!"
        return_code, output_dir = flametracking.export()
        assert (return_code == 0), "flametracking export failed!"

        image_path = os.path.join(output_dir, "images/00000_00.png")
        mask_path = image_path.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
        print(image_path, mask_path)

        aspect_standard = 1.0 / 1.0
        source_size = cfg.source_size
        render_size = cfg.render_size
        render_fps = 30
        # prepare reference image
        image, _, _, shape_param = preprocess_image(image_path, mask_path=mask_path, intr=None, pad_ratio=0,
                                                    bg_color=1.,
                                                    max_tgt_size=None, aspect_standard=aspect_standard,
                                                    enlarge_ratio=[1.0, 1.0],
                                                    render_tgt_size=source_size, multiply=14, need_mask=True,
                                                    get_shape_param=True)

        # save masked image for vis
        save_ref_img_path = os.path.join(dump_tmp_dir, "output.png")
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(save_ref_img_path)

        # prepare motion seq
        src = image_path.split('/')[-3]
        driven = motion_seqs_dir.split('/')[-2]
        src_driven = [src, driven]
        motion_seq = prepare_motion_seqs(motion_seqs_dir, None, save_root=dump_tmp_dir, fps=render_fps,
                                         bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1, 0],
                                         render_image_res=render_size, multiply=16,
                                         need_mask=motion_img_need_mask, vis_motion=vis_motion,
                                         shape_param=shape_param, test_sample=False, cross_id=False,
                                         src_driven=src_driven)

        device, dtype = "cuda", torch.float32
        print('motion_seq["render_c2ws"] shape:', motion_seq["render_c2ws"].shape)
        print('motion_seq["render_c2ws"][0] shape:', motion_seq["render_c2ws"][0].shape)
        print('motion_seq["render_intrs"] shape:', motion_seq["render_intrs"].shape)
        print('motion_seq["render_bg_colors"] shape:', motion_seq["render_bg_colors"].shape)
        print('image shape:', image.shape)
        print('shape_param shape:', shape_param.shape)
        # 不做任何相机旋转，直接用原始render_c2ws
        # new_c2ws = motion_seq["render_c2ws"][0].to(device)
        # angle_tag = ''
        # dump_video_path = os.path.join(dump_image_dir, f'output{angle_tag}.mp4')
        # dump_image_path = os.path.join(dump_image_dir, f'output{angle_tag}.png')
        # save_ref_img_path = os.path.join(dump_tmp_dir, f'output{angle_tag}.png')
        # dump_video_path_wa = os.path.join(dump_image_dir, f'output{angle_tag}_audio.mp4')
        # print('不做相机旋转，输出文件名无角度后缀')
        # print('new_c2ws shape:', new_c2ws.shape)
        # print('new_c2ws.unsqueeze(0) shape:', new_c2ws.unsqueeze(0).shape)

        # === 同时应用俯仰和偏航 ===
        orig_c2w = motion_seq["render_c2ws"].to(device)  # 保持batch维 [1, N, 4, 4]
        pitch_mat = make_pitch_rotation_matrix(pitch_deg, device=device)
        yaw_mat = make_yaw_rotation_matrix(yaw_deg, device=device)
        R_total = yaw_mat @ pitch_mat  # 先俯仰再偏航
        R_batch = R_total.unsqueeze(0)  # [1, 4, 4]
        new_c2ws = torch.matmul(R_batch, orig_c2w)  # [1, N, 4, 4]
        angle_tag = f'_yaw{int(yaw_deg)}_pitch{int(pitch_deg)}'
        dump_video_path = os.path.join(dump_image_dir, f'output{angle_tag}.mp4')
        dump_image_path = os.path.join(dump_image_dir, f'output{angle_tag}.png')
        save_ref_img_path = os.path.join(dump_tmp_dir, f'output{angle_tag}.png')
        dump_video_path_wa = os.path.join(dump_image_dir, f'output{angle_tag}_audio.mp4')
        print('已绕Y轴右转30°，输出文件名带角度后缀')
        print('new_c2ws shape:', new_c2ws.shape)
        print('new_c2ws.unsqueeze(0) shape:', new_c2ws.unsqueeze(0).shape)

        # 辅助定位黑屏
        print('render_bg_colors[0:3]:', motion_seq["render_bg_colors"][0][:3])
        # start inference
        num_views = motion_seq["render_c2ws"].shape[1]
        # betas只保持(1, D)，不扩展view维
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)  # (1, 300)
        print('num_views:', num_views)
        for k, v in motion_seq["flame_params"].items():
            print(f'{k} shape:', v.shape)
        print("start to inference...................")
        # ---------- 收集所有帧 ----------
        frames = []
        batch_size = 64
        for start in tqdm(range(0, num_views, batch_size), desc='推理进度'):
            end = min(start + batch_size, num_views)
            c2ws = new_c2ws[:, start:end, ...].to(device, dtype).contiguous()
            intrs = motion_seq["render_intrs"][:, start:end, ...].to(device, dtype).contiguous()
            bg_colors = motion_seq["render_bg_colors"][:, start:end, ...].to(device, dtype).contiguous()
            flame_params_batch = {}
            for k, v in motion_seq["flame_params"].items():
                if k == "betas":
                    chunk = v
                else:
                    chunk = v[:, start:end, ...]
                flame_params_batch[k] = chunk.to(device, dtype).contiguous()
            with torch.no_grad():
                # 将4维image扩展为5维 [B, N_ref, C, H, W]，其中N_ref=1
                image_5d = image.to(device, dtype).unsqueeze(1)  # [B, 1, C, H, W]
                res = lam.infer_single_view(
                    image_5d, None, None,
                    render_c2ws=c2ws,
                    render_intrs=intrs,
                    render_bg_colors=bg_colors,
                    flame_params=flame_params_batch,
                )
            rgb = res["comp_rgb"].cpu().numpy()
            print('res["comp_rgb"].shape:', rgb.shape)
            if rgb.ndim == 5:  # (1, m, 3, H, W)
                rgb = rgb[0].transpose(0, 2, 3, 1)
            elif rgb.ndim == 4 and rgb.shape[-1] == 3:  # (batch, H, W, 3)
                pass
            else:
                raise ValueError(f"Unexpected rgb shape: {rgb.shape}")
            rgb = (rgb * 255).astype(np.uint8)
            frames.extend(list(rgb))
        print('frames collected:', len(frames), 'shape of first frame:', frames[0].shape)
        # 保存第一帧为图片
        Image.fromarray(frames[0]).save(os.path.join(dump_image_dir, 'debug_first_frame.png'))
        # ---------- 写视频 ----------
        save_images2video(frames, dump_video_path, render_fps)
        
        # 使用从源视频提取的音频文件，而不是预设的音频
        extracted_audio_path = os.path.join("./assets/sample_motion/export", base_vid, base_vid + ".wav")
        fallback_audio_path = os.path.join("./assets/sample_motion/export", "Joe_Biden", "Joe_Biden.wav")
        
        # 检查提取的音频文件是否存在且不为空
        if os.path.exists(extracted_audio_path) and os.path.getsize(extracted_audio_path) > 0:
            audio_path = extracted_audio_path
            print(f"✅ 使用从源视频提取的音频: {audio_path}")
        elif os.path.exists(fallback_audio_path):
            audio_path = fallback_audio_path  
            print(f"⚠️  源视频音频不可用，使用占位符音频: {audio_path}")
        else:
            print("❌ 没有可用的音频文件，跳过音频添加")
            return dump_image_path, dump_video_path  # 返回无音频版本
        
        add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)

        return dump_image_path, dump_video_path_wa

    return core_fn


def _build_model(cfg):
    from lam.models import model_dict
    from lam.utils.hf_hub import wrap_model_hub

    hf_model_cls = wrap_model_hub(model_dict["lam"])
    model = hf_model_cls.from_pretrained(cfg.model_name)

    return model


def get_video_info(video_path):
    """
    获取视频文件基本信息
    
    Args:
        video_path: 视频文件路径
        
    Returns:
        dict: 包含视频信息的字典
    """
    try:
        from moviepy.editor import VideoFileClip
        clip = VideoFileClip(video_path)
        
        info = {
            'duration': clip.duration,
            'fps': clip.fps,
            'size': clip.size,
            'has_audio': clip.audio is not None,
            'format': os.path.splitext(video_path)[1].lower()
        }
        
        clip.close()
        return info
    except Exception as e:
        print(f"获取视频信息失败: {str(e)}")
        return None


def convert_video_format(input_video_path, output_video_path, target_fps=30):
    """
    转换视频格式并提取音频
    
    Args:
        input_video_path: 输入视频文件路径
        output_video_path: 输出MP4文件路径
        target_fps: 目标帧率
    
    Returns:
        tuple: (video_success, audio_path) 
    """
    try:
        from moviepy.editor import VideoFileClip
        print(f"🎬 正在转换视频格式: {input_video_path} -> {output_video_path}")
        
        # 确保输出目录存在
        os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
        
        # 加载视频
        clip = VideoFileClip(input_video_path)
        print(f"   原始视频信息: {clip.duration:.2f}秒, {clip.fps:.2f}FPS, {clip.size[0]}x{clip.size[1]}")
        
        # 转换为目标帧率的MP4，保持原始分辨率比例
        if clip.size[1] > 720:  # 如果高度超过720p，进行缩放
            clip_resized = clip.resize(height=720)
            print(f"   调整分辨率为: {clip_resized.size[0]}x{clip_resized.size[1]}")
        else:
            clip_resized = clip
        
        # 写入视频文件
        clip_resized.write_videofile(
            output_video_path, 
            codec='libx264', 
            fps=target_fps,
            audio_codec='aac',
            verbose=False,  # 减少输出信息
            logger=None
        )
        
        # 单独提取音频，确保高质量
        audio_path = output_video_path.replace('.mp4', '.wav')
        audio_success = False
        
        if clip.audio is not None:
            try:
                print(f"🎵 正在提取音频...")
                clip.audio.write_audiofile(
                    audio_path,
                    verbose=False,
                    logger=None
                )
                if os.path.exists(audio_path) and os.path.getsize(audio_path) > 0:
                    audio_size = os.path.getsize(audio_path) / 1024  # KB
                    print(f"✅ 音频提取成功: {audio_path} ({audio_size:.1f} KB)")
                    audio_success = True
                else:
                    print("⚠️  音频文件生成失败或为空")
                    audio_path = None
            except Exception as audio_error:
                print(f"⚠️  音频提取失败: {str(audio_error)}")
                audio_path = None
        else:
            print("⚠️  源视频没有音频轨道")
            audio_path = None
        
        # 清理资源
        if clip.audio:
            clip.audio.close()
        clip.close()
        if clip_resized != clip:
            clip_resized.close()
        
        video_size = os.path.getsize(output_video_path) / 1024 / 1024  # MB
        print(f"✅ 视频转换成功: {output_video_path} ({video_size:.1f} MB)")
        return True, audio_path
        
    except Exception as e:
        print(f"❌ 视频转换失败: {str(e)}")
        return False, None


def setup_drive_data(drive_name, source_video_path=None, max_frames=264):
    """
    设置和修复驱动数据，支持多种视频格式
    
    Args:
        drive_name: 驱动数据名称
        source_video_path: 源视频文件路径（支持.mp4/.mov/.avi/.mkv等格式）
        max_frames: 最大帧数
    """
    drive_dir = f'./assets/sample_motion/export/{drive_name}'
    os.makedirs(drive_dir, exist_ok=True)
    
    # 1. 处理视频文件
    if source_video_path and os.path.exists(source_video_path):
        target_mp4 = f'{drive_dir}/{drive_name}.mp4'
        target_audio = f'{drive_dir}/{drive_name}.wav'
        
        # 获取并显示视频信息
        print(f"📹 分析视频文件: {source_video_path}")
        video_info = get_video_info(source_video_path)
        if video_info:
            print(f"   格式: {video_info['format']}")
            print(f"   时长: {video_info['duration']:.2f}秒")
            print(f"   帧率: {video_info['fps']:.2f} FPS")
            print(f"   分辨率: {video_info['size'][0]}x{video_info['size'][1]}")
            print(f"   包含音频: {'是' if video_info['has_audio'] else '否'}")
        
        # 获取源文件扩展名
        source_ext = os.path.splitext(source_video_path)[1].lower()
        supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']
        
        if source_ext not in supported_formats:
            print(f"⚠️  警告: 不支持的视频格式 {source_ext}")
            print(f"支持的格式: {', '.join(supported_formats)}")
            return False
        
        # 检查是否需要转换
        need_convert = True
        if source_ext == '.mp4':
            # 如果是MP4格式，检查是否可以直接复制
            if not os.path.exists(target_mp4) or os.path.getsize(target_mp4) == 0:
                print(f"复制MP4文件: {source_video_path} -> {target_mp4}")
                import shutil
                shutil.copy2(source_video_path, target_mp4)
                need_convert = False
                
                # 尝试提取音频
                try:
                    from moviepy.editor import VideoFileClip
                    clip = VideoFileClip(source_video_path)
                    if clip.audio is not None:
                        clip.audio.write_audiofile(target_audio)
                        print(f"音频提取成功: {target_audio}")
                    clip.close()
                except:
                    print("⚠️  音频提取失败，将使用占位符音频")
            else:
                need_convert = False
        
        # 转换其他格式到MP4
        if need_convert and (not os.path.exists(target_mp4) or os.path.getsize(target_mp4) == 0):
            video_success, audio_path = convert_video_format(source_video_path, target_mp4)
            if not video_success:
                print(f"❌ 视频处理失败: {source_video_path}")
                return False
    
    # 2. 修复transforms.json文件，限制帧数
    transforms_json = f'{drive_dir}/transforms.json'
    if os.path.exists(transforms_json):
        import json
        with open(transforms_json, 'r') as f:
            data = json.load(f)
        
        original_frames = len(data['frames'])
        if original_frames > max_frames:
            print(f"修复transforms.json: {original_frames}帧 -> {max_frames}帧")
            data['frames'] = data['frames'][:max_frames]
            
            with open(transforms_json, 'w') as f:
                json.dump(data, f, indent=2)
    
    # 3. 检查FLAME参数文件数量
    flame_dir = f'{drive_dir}/flame_param'
    if os.path.exists(flame_dir):
        flame_files = len([f for f in os.listdir(flame_dir) if f.endswith('.npz')])
        print(f"FLAME参数文件数量: {flame_files}")
        
        if flame_files != max_frames:
            print(f"⚠️  警告: FLAME参数文件数量({flame_files})与预期帧数({max_frames})不匹配")
    
    # 4. 检查音频文件
    audio_file = f'{drive_dir}/{drive_name}.wav'
    if not os.path.exists(audio_file):
        print(f"⚠️  警告: 缺少音频文件 {audio_file}")
        # 可以从Joe_Biden复制一个作为占位符
        fallback_audio = './assets/sample_motion/export/Joe_Biden/Joe_Biden.wav'
        if os.path.exists(fallback_audio):
            print(f"使用占位符音频: {fallback_audio}")
            import shutil
            shutil.copy2(fallback_audio, audio_file)
        else:
            print("⚠️  警告: 找不到占位符音频文件，可能会影响最终视频输出")
    
    print(f"✅ 驱动数据设置完成: {drive_name}")
    return True


if __name__ == '__main__':
    # 设置必要的环境变量
    os.environ["APP_MODEL_NAME"] = "./exps/releases/lam/lam-20k/step_045500/"
    os.environ["APP_INFER"] = "./configs/inference/lam-20k-8gpu.yaml"
    
    # === 配置部分 ===
    """
    使用说明:
    1. 将您的视频文件路径设置到 source_video_path（支持.mp4/.mov/.avi/.mkv等格式）
    2. 如果使用VHAP转换的数据，drive_name保持为 'custom_jp_drive'
    3. 如果使用其他预设驱动，修改drive_name为: Joe_Biden, Taylor_Swift等
    4. 程序会自动转换视频格式为MP4并提取音频，修复数据格式
    """
    
    # 输入图片路径
    input_image_path = '/root/autodl-tmp/datasets/mono_jp/apple_jp_3/images_4/000000.jpg'
    
    # 驱动数据名称（在assets/sample_motion/export目录下）
    drive_name = 'custom_jp_drive'  # 可选: Joe_Biden, Taylor_Swift, custom_jp_drive等
    
    # 源视频文件路径（支持多种格式：.mp4/.mov/.avi/.mkv等）
    source_video_path = '/root/autodl-tmp/datasets/mono_jp/apple_jp_3.mov'  # 请修改为您的实际视频文件路径
    
    # 输出目录
    output_dir = './output_test/jp_custom/'
    
    # 自动生成的路径
    video_path = f'./assets/sample_motion/export/{drive_name}/{drive_name}.mp4'

    # === 全局旋转角度常量 ===
    # GLOBAL_YAW = 0  # 绕 Y 轴偏航，正值向右看
    # GLOBAL_PITCH = 0  # 绕 X 轴俯仰，正值抬头

    # === 多角度配置 ===
    YAW_ANGLES = [-30, -20,-10,0,10,20, 30]  # 偏航角度列表，正值向右看
    PITCH_ANGLES = [-10, -5, 0, 5, 10]  # 俯仰角度列表，正值抬头

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # === 设置驱动数据 ===
    print("=== 设置驱动数据 ===")
    setup_drive_data(
        drive_name=drive_name,
        source_video_path=source_video_path,
        max_frames=264
    )

    # 配置与模型加载
    cfg, _ = parse_configs()
    cfg.model_name = "/root/autodl-tmp/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500"
    # cfg.source_size = 128  # 强制减小输入分辨率
    # cfg.render_size = 128  # 强制减小渲染分辨率
    lam = _build_model(cfg)
    lam.to('cuda')
    flametracking = FlameTrackingSingleImage(
        output_dir='tracking_output',
        alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
        vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
        human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
        facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
        detect_iris_landmarks=True
    )

    # 直接用output_dir作为working_dir
    class DummyDir:
        def __init__(self, name):
            self.name = name

    working_dir = DummyDir(output_dir)

    # === 多角度视频生成 ===
    print("=== 开始多角度视频生成 ===")
    total_combinations = len(YAW_ANGLES) * len(PITCH_ANGLES)
    print(f"将生成 {total_combinations} 个不同角度的视频")
    print(f"Yaw角度: {YAW_ANGLES}")
    print(f"Pitch角度: {PITCH_ANGLES}")

    generated_videos = []
    for i, yaw in enumerate(YAW_ANGLES):
        for j, pitch in enumerate(PITCH_ANGLES):
            current_combo = i * len(PITCH_ANGLES) + j + 1
            print(f"\n=== 生成视频 {current_combo}/{total_combinations}: Yaw={yaw}°, Pitch={pitch}° ===")
            
            # 调用主推理流程
            core_fn = demo_lam(
                flametracking, lam, cfg,
                yaw_deg=yaw,
                pitch_deg=pitch
            )
            
            try:
                img_path, vid_path = core_fn(input_image_path, video_path, working_dir)
                print(f'✅ 处理后图片保存于: {img_path}')
                print(f'✅ 生成视频保存于: {vid_path}')
                generated_videos.append({
                    'yaw': yaw,
                    'pitch': pitch,
                    'video_path': vid_path,
                    'image_path': img_path
                })
            except Exception as e:
                print(f'❌ 生成失败 (Yaw={yaw}°, Pitch={pitch}°): {str(e)}')
                continue

    print(f"\n=== 多角度视频生成完成 ===")
    print(f"成功生成 {len(generated_videos)} 个视频:")
    for video_info in generated_videos:
        print(f"  Yaw={video_info['yaw']:3d}°, Pitch={video_info['pitch']:3d}° -> {video_info['video_path']}")