#!/usr/bin/env python3
"""
整合的头像生成系统
结合了VHAP数据转换和多视角头像视频生成功能

功能特性：
1. 自动转换VHAP输出为LAM兼容格式
2. 支持多种视频格式输入处理
3. 生成多个不同视角的头像动画视频
4. 自动音频提取和同步
5. 完整的端到端工作流程

使用方法：
python integrated_avatar_generation.py --input_image path/to/image.jpg --source_video path/to/video.mp4
"""

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
from pathlib import Path
import json
import shutil
from tqdm import tqdm

# LAM相关导入
from lam.utils.ffmpeg_utils import images_to_video
from tools.flame_tracking_single_image import FlameTrackingSingleImage
from lam.runners.infer.head_utils import prepare_motion_seqs, preprocess_image

try:
    import spaces
except:
    pass

# 全局配置
h5_rendering = False

class IntegratedAvatarGenerator:
    """整合的头像生成器类"""
    
    def __init__(self, config):
        self.config = config
        self.lam_model = None
        self.flametracking = None
        self.preprocessor = None
        
    def init_preprocessor(self):
        """初始化预处理器"""
        from lam.utils.preprocess import Preprocessor
        self.preprocessor = Preprocessor()
        print("✅ 预处理器初始化完成")
        
    def init_flame_tracking(self):
        """初始化FLAME跟踪器"""
        self.flametracking = FlameTrackingSingleImage(
            output_dir='tracking_output',
            alignment_model_path='./model_zoo/flame_tracking_models/68_keypoints_model.pkl',
            vgghead_model_path='./model_zoo/flame_tracking_models/vgghead/vgg_heads_l.trcd',
            human_matting_path='./model_zoo/flame_tracking_models/matting/stylematte_synth.pt',
            facebox_model_path='./model_zoo/flame_tracking_models/FaceBoxesV2.pth',
            detect_iris_landmarks=True
        )
        print("✅ FLAME跟踪器初始化完成")
        
    def init_lam_model(self):
        """初始化LAM模型"""
        from lam.models import model_dict
        from lam.utils.hf_hub import wrap_model_hub
        
        hf_model_cls = wrap_model_hub(model_dict["lam"])
        self.lam_model = hf_model_cls.from_pretrained(self.config.model_name)
        self.lam_model.to('cuda')
        print("✅ LAM模型初始化完成")
        
    def get_video_info(self, video_path):
        """获取视频文件基本信息"""
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
            
    def convert_video_format(self, input_video_path, output_video_path, target_fps=30):
        """转换视频格式并提取音频"""
        try:
            from moviepy.editor import VideoFileClip
            print(f"🎬 正在转换视频格式: {input_video_path} -> {output_video_path}")
            
            os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
            
            clip = VideoFileClip(input_video_path)
            print(f"   原始视频信息: {clip.duration:.2f}秒, {clip.fps:.2f}FPS, {clip.size[0]}x{clip.size[1]}")
            
            if clip.size[1] > 720:
                clip_resized = clip.resize(height=720)
                print(f"   调整分辨率为: {clip_resized.size[0]}x{clip_resized.size[1]}")
            else:
                clip_resized = clip
            
            clip_resized.write_videofile(
                output_video_path, 
                codec='libx264', 
                fps=target_fps,
                audio_codec='aac',
                verbose=False,
                logger=None
            )
            
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
                        audio_size = os.path.getsize(audio_path) / 1024
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
            
            if clip.audio:
                clip.audio.close()
            clip.close()
            if clip_resized != clip:
                clip_resized.close()
            
            video_size = os.path.getsize(output_video_path) / 1024 / 1024
            print(f"✅ 视频转换成功: {output_video_path} ({video_size:.1f} MB)")
            return True, audio_path
            
        except Exception as e:
            print(f"❌ 视频转换失败: {str(e)}")
            return False, None
            
    def process_video_file(self, source_video_path, output_dir, target_name):
        """处理视频文件，复制/转换视频并提取音频"""
        if not source_video_path or not os.path.exists(source_video_path):
            print(f"⚠️  跳过视频处理：视频文件不存在 {source_video_path}")
            return False
        
        output_path = Path(output_dir)
        target_mp4 = output_path / f'{target_name}.mp4'
        target_audio = output_path / f'{target_name}.wav'
        
        print(f"\n📹 分析视频文件: {source_video_path}")
        video_info = self.get_video_info(source_video_path)
        if video_info:
            print(f"   格式: {video_info['format']}")
            print(f"   时长: {video_info['duration']:.2f}秒")
            print(f"   帧率: {video_info['fps']:.2f} FPS")
            print(f"   分辨率: {video_info['size'][0]}x{video_info['size'][1]}")
            print(f"   包含音频: {'是' if video_info['has_audio'] else '否'}")
        
        source_ext = os.path.splitext(source_video_path)[1].lower()
        supported_formats = ['.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm']
        
        if source_ext not in supported_formats:
            print(f"⚠️  警告: 不支持的视频格式 {source_ext}")
            print(f"支持的格式: {', '.join(supported_formats)}")
            return False
        
        need_convert = True
        if source_ext == '.mp4':
            print(f"📋 复制MP4文件: {source_video_path} -> {target_mp4}")
            shutil.copy2(source_video_path, target_mp4)
            need_convert = False
            
            try:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(source_video_path)
                if clip.audio is not None:
                    print(f"🎵 提取音频文件...")
                    clip.audio.write_audiofile(str(target_audio), verbose=False, logger=None)
                    print(f"✅ 音频提取成功: {target_audio}")
                else:
                    print("⚠️  源视频没有音频轨道")
                clip.close()
            except Exception as e:
                print(f"⚠️  音频提取失败: {str(e)}")
        
        if need_convert:
            video_success, audio_path = self.convert_video_format(source_video_path, str(target_mp4))
            if not video_success:
                print(f"❌ 视频处理失败: {source_video_path}")
                return False
        
        video_exists = target_mp4.exists() and target_mp4.stat().st_size > 0
        audio_exists = target_audio.exists() and target_audio.stat().st_size > 0
        
        print(f"📋 视频文件: {'✅' if video_exists else '❌'} {target_mp4}")
        print(f"🎵 音频文件: {'✅' if audio_exists else '❌'} {target_audio}")
        
        return video_exists
        
    def create_flame_params_json(self, flame_param_dir, output_dir, max_frames):
        """创建合并的flame_params.json文件"""
        all_params = {
            "expr": [],
            "rotation": [], 
            "neck_pose": [],
            "jaw_pose": [],
            "eyes_pose": [],
            "translation": [],
            "betas": None
        }
        
        for i in range(max_frames):
            flame_file = Path(flame_param_dir) / f"{i:05d}.npz"
            if flame_file.exists():
                data = np.load(flame_file)
                
                for key in ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation']:
                    if key in data:
                        if data[key].ndim == 2:
                            all_params[key].append(data[key][0].tolist())
                        else:
                            all_params[key].append(data[key].tolist())
                
                if all_params["betas"] is None and "betas" in data:
                    all_params["betas"] = data["betas"][0].tolist()
        
        json_file = Path(output_dir) / "flame_params.json"
        with open(json_file, 'w') as f:
            json.dump(all_params, f, indent=2)
            
    def convert_vhap_to_lam(self, vhap_dir, output_dir, max_frames=264, target_name="custom_drive", source_video_path=None):
        """转换VHAP输出为LAM格式"""
        vhap_path = Path(vhap_dir)
        output_path = Path(output_dir) / target_name
        output_path.mkdir(parents=True, exist_ok=True)
        
        print(f"🔄 转换VHAP数据：{vhap_path} -> {output_path}")
        
        # 1. 复制并转换flame_param文件
        flame_param_dir = output_path / "flame_param"
        flame_param_dir.mkdir(exist_ok=True)
        
        vhap_flame_dir = vhap_path / "flame_param"
        flame_files = sorted([f for f in os.listdir(vhap_flame_dir) if f.endswith('.npz')])
        
        flame_files = flame_files[:max_frames]
        print(f"🔥 处理{len(flame_files)}个FLAME参数文件...")
        
        for i, flame_file in enumerate(flame_files):
            vhap_data = np.load(vhap_flame_dir / flame_file)
            
            lam_data = {}
            
            for key in ['translation', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'expr']:
                if key in vhap_data:
                    lam_data[key] = vhap_data[key]
            
            if i == 0 and 'shape' in vhap_data:
                lam_data['betas'] = vhap_data['shape'].reshape(1, -1)
            
            output_file = flame_param_dir / f"{i:05d}.npz"
            np.savez(output_file, **lam_data)
        
        # 2. 复制canonical_flame_param.npz
        canonical_src = vhap_path / "canonical_flame_param.npz"
        canonical_dst = output_path / "canonical_flame_param.npz"
        if canonical_src.exists():
            shutil.copy2(canonical_src, canonical_dst)
            print("📋 复制canonical_flame_param.npz")
        
        # 3. 复制transforms相关文件
        for transform_file in ['transforms.json', 'transforms_train.json', 'transforms_test.json', 'transforms_val.json']:
            src_file = vhap_path / transform_file
            dst_file = output_path / transform_file
            if src_file.exists():
                shutil.copy2(src_file, dst_file)
                print(f"📋 复制{transform_file}")
        
        # 4. 创建flame_params.json
        print("📝 生成flame_params.json...")
        self.create_flame_params_json(flame_param_dir, output_path, max_frames)
        
        # 5. 处理视频文件
        if source_video_path:
            print(f"\n🎬 处理视频文件...")
            video_success = self.process_video_file(source_video_path, output_path, target_name)
            if video_success:
                print(f"✅ 视频处理完成")
            else:
                print(f"⚠️  视频处理失败")
        
        # 6. 最终状态检查
        print(f"\n📋 最终文件检查:")
        required_files = [
            (output_path / f"{target_name}.mp4", "驱动视频"),
            (output_path / f"{target_name}.wav", "音频文件"),
            (output_path / "flame_param", "FLAME参数目录"),
            (output_path / "flame_params.json", "合并的FLAME参数"),
            (output_path / "transforms.json", "变换矩阵")
        ]
        
        all_ready = True
        for file_path, description in required_files:
            exists = file_path.exists()
            status = "✅" if exists else "❌"
            print(f"   {status} {description}: {file_path}")
            if not exists:
                all_ready = False
        
        if all_ready:
            print(f"\n🎉 VHAP转换完成！所有必需文件已就绪")
            print(f"   驱动名称设置为: {target_name}")
        else:
            print(f"\n⚠️  VHAP转换完成，但部分文件缺失")
        
        return str(output_path), all_ready
        
    def make_yaw_rotation_matrix(self, yaw_deg, device='cpu', dtype=torch.float32):
        """绕世界坐标系Y轴旋转的4x4齐次矩阵"""
        theta = np.deg2rad(yaw_deg)
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [c, 0.0, s, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [-s, 0.0, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=device, dtype=dtype)
        
    def make_pitch_rotation_matrix(self, pitch_deg, device='cpu', dtype=torch.float32):
        """绕世界坐标系X轴旋转的4x4齐次矩阵"""
        theta = np.deg2rad(pitch_deg)
        c, s = np.cos(theta), np.sin(theta)
        return torch.tensor([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, c, -s, 0.0],
            [0.0, s, c, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], device=device, dtype=dtype)
        
    def save_images2video(self, img_lst, v_pth, fps):
        """保存图像列表为视频"""
        from moviepy.editor import ImageSequenceClip
        images = [image.astype(np.uint8) for image in img_lst]
        clip = ImageSequenceClip(images, fps=fps)
        clip.write_videofile(v_pth, codec='libx264')
        print(f"Video saved successfully at {v_pth}")
        
    def add_audio_to_video(self, video_path, out_path, audio_path):
        """为视频添加音频"""
        from moviepy.editor import VideoFileClip, AudioFileClip
        
        video_clip = VideoFileClip(video_path)
        video_duration = video_clip.duration
        
        audio_clip = AudioFileClip(audio_path)
        audio_duration = audio_clip.duration
        
        print(f"视频时长: {video_duration:.2f}秒, 音频时长: {audio_duration:.2f}秒")
        
        if audio_duration > video_duration:
            audio_clip = audio_clip.subclip(0, video_duration)
            print(f"音频过长，截断到 {video_duration:.2f}秒")
        elif audio_duration < video_duration:
            audio_clip = audio_clip.loop(duration=video_duration)
            print(f"音频过短，循环播放到 {video_duration:.2f}秒")
        
        audio_clip = audio_clip.set_duration(video_duration)
        video_clip_with_audio = video_clip.set_audio(audio_clip)
        
        video_clip_with_audio.write_videofile(
            out_path, 
            codec='libx264', 
            audio_codec='aac',
            temp_audiofile='temp-audio.m4a',
            remove_temp=True
        )
        
        print(f"✅ 音频添加成功，最终视频时长: {video_duration:.2f}秒，保存至: {out_path}")
        
    def generate_multiview_videos(self, image_path, drive_path, output_dir, yaw_angles, pitch_angles):
        """生成多视角头像视频"""
        print("=== 开始多视角视频生成 ===")
        
        class DummyDir:
            def __init__(self, name):
                self.name = name
        
        working_dir = DummyDir(output_dir)
        
        total_combinations = len(yaw_angles) * len(pitch_angles)
        print(f"将生成 {total_combinations} 个不同角度的视频")
        print(f"Yaw角度: {yaw_angles}")
        print(f"Pitch角度: {pitch_angles}")
        
        generated_videos = []
        
        for i, yaw in enumerate(yaw_angles):
            for j, pitch in enumerate(pitch_angles):
                current_combo = i * len(pitch_angles) + j + 1
                print(f"\n=== 生成视频 {current_combo}/{total_combinations}: Yaw={yaw}°, Pitch={pitch}° ===")
                
                try:
                    img_path, vid_path = self._generate_single_view(
                        image_path, drive_path, working_dir, yaw, pitch
                    )
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
        
        print(f"\n=== 多视角视频生成完成 ===")
        print(f"成功生成 {len(generated_videos)} 个视频:")
        for video_info in generated_videos:
            print(f"  Yaw={video_info['yaw']:3d}°, Pitch={video_info['pitch']:3d}° -> {video_info['video_path']}")
            
        return generated_videos
        
    def _generate_single_view(self, image_path, video_params, working_dir, yaw_deg=0.0, pitch_deg=0.0):
        """生成单个视角的视频"""
        image_raw = os.path.join(working_dir.name, "raw.png")
        with Image.open(image_path).convert('RGB') as img:
            img.save(image_raw)
        
        base_vid = os.path.basename(video_params).split(".")[0]
        flame_params_dir = os.path.join("./assets/sample_motion/export", base_vid, "flame_param")
        
        dump_image_dir = working_dir.name
        os.makedirs(dump_image_dir, exist_ok=True)
        
        angle_tag = f'_yaw{int(yaw_deg)}_pitch{int(pitch_deg)}'
        dump_video_path = os.path.join(dump_image_dir, f'output{angle_tag}.mp4')
        dump_image_path = os.path.join(dump_image_dir, f'output{angle_tag}.png')
        dump_video_path_wa = os.path.join(dump_image_dir, f'output{angle_tag}_audio.mp4')
        
        # 预处理输入图像
        return_code = self.flametracking.preprocess(image_raw)
        assert (return_code == 0), "flametracking preprocess failed!"
        return_code = self.flametracking.optimize()
        assert (return_code == 0), "flametracking optimize failed!"
        return_code, output_dir = self.flametracking.export()
        assert (return_code == 0), "flametracking export failed!"
        
        image_path_processed = os.path.join(output_dir, "images/00000_00.png")
        mask_path = image_path_processed.replace("/images/", "/fg_masks/").replace(".jpg", ".png")
        
        aspect_standard = 1.0 / 1.0
        source_size = self.config.source_size
        render_size = self.config.render_size
        render_fps = 30
        
        # 准备参考图像
        image, _, _, shape_param = preprocess_image(
            image_path_processed, mask_path=mask_path, intr=None, pad_ratio=0,
            bg_color=1., max_tgt_size=None, aspect_standard=aspect_standard,
            enlarge_ratio=[1.0, 1.0], render_tgt_size=source_size, multiply=14, 
            need_mask=True, get_shape_param=True
        )
        
        # 保存处理后的图像
        vis_ref_img = (image[0].permute(1, 2, 0).cpu().detach().numpy() * 255).astype(np.uint8)
        Image.fromarray(vis_ref_img).save(dump_image_path)
        
        # 准备运动序列
        src = image_path.split('/')[-3] if len(image_path.split('/')) > 3 else 'default'
        driven = flame_params_dir.split('/')[-2]
        src_driven = [src, driven]
        
        motion_seq = prepare_motion_seqs(
            flame_params_dir, None, save_root=dump_image_dir, fps=render_fps,
            bg_color=1., aspect_standard=aspect_standard, enlarge_ratio=[1.0, 1.0],
            render_image_res=render_size, multiply=16, need_mask=False, vis_motion=False,
            shape_param=shape_param, test_sample=False, cross_id=False,
            src_driven=src_driven
        )
        
        device, dtype = "cuda", torch.float32
        
        # 应用旋转变换
        orig_c2w = motion_seq["render_c2ws"].to(device)
        pitch_mat = self.make_pitch_rotation_matrix(pitch_deg, device=device)
        yaw_mat = self.make_yaw_rotation_matrix(yaw_deg, device=device)
        R_total = yaw_mat @ pitch_mat
        R_batch = R_total.unsqueeze(0)
        new_c2ws = torch.matmul(R_batch, orig_c2w)
        
        # 开始推理
        num_views = motion_seq["render_c2ws"].shape[1]
        motion_seq["flame_params"]["betas"] = shape_param.unsqueeze(0)
        
        print("开始推理...")
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
                image_5d = image.to(device, dtype).unsqueeze(1)
                res = self.lam_model.infer_single_view(
                    image_5d, None, None,
                    render_c2ws=c2ws,
                    render_intrs=intrs,
                    render_bg_colors=bg_colors,
                    flame_params=flame_params_batch,
                )
            
            rgb = res["comp_rgb"].cpu().numpy()
            if rgb.ndim == 5:
                rgb = rgb[0].transpose(0, 2, 3, 1)
            elif rgb.ndim == 4 and rgb.shape[-1] == 3:
                pass
            else:
                raise ValueError(f"Unexpected rgb shape: {rgb.shape}")
            
            rgb = (rgb * 255).astype(np.uint8)
            frames.extend(list(rgb))
        
        print(f'收集到 {len(frames)} 帧，第一帧形状: {frames[0].shape}')
        
        # 保存视频
        self.save_images2video(frames, dump_video_path, render_fps)
        
        # 添加音频
        extracted_audio_path = os.path.join("./assets/sample_motion/export", base_vid, base_vid + ".wav")
        fallback_audio_path = os.path.join("./assets/sample_motion/export", "Joe_Biden", "Joe_Biden.wav")
        
        if os.path.exists(extracted_audio_path) and os.path.getsize(extracted_audio_path) > 0:
            audio_path = extracted_audio_path
            print(f"✅ 使用从源视频提取的音频: {audio_path}")
        elif os.path.exists(fallback_audio_path):
            audio_path = fallback_audio_path
            print(f"⚠️  源视频音频不可用，使用占位符音频: {audio_path}")
        else:
            print("❌ 没有可用的音频文件，跳过音频添加")
            return dump_image_path, dump_video_path
        
        self.add_audio_to_video(dump_video_path, dump_video_path_wa, audio_path)
        
        return dump_image_path, dump_video_path_wa

def parse_configs():
    """解析配置参数"""
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str)
    parser.add_argument("--infer", type=str)
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.create()
    cli_cfg = OmegaConf.from_cli(unknown)

    # 从环境变量解析
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

    if args.infer is not None:
        cfg_infer = OmegaConf.load(args.infer)
        cfg.merge_with(cfg_infer)

    cfg.motion_video_read_fps = 30
    cfg.merge_with(cli_cfg)
    cfg.setdefault("logger", "INFO")

    return cfg

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='整合的头像生成系统')
    
    # 输入参数
    parser.add_argument('--input_image', type=str, required=True,
                       help='输入图片路径')
    parser.add_argument('--source_video', type=str,
                       help='源视频文件路径（支持多种格式）')
    parser.add_argument('--vhap_dir', type=str,
                       help='VHAP输出目录路径（如果需要转换）')
    
    # 输出参数
    parser.add_argument('--output_dir', type=str, default='./output_integrated/',
                       help='输出目录路径')
    parser.add_argument('--drive_name', type=str, default='custom_integrated_drive',
                       help='驱动数据名称')
    
    # 模型参数
    parser.add_argument('--model_name', type=str, 
                       default="/root/autodl-tmp/LAM/model_zoo/lam_models/releases/lam/lam-20k/step_045500",
                       help='LAM模型路径')
    
    # 生成参数
    parser.add_argument('--yaw_angles', nargs='+', type=int, 
                       default=[-30, -20, -10, 0, 10, 20, 30],
                       help='偏航角度列表')
    parser.add_argument('--pitch_angles', nargs='+', type=int,
                       default=[-10, -5, 0, 5, 10],
                       help='俯仰角度列表')
    parser.add_argument('--max_frames', type=int, default=264,
                       help='最大帧数')
    
    args = parser.parse_args()
    
    # 设置环境变量
    os.environ["APP_MODEL_NAME"] = args.model_name
    os.environ["APP_INFER"] = "./configs/inference/lam-20k-8gpu.yaml"
    
    print("🚀 启动整合的头像生成系统")
    print(f"   输入图片: {args.input_image}")
    print(f"   源视频: {args.source_video}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   驱动名称: {args.drive_name}")
    print(f"   Yaw角度: {args.yaw_angles}")
    print(f"   Pitch角度: {args.pitch_angles}")
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 解析配置
    cfg = parse_configs()
    cfg.model_name = args.model_name
    
    # 初始化生成器
    generator = IntegratedAvatarGenerator(cfg)
    
    # 初始化组件
    print("\n=== 初始化系统组件 ===")
    generator.init_preprocessor()
    generator.init_flame_tracking()
    generator.init_lam_model()
    
    # 如果提供了VHAP目录，先进行转换
    if args.vhap_dir:
        print("\n=== 转换VHAP数据 ===")
        drive_output_dir = "./assets/sample_motion/export"
        drive_path, conversion_success = generator.convert_vhap_to_lam(
            vhap_dir=args.vhap_dir,
            output_dir=drive_output_dir,
            max_frames=args.max_frames,
            target_name=args.drive_name,
            source_video_path=args.source_video
        )
        
        if not conversion_success:
            print("❌ VHAP转换失败，请检查输入数据")
            return
            
        video_path = os.path.join(drive_path, f"{args.drive_name}.mp4")
    else:
        # 直接使用现有的驱动数据
        video_path = f"./assets/sample_motion/export/{args.drive_name}/{args.drive_name}.mp4"
        if not os.path.exists(video_path):
            print(f"❌ 驱动视频不存在: {video_path}")
            print("请提供 --vhap_dir 进行数据转换，或确保驱动数据已存在")
            return
    
    # 生成多视角视频
    print("\n=== 生成多视角头像视频 ===")
    generated_videos = generator.generate_multiview_videos(
        image_path=args.input_image,
        drive_path=video_path,
        output_dir=args.output_dir,
        yaw_angles=args.yaw_angles,
        pitch_angles=args.pitch_angles
    )
    
    print(f"\n🎉 整合任务完成！")
    print(f"   生成了 {len(generated_videos)} 个多视角视频")
    print(f"   输出目录: {args.output_dir}")

if __name__ == '__main__':
    main() 