#!/usr/bin/env python3
"""
转换VHAP_track输出为LAM兼容的驱动文件格式

新增功能：
- 自动处理视频文件（支持.mp4/.mov/.avi/.mkv等格式）
- 自动提取音频文件
- 生成完全符合multi_view_image_generation.py要求的输入数据配置

使用方法：
1. 设置vhap_dir为VHAP输出目录
2. 设置output_dir为LAM的assets/sample_motion/export目录
3. 设置source_video_path为原始视频文件路径（可选）
4. 运行脚本，所有必需文件将自动生成
"""

import os
import numpy as np
import json
import shutil
import argparse
from pathlib import Path

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

def process_video_file(source_video_path, output_dir, target_name):
    """
    处理视频文件，复制/转换视频并提取音频
    
    Args:
        source_video_path: 源视频文件路径
        output_dir: 输出目录路径
        target_name: 目标文件名（不含扩展名）
        
    Returns:
        bool: 处理是否成功
    """
    if not source_video_path or not os.path.exists(source_video_path):
        print(f"⚠️  跳过视频处理：视频文件不存在 {source_video_path}")
        return False
    
    target_mp4 = output_dir / f'{target_name}.mp4'
    target_audio = output_dir / f'{target_name}.wav'
    
    # 获取并显示视频信息
    print(f"\n📹 分析视频文件: {source_video_path}")
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
        print(f"📋 复制MP4文件: {source_video_path} -> {target_mp4}")
        shutil.copy2(source_video_path, target_mp4)
        need_convert = False
        
        # 尝试提取音频
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
    
    # 转换其他格式到MP4
    if need_convert:
        video_success, audio_path = convert_video_format(source_video_path, str(target_mp4))
        if not video_success:
            print(f"❌ 视频处理失败: {source_video_path}")
            return False
    
    # 检查最终文件
    video_exists = target_mp4.exists() and target_mp4.stat().st_size > 0
    audio_exists = target_audio.exists() and target_audio.stat().st_size > 0
    
    print(f"📋 视频文件: {'✅' if video_exists else '❌'} {target_mp4}")
    print(f"🎵 音频文件: {'✅' if audio_exists else '❌'} {target_audio}")
    
    return video_exists

def convert_vhap_to_lam(vhap_dir, output_dir, max_frames=264, target_name="custom_drive", source_video_path=None):
    """
    转换VHAP输出为LAM格式
    
    Args:
        vhap_dir: VHAP输出目录路径
        output_dir: 输出目录路径  
        max_frames: 最大帧数（默认264帧）
        target_name: 目标驱动名称
        source_video_path: 源视频文件路径（可选）
    """
    vhap_path = Path(vhap_dir)
    output_path = Path(output_dir) / target_name
    output_path.mkdir(parents=True, exist_ok=True)
    
    print(f"🔄 转换VHAP数据：{vhap_path} -> {output_path}")
    
    # 1. 复制并转换flame_param文件
    flame_param_dir = output_path / "flame_param"
    flame_param_dir.mkdir(exist_ok=True)
    
    vhap_flame_dir = vhap_path / "flame_param"
    flame_files = sorted([f for f in os.listdir(vhap_flame_dir) if f.endswith('.npz')])
    
    # 限制最大帧数
    flame_files = flame_files[:max_frames]
    print(f"🔥 处理{len(flame_files)}个FLAME参数文件...")
    
    for i, flame_file in enumerate(flame_files):
        # 读取VHAP格式
        vhap_data = np.load(vhap_flame_dir / flame_file)
        
        # 转换为LAM格式
        lam_data = {}
        
        # 直接复制的参数
        for key in ['translation', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'expr']:
            if key in vhap_data:
                lam_data[key] = vhap_data[key]
        
        # 转换shape为betas（只在第一帧保存）
        if i == 0 and 'shape' in vhap_data:
            # 将(300,) -> (1, 300)
            lam_data['betas'] = vhap_data['shape'].reshape(1, -1)
        
        # 保存转换后的文件
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
    
    # 4. 创建flame_params.json（合并所有帧的参数）
    print("📝 生成flame_params.json...")
    create_flame_params_json(flame_param_dir, output_path, max_frames)
    
    # 5. 处理视频文件（新增功能）
    if source_video_path:
        print(f"\n🎬 处理视频文件...")
        video_success = process_video_file(source_video_path, output_path, target_name)
        if video_success:
            print(f"✅ 视频处理完成")
        else:
            print(f"⚠️  视频处理失败，您需要手动添加视频文件")
    else:
        print(f"\n⚠️  未提供视频路径，需要手动添加以下文件：")
        print(f"   - {target_name}.mp4 (驱动视频)")
        print(f"   - {target_name}.wav (音频文件)")
    
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
        print(f"\n🎉 转换完成！所有必需文件已就绪，可直接在multi_view_image_generation.py中使用")
        print(f"   驱动名称设置为: drive_name = '{target_name}'")
    else:
        print(f"\n⚠️  转换完成，但部分文件缺失，请手动补齐后使用")
    
    print(f"📁 输出目录：{output_path}")

def create_flame_params_json(flame_param_dir, output_dir, max_frames):
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
    
    # 读取所有帧的参数
    for i in range(max_frames):
        flame_file = flame_param_dir / f"{i:05d}.npz"
        if flame_file.exists():
            data = np.load(flame_file)
            
            for key in ['expr', 'rotation', 'neck_pose', 'jaw_pose', 'eyes_pose', 'translation']:
                if key in data:
                    # 转换为列表格式
                    if data[key].ndim == 2:
                        all_params[key].append(data[key][0].tolist())
                    else:
                        all_params[key].append(data[key].tolist())
            
            # betas只保存一次
            if all_params["betas"] is None and "betas" in data:
                all_params["betas"] = data["betas"][0].tolist()
    
    # 保存JSON文件
    json_file = output_dir / "flame_params.json"
    with open(json_file, 'w') as f:
        json.dump(all_params, f, indent=2)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='转换VHAP输出为LAM兼容格式')
    parser.add_argument('--vhap_dir', type=str, 
                       default="/root/autodl-tmp/VHAP_track/mono_jp/export_epoch0",
                       help='VHAP输出目录路径')
    parser.add_argument('--output_dir', type=str,
                       default="/root/autodl-tmp/LAM/assets/sample_motion/export", 
                       help='LAM输出目录路径')
    parser.add_argument('--target_name', type=str, default="custom_jp_drive",
                       help='目标驱动名称')
    parser.add_argument('--max_frames', type=int, default=264,
                       help='最大帧数')
    parser.add_argument('--source_video_path', type=str,
                       default="/root/autodl-tmp/datasets/mono_jp/apple_jp_3.mov",
                       help='源视频文件路径（支持.mp4/.mov/.avi/.mkv等格式）')
    
    args = parser.parse_args()
    
    print(f"🚀 开始转换VHAP数据...")
    print(f"   VHAP目录: {args.vhap_dir}")
    print(f"   输出目录: {args.output_dir}")
    print(f"   目标名称: {args.target_name}")
    print(f"   最大帧数: {args.max_frames}")
    print(f"   源视频路径: {args.source_video_path}")
    
    convert_vhap_to_lam(
        vhap_dir=args.vhap_dir,
        output_dir=args.output_dir,
        max_frames=args.max_frames,
        target_name=args.target_name,
        source_video_path=args.source_video_path
    ) 