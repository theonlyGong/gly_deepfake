"""
DeepfakeTIMIT 视频分帧 + 人脸截取脚本
使用 dlib 检测人脸，扩展宽高 25%，保存到 video_imgs 文件夹
"""

import os
import sys
import cv2
import numpy as np
from pathlib import Path
import argparse

# dlib 人脸检测
try:
    import dlib
except ImportError:
    print("Error: dlib not installed. Please install with: pip install dlib")
    sys.exit(1)


def get_face_detector():
    """获取 dlib 人脸检测器"""
    # 尝试加载 shape_predictor 和 face_detector
    detector = dlib.get_frontal_face_detector()

    # 尝试查找 shape_predictor 模型文件
    predictor_paths = [
        "shape_predictor_68_face_landmarks.dat",    # 你自己下载的默认目录
        "../checkpoints/shape_predictor_68_face_landmarks.dat"
    ]

    predictor = None
    for path in predictor_paths:
        if os.path.exists(path):
            try:
                predictor = dlib.shape_predictor(path)
                print(f"Loaded shape predictor from: {path}")
                break
            except Exception as e:
                print(f"Failed to load predictor from {path}: {e}")

    if predictor is None:
        print("Warning: shape_predictor_68_face_landmarks.dat not found.")
        print("Face detection will work but cropping may be less accurate.")
        print("Download from: http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2")

    return detector, predictor


def expand_bbox(x1, y1, x2, y2, img_width, img_height, expand_ratio=0.25):
    """
    扩展人脸边界框，宽高各扩展 expand_ratio（默认25%）

    Args:
        x1, y1, x2, y2: 原始边界框坐标
        img_width, img_height: 图像尺寸
        expand_ratio: 扩展比例

    Returns:
        扩展后的边界框坐标 (x1, y1, x2, y2)
    """
    width = x2 - x1
    height = y2 - y1

    # 计算扩展量（每边扩展12.5%，总共25%）
    expand_x = int(width * expand_ratio / 2)
    expand_y = int(height * expand_ratio / 2)

    # 扩展边界
    x1_new = max(0, x1 - expand_x)
    y1_new = max(0, y1 - expand_y)
    x2_new = min(img_width, x2 + expand_x)
    y2_new = min(img_height, y2 + expand_y)

    return x1_new, y1_new, x2_new, y2_new


def detect_and_crop_face(img, detector, predictor=None, expand_ratio=0.25):
    """
    检测人脸并截取（带扩展）

    Args:
        img: 输入图像 (numpy array)
        detector: dlib 人脸检测器
        predictor: dlib 人脸关键点预测器（可选）
        expand_ratio: 边界框扩展比例

    Returns:
        cropped_face: 截取的人脸图像，如果未检测到则返回 None
        bbox: 边界框坐标 (x1, y1, x2, y2)
    """
    if img is None or img.size == 0:
        return None, None

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = detector(gray, 1)  # 1 表示 upsample 次数

    if len(faces) == 0:
        return None, None

    # 选择最大的人脸
    largest_face = max(faces, key=lambda f: f.width() * f.height())

    # 获取边界框
    x1 = largest_face.left()
    y1 = largest_face.top()
    x2 = largest_face.right()
    y2 = largest_face.bottom()

    # 如果有关键点预测器，使用关键点优化边界框
    if predictor is not None:
        landmarks = predictor(gray, largest_face)
        # 使用关键点重新计算边界（可选：更精确的边界）
        x_coords = [landmarks.part(i).x for i in range(68)]
        y_coords = [landmarks.part(i).y for i in range(68)]
        x1 = min(x_coords)
        y1 = min(y_coords)
        x2 = max(x_coords)
        y2 = max(y_coords)

    img_height, img_width = img.shape[:2]

    # 扩展边界框
    x1, y1, x2, y2 = expand_bbox(x1, y1, x2, y2, img_width, img_height, expand_ratio)

    # 截取人脸
    cropped_face = img[y1:y2, x1:x2]

    return cropped_face, (x1, y1, x2, y2)


def process_video(video_path, output_dir, detector, predictor,
                  frame_interval=1, max_frames=None, min_face_size=(80, 80)):
    """
    处理单个视频文件

    Args:
        video_path: 视频文件路径
        output_dir: 输出目录
        detector: dlib 人脸检测器
        predictor: dlib 人脸关键点预测器
        frame_interval: 抽帧间隔（每 N 帧取一帧，1 表示每帧都取）
        max_frames: 最大处理帧数（None 表示不限制）
        min_face_size: 最小人脸尺寸 (宽, 高)，小于此尺寸跳过

    Returns:
        saved_count: 成功保存的帧数
    """
    # 获取视频文件名（不含扩展名）
    video_name = Path(video_path).stem

    # 打开视频
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"  Error: Cannot open video: {video_path}")
        return 0

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    print(f"  Total frames: {total_frames}, FPS: {fps:.2f}")
    print(f"  Frame interval: {frame_interval}, Expected frames to process: ~{total_frames // frame_interval}")

    saved_count = 0
    frame_idx = 0
    processed_idx = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_idx += 1

        # 按间隔抽帧
        if frame_idx % frame_interval != 0:
            continue

        # 检测并截取人脸
        face_img, bbox = detect_and_crop_face(frame, detector, predictor, expand_ratio=0.25)

        if face_img is not None and face_img.size > 0:
            # 检查最小尺寸
            h, w = face_img.shape[:2]
            if w >= min_face_size[0] and h >= min_face_size[1]:
                # 保存图片: 视频名_帧号.jpg
                output_filename = f"{video_name}_{frame_idx:06d}.jpg"
                output_path = os.path.join(output_dir, output_filename)

                cv2.imwrite(output_path, face_img)
                saved_count += 1
                processed_idx += 1

                if processed_idx % 50 == 0:
                    print(f"    Processed {processed_idx} frames, saved {saved_count} faces...")
            else:
                print(f"    Skipped frame {frame_idx}: face too small ({w}x{h})")

        # 检查最大帧数限制
        if max_frames is not None and processed_idx >= max_frames:
            print(f"    Reached max_frames limit: {max_frames}")
            break

    cap.release()
    return saved_count


def find_avi_files(root_dir):
    """递归查找所有 .avi 文件"""
    avi_files = []
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.lower().endswith('.avi'):
                avi_files.append(os.path.join(root, file))
    return sorted(avi_files)


def main():
    parser = argparse.ArgumentParser(description='DeepfakeTIMIT Video Frame Extraction with Face Cropping')

    parser.add_argument('--input_dir', type=str, default='../DeepfakeTIMIT',
                        help='输入视频根目录（包含 .avi 文件）')
    parser.add_argument('--output_dir', type=str, default='../video_imgs',
                        help='输出图片保存目录')
    parser.add_argument('--frame_interval', type=int, default=5,
                        help='抽帧间隔（每 N 帧取一帧，默认 5）')
    parser.add_argument('--max_frames', type=int, default=10,
                        help='每个视频最大处理帧数（默认不限制）')
    parser.add_argument('--min_face_size', type=int, nargs=2, default=[80, 80],
                        help='最小人脸尺寸 宽 高（默认 80 80）')
    parser.add_argument('--expand_ratio', type=float, default=0.25,
                        help='人脸边界框扩展比例（默认 0.25 = 25%）')

    args = parser.parse_args()

    # 检查输入目录
    if not os.path.exists(args.input_dir):
        print(f"Error: Input directory not found: {args.input_dir}")
        sys.exit(1)

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    print(f"Output directory: {args.output_dir}")

    # 加载人脸检测器
    print("\nLoading dlib face detector...")
    detector, predictor = get_face_detector()
    print("Face detector loaded!\n")

    # 查找所有 AVI 文件
    print(f"Scanning for .avi files in: {args.input_dir}")
    avi_files = find_avi_files(args.input_dir)
    print(f"Found {len(avi_files)} video files\n")

    if len(avi_files) == 0:
        print("No .avi files found!")
        sys.exit(0)

    # 处理每个视频
    total_saved = 0
    total_videos = len(avi_files)

    for idx, video_path in enumerate(avi_files, 1):
        print(f"[{idx}/{total_videos}] Processing: {video_path}")

        try:
            saved = process_video(
                video_path,
                args.output_dir,
                detector,
                predictor,
                frame_interval=args.frame_interval,
                max_frames=args.max_frames,
                min_face_size=tuple(args.min_face_size)
            )
            total_saved += saved
            print(f"  Saved {saved} face images\n")
        except Exception as e:
            print(f"  Error processing video: {e}\n")
            import traceback
            traceback.print_exc()

    print("=" * 60)
    print("Processing Complete!")
    print(f"Total videos processed: {total_videos}")
    print(f"Total face images saved: {total_saved}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print("=" * 60)


if __name__ == '__main__':
    main()