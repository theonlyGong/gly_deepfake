#!/bin/bash

# =============================================================================
# Deepfake Detection 环境一键安装脚本
# 项目名称: gly_deepfake
# 作者: Claude
# 日期: 2025-06-15
# =============================================================================

set -e  # 遇到错误立即退出

echo "=========================================="
echo "  Deepfake Detection 环境安装脚本"
echo "=========================================="
echo ""

# 颜色定义
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# 打印带颜色的信息
print_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查命令是否存在
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# =============================================================================
# 步骤 1: 检查 Python 环境
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 1/5: 检查 Python 环境"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command_exists python3; then
    PYTHON_CMD="python3"
elif command_exists python; then
    PYTHON_CMD="python"
else
    print_error "未找到 Python，请先安装 Python 3.8 或更高版本"
    exit 1
fi

PYTHON_VERSION=$($PYTHON_CMD --version 2>&1 | awk '{print $2}')
print_info "检测到 Python 版本: $PYTHON_VERSION"

# 检查 Python 版本是否 >= 3.8
REQUIRED_VERSION="3.8"
if ! $PYTHON_CMD -c "import sys; exit(0 if sys.version_info >= (3, 8) else 1)" 2>/dev/null; then
    print_error "Python 版本过低，需要 3.8 或更高版本"
    exit 1
fi
print_success "Python 版本符合要求"

# =============================================================================
# 步骤 2: 检查 pip
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 2/5: 检查 pip"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

if command_exists pip3; then
    PIP_CMD="pip3"
elif command_exists pip; then
    PIP_CMD="pip"
else
    print_warning "未找到 pip，尝试安装..."
    $PYTHON_CMD -m ensurepip --upgrade 2>/dev/null || {
        print_error "pip 安装失败，请手动安装 pip"
        exit 1
    }
    PIP_CMD="$PYTHON_CMD -m pip"
fi

PIP_VERSION=$($PIP_CMD --version 2>&1 | awk '{print $2}')
print_info "检测到 pip 版本: $PIP_VERSION"
print_success "pip 可用"

# 升级 pip
print_info "升级 pip..."
$PIP_CMD install --upgrade pip -q

# =============================================================================
# 步骤 3: 安装 PyTorch (带 CUDA 支持)
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 3/5: 安装 PyTorch"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

print_info "检测 CUDA 可用性..."
if command_exists nvidia-smi; then
    CUDA_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null | head -n 1)
    if [ -n "$CUDA_VERSION" ]; then
        print_success "检测到 NVIDIA GPU，驱动版本: $CUDA_VERSION"
        print_info "安装 PyTorch with CUDA 支持..."
        $PIP_CMD install torch torchvision --index-url https://download.pytorch.org/whl/cu118 -q
    else
        print_warning "未检测到 NVIDIA GPU"
        print_info "安装 PyTorch CPU 版本..."
        $PIP_CMD install torch torchvision -q
    fi
else
    print_warning "未找到 nvidia-smi，安装 PyTorch CPU 版本..."
    $PIP_CMD install torch torchvision -q
fi

# 验证 PyTorch 安装
print_info "验证 PyTorch 安装..."
$PYTHON_CMD -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')" 2>/dev/null || {
    print_error "PyTorch 安装验证失败"
    exit 1
}
print_success "PyTorch 安装成功"

# =============================================================================
# 步骤 4: 安装其他依赖包
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 4/5: 安装其他依赖"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

DEPENDENCIES=(
    "Pillow"              # 图像处理
    "numpy"               # 数值计算
    "scikit-learn"        # 机器学习工具 (评估指标)
    "tqdm"                # 进度条 (训练时可能用到)
    "matplotlib"          # 绘图 (可视化用)
    "opencv-python"       # OpenCV (图像处理)
)

print_info "将安装以下依赖包:"
for dep in "${DEPENDENCIES[@]}"; do
    echo "  - $dep"
done
echo ""

for dep in "${DEPENDENCIES[@]}"; do
    print_info "安装 $dep..."
    $PIP_CMD install "$dep" -q || {
        print_warning "$dep 安装可能遇到问题，尝试强制安装..."
        $PIP_CMD install "$dep" --force-reinstall -q
    }
done

print_success "所有依赖安装完成"

# =============================================================================
# 步骤 5: 创建 requirements.txt
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  步骤 5/5: 生成 requirements.txt"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_DIR"

$PIP_CMD freeze > requirements.txt 2>/dev/null || {
    print_warning "无法生成 requirements.txt (可能在虚拟环境中)"
}

# 同时创建一个简化的 requirements.txt
 cat > requirements.txt << EOF
# Deepfake Detection 项目依赖
# 生成日期: $(date '+%Y-%m-%d %H:%M:%S')

# 核心依赖
torch>=2.0.0
torchvision>=0.15.0

# 图像处理
Pillow>=9.0.0
opencv-python>=4.7.0

# 科学计算
numpy>=1.24.0
scikit-learn>=1.2.0

# 其他
tqdm>=4.65.0
matplotlib>=3.7.0
EOF

print_success "已生成 requirements.txt"

# =============================================================================
# 验证安装
# =============================================================================
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "  验证安装"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

print_info "验证所有依赖是否可以正常导入..."

$PYTHON_CMD << 'EOF'
import sys

def check_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f"  ✓ {package_name}")
        return True
    except ImportError as e:
        print(f"  ✗ {package_name}: {e}")
        return False

all_ok = True
all_ok &= check_import("torch")
all_ok &= check_import("torchvision")
all_ok &= check_import("PIL", "Pillow")
all_ok &= check_import("numpy")
all_ok &= check_import("sklearn", "scikit-learn")
all_ok &= check_import("cv2", "opencv-python")
all_ok &= check_import("tqdm")
all_ok &= check_import("matplotlib")

if all_ok:
    print("\n所有依赖验证通过!")
    sys.exit(0)
else:
    print("\n部分依赖验证失败，请检查安装")
    sys.exit(1)
EOF

if [ $? -ne 0 ]; then
    print_error "依赖验证失败"
    exit 1
fi

# =============================================================================
# 完成
# =============================================================================
echo ""
echo "=========================================="
echo "  环境安装完成!"
echo "=========================================="
echo ""
echo "项目路径: $PROJECT_DIR"
echo ""
echo "可用命令:"
echo "  训练模型:   python gly_train.py"
echo "  单张预测:   python gly_single.py --image <图片路径>"
echo "  批量测试:   python gly_batch_test.py"
echo ""
echo "注意: 请确保已将预训练权重文件放入 checkpoints/ 目录:"
echo "  - checkpoints/freqnet_backbone_only.pth"
echo "  - checkpoints/ViT-L-14.pt"
echo ""
echo "=========================================="