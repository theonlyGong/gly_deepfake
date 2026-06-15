@echo off
chcp 65001 >nul
REM =============================================================================
REM Deepfake Detection 环境一键安装脚本 (Windows版本)
REM 项目名称: gly_deepfake
REM =============================================================================

echo ==========================================
echo   Deepfake Detection 环境安装脚本
echo ==========================================
echo.

REM 设置颜色
set "INFO=[INFO]"
set "SUCCESS=[SUCCESS]"
set "WARNING=[WARNING]"
set "ERROR=[ERROR]"

REM =============================================================================
REM 步骤 1: 检查 Python 环境
REM =============================================================================
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   步骤 1/5: 检查 Python 环境
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% 未找到 Python，请先安装 Python 3.8 或更高版本
    echo 请从 https://www.python.org/downloads/ 下载安装
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('python --version 2^>^&1') do set PYTHON_VERSION=%%a
echo %INFO% 检测到 %PYTHON_VERSION%

REM 检查 Python 版本 (简化检查)
python -c "import sys; sys.exit(0 if sys.version_info >= (3, 8) else 1)" >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% Python 版本过低，需要 3.8 或更高版本
    pause
    exit /b 1
)
echo %SUCCESS% Python 版本符合要求

REM =============================================================================
REM 步骤 2: 检查 pip
REM =============================================================================
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   步骤 2/5: 检查 pip
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

pip --version >nul 2>&1
if %errorlevel% neq 0 (
    echo %ERROR% 未找到 pip
    pause
    exit /b 1
)

for /f "tokens=*" %%a in ('pip --version 2^>^&1') do set PIP_VERSION=%%a
echo %INFO% 检测到 %PIP_VERSION%
echo %SUCCESS% pip 可用

echo %INFO% 升级 pip...
pip install --upgrade pip -q

REM =============================================================================
REM 步骤 3: 安装 PyTorch
REM =============================================================================
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   步骤 3/5: 安装 PyTorch
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo %INFO% 安装 PyTorch (CPU版本)...
echo %WARNING% 如需 CUDA 支持，请手动访问 https://pytorch.org/get-started/locally/ 安装

pip install torch torchvision -q
if %errorlevel% neq 0 (
    echo %ERROR% PyTorch 安装失败
    pause
    exit /b 1
)

echo %INFO% 验证 PyTorch 安装...
python -c "import torch; print(f'PyTorch 版本: {torch.__version__}'); print(f'CUDA 可用: {torch.cuda.is_available()}')"
if %errorlevel% neq 0 (
    echo %ERROR% PyTorch 验证失败
    pause
    exit /b 1
)
echo %SUCCESS% PyTorch 安装成功

REM =============================================================================
REM 步骤 4: 安装其他依赖
REM =============================================================================
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   步骤 4/5: 安装其他依赖
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo %INFO% 安装依赖包...
echo   - Pillow (图像处理)
echo   - numpy (数值计算)
echo   - scikit-learn (机器学习工具)
echo   - tqdm (进度条)
echo   - matplotlib (绘图)
echo   - opencv-python (OpenCV图像处理)
echo.

pip install Pillow numpy scikit-learn tqdm matplotlib opencv-python -q
if %errorlevel% neq 0 (
    echo %WARNING% 部分包安装可能遇到问题
) else (
    echo %SUCCESS% 依赖包安装完成
)

REM =============================================================================
REM 步骤 5: 生成 requirements.txt
REM =============================================================================
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   步骤 5/5: 生成 requirements.txt
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo # Deepfake Detection 项目依赖 > requirements.txt
echo # 生成日期: %date% %time% >> requirements.txt
echo. >> requirements.txt
echo # 核心依赖 >> requirements.txt
echo torch>=2.0.0 >> requirements.txt
echo torchvision>=0.15.0 >> requirements.txt
echo. >> requirements.txt
echo # 图像处理 >> requirements.txt
echo Pillow>=9.0.0 >> requirements.txt
echo opencv-python>=4.7.0 >> requirements.txt
echo. >> requirements.txt
echo # 科学计算 >> requirements.txt
echo numpy>=1.24.0 >> requirements.txt
echo scikit-learn>=1.2.0 >> requirements.txt
echo. >> requirements.txt
echo # 其他 >> requirements.txt
echo tqdm>=4.65.0 >> requirements.txt
echo matplotlib>=3.7.0 >> requirements.txt

echo %SUCCESS% 已生成 requirements.txt

REM =============================================================================
REM 验证安装
REM =============================================================================
echo.
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
echo   验证安装
echo ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

echo %INFO% 验证依赖导入...

python -c "
import sys

def check_import(module_name, package_name=None):
    if package_name is None:
        package_name = module_name
    try:
        __import__(module_name)
        print(f'  [OK] {package_name}')
        return True
    except ImportError as e:
        print(f'  [FAIL] {package_name}')
        return False

all_ok = True
all_ok &= check_import('torch')
all_ok &= check_import('torchvision')
all_ok &= check_import('PIL', 'Pillow')
all_ok &= check_import('numpy')
all_ok &= check_import('sklearn', 'scikit-learn')
all_ok &= check_import('cv2', 'opencv-python')
all_ok &= check_import('tqdm')
all_ok &= check_import('matplotlib')

if all_ok:
    print('')
    print('所有依赖验证通过!')
    sys.exit(0)
else:
    print('')
    print('部分依赖验证失败')
    sys.exit(1)
"

if %errorlevel% neq 0 (
    echo %ERROR% 验证失败
    pause
    exit /b 1
)

REM =============================================================================
REM 完成
REM =============================================================================
echo.
echo ==========================================
echo   环境安装完成!
echo ==========================================
echo.
echo 可用命令:
echo   训练模型:   python gly_train.py
echo   单张预测:   python gly_single.py --image ^<图片路径^>
echo   批量测试:   python gly_batch_test.py
echo.
echo 注意: 请确保已将预训练权重文件放入 checkpoints\ 目录:
echo   - checkpoints\freqnet_backbone_only.pth
echo   - checkpoints\ViT-L-14.pt
echo.
pause