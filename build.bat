@echo off
chcp 65001 >nul 2>&1
echo ========================================
echo  EGO AGI Nuitka 打包脚本
echo ========================================
echo.

REM 检查 Python 3.12
where py -3.12 >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo [错误] 未找到 Python 3.12，请先安装
    echo 下载地址: https://www.python.org/downloads/
    pause
    exit /b 1
)

REM 创建虚拟环境
echo [1/5] 创建虚拟环境...
if not exist _build_venv (
    py -3.12 -m venv _build_venv
    if %ERRORLEVEL% neq 0 (
        echo [错误] 创建虚拟环境失败
        pause
        exit /b 1
    )
) else (
    echo 虚拟环境已存在，跳过创建
)

REM 安装 Nuitka
echo [2/5] 安装 Nuitka 和依赖...
_build_venv\Scripts\pip.exe install nuitka ordered-set zstandard requests chromadb python-dotenv
if %ERRORLEVEL% neq 0 (
    echo [错误] 安装 Nuitka 失败
    pause
    exit /b 1
)

REM 预配置 Nuitka 缓存：检查 MinGW64 编译器
echo [3/5] 配置编译器缓存...
set "NUITKA_CACHE_DIR=%USERPROFILE%\AppData\Local\Nuitka\Nuitka\Cache"
set "MINGW_CACHE=%NUITKA_CACHE_DIR%\downloads\gcc\x86_64\15.2.0posix-13.0.0-msvcrt-r6"
if not exist "%MINGW_CACHE%\mingw64\bin\gcc.exe" (
    echo [提示] MinGW64 未在缓存中找到，Nuitka 将自动下载
    echo 如果下载失败，请手动从以下地址下载并解压到 %MINGW_CACHE%
    echo https://github.com/brechtsanders/winlibs_mingw/releases/download/15.2.0posix-13.0.0-msvcrt-r6/winlibs-x86_64-posix-seh-gcc-15.2.0-mingw-w64msvcrt-13.0.0-r6.zip
) else (
    echo MinGW64 缓存已就绪
)

REM 清理旧的编译产物
echo [4/5] 清理旧产物...
if exist dist rmdir /s /q dist
if exist EGO_GUI.build rmdir /s /q EGO_GUI.build

REM 运行 Nuitka 编译
echo [5/5] 开始编译（预计 10-30 分钟）...
echo.
set NUITKA_CACHE_DIR=%NUITKA_CACHE_DIR%
_build_venv\Scripts\python.exe -m nuitka --assume-yes-for-downloads --onefile --standalone --windows-disable-console --windows-icon-from-ico=EGO.ico --output-dir=dist --enable-plugin=tk-inter --include-package=agent --include-package=chromadb --include-package=dotenv --nofollow-import-to=tkinter.test --nofollow-import-to=test --mingw64 EGO_GUI.py

if %ERRORLEVEL% neq 0 (
    echo.
    echo [错误] 编译失败，请检查错误信息
    pause
    exit /b 1
)

echo.
echo ========================================
echo  编译完成！
echo ========================================
echo 输出位置: dist\EGO_GUI.exe
echo.
echo 使用方法:
echo 1. 将 dist\EGO_GUI.exe 复制到项目根目录（与 data/ 文件夹同级）
echo 2. 或将整个 data/ 文件夹复制到 EGO_GUI.exe 所在目录
echo 3. 运行时需要 LM Studio 服务在 localhost:1234 运行
echo 4. 首次运行会在同目录创建 data/logs/ 等运行时目录
echo 5. sys.json 和 data/prompts/ 为配置文件，可按需编辑
echo 6. EGO.ico 为程序图标（由 EGO.png 转换），请保留用于重新打包
echo.
pause
