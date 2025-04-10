#!/usr/bin/env fish

# Step 1: 激活 Conda 环境（确保 conda 命令可用）
if type -q conda
    echo "Activating torch_env..."
    conda activate torch_env
else
    echo "❌ conda 未安装或不可用，请检查 miniconda3 路径"
    exit 1
end

# Step 2: 设置动态库路径
set -x DYLD_LIBRARY_PATH $CONDA_PREFIX/lib/python3.10/site-packages/torch/lib $DYLD_LIBRARY_PATH
echo "✅ DYLD_LIBRARY_PATH 已设置为: $DYLD_LIBRARY_PATH"

# Step 3: 构建测试目标
echo "📦 编译测试中..."
cargo test --no-run

# Step 4: 找到测试二进制路径
set target_path (find target/debug/deps -type f -perm +111 -name "*integration_test*" | head -n 1)

if test -z "$target_path"
    echo "❌ 未找到测试二进制文件"
    exit 1
end

echo "🐞 使用 lldb 调试 $target_path"
echo "提示：在 lldb 中输入 `run dqn_cart_pole_test` 启动测试"

# Step 5: 启动 LLDB
lldb $target_path