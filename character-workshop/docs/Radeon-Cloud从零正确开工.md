# Radeon Cloud · 从零正确开工（防 CUDA torch 踩坑）

面向：环境弄乱后 **Destroy 重来**，或第一次上云。  
按顺序做，**不要跳步**。每步有 **通过标准**，不通过就停，不要继续 `pip install`。

| 项 | 值 |
|----|-----|
| 仓库 | https://github.com/Marsbler/qingxi-character-workshop |
| 分支 | `feature/character-workshop` |
| 应用目录 | `…/qingxi-character-workshop/character-workshop`（含 `app.py`） |
| 云控制台 | https://radeon-global.anruicloud.com/ |

---

## 绝对禁止（违反必炸）

```bash
# ❌ 禁止：在未确认 ROCm torch 之前执行
pip install -r requirements.txt
pip install torch
pip install accelerate          # 会从默认源拖 CUDA torch
pip install torch -i https://pypi.tuna.tsinghua.edu.cn/simple
```

```text
❌ 禁止：选 NVIDIA / CUDA 专用镜像（赛题要 AMD Radeon + ROCm）
❌ 禁止：看到 device=cpu、torch=+cu* 还继续 download_models / 跑生成
```

---

## 步骤 0 · 销毁旧实例（你要重来时）

1. 浏览器打开 Radeon Cloud → **Profile**  
2. **Active Instance** → **Destroy Instance**  
3. 等实例消失  

**PVC 说明：**

- 若 PVC 里以前 `pip install --user` 装过 `+cu` torch，新实例仍可能脏。  
- 重来后若一上来 torch 仍是 `+cu*`，执行文末 **附录：清洗用户 site-packages**。

---

## 步骤 1 · 新建 / 启动正确 Template

1. **Profile → My Templates → Add Template**（或用已有正确模板）  
2. 必填：
   - **Container Image：** 名称含 **ROCm** / **Radeon** / **PyTorch ROCm**（以平台列表为准）  
   - **Storage：Persistent (PVC)**  
3. **Launch** → 等 100% → **Open Notebook** → 打开 **Terminal**

当前目录常见为 `/workspace`。

---

## 步骤 2 · 硬件门禁（不过就换镜像）

```bash
rocm-smi
# 若无命令：
ls -la /dev/kfd /dev/dri 2>/dev/null || true
cat /opt/rocm/.info/version 2>/dev/null || true
```

**通过标准：**

- `rocm-smi` 能看到 AMD GPU，**或**  
- `/dev/kfd` 存在且 `/opt/rocm` 有版本信息  

**失败：** Destroy，换 ROCm 镜像 Template，不要装软件硬撑。

---

## 步骤 3 · 镜像自带 torch 门禁（先看再装）

```bash
python3 - <<'PY'
import torch
print("version:", torch.__version__)
print("available:", torch.cuda.is_available())
print("hip:", getattr(torch.version, "hip", None))
print("cuda_str:", getattr(torch.version, "cuda", None))
print("file:", torch.__file__)
if torch.cuda.is_available():
    print("name:", torch.cuda.get_device_name(0))
PY
```

### 情况 A — 理想（直接进入步骤 5）

```text
version: 含 rocm（如 2.x.x+rocm6.x）
available: True
hip: 有版本号
```

→ **不要**再 `pip install torch`。跳到 **步骤 5**。

### 情况 B — 没有 torch / ImportError

→ 做 **步骤 4** 安装 ROCm torch。

### 情况 C — `+cu*` 或 `available: False`

→ 先清洗再装 ROCm（步骤 4 全文）。若清洗后仍 False → 镜像/GPU 问题，换 Template。

---

## 步骤 4 · 仅在需要时安装 ROCm PyTorch

```bash
# 4.1 清掉错误的 CUDA 包（有就卸，没有也无妨）
python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
python3 -m pip freeze 2>/dev/null | grep -iE '^nvidia-|^cuda-toolkit|^triton==' | cut -d= -f1 | xargs -r python3 -m pip uninstall -y

# 4.2 看 ROCm 大版本（用于选 wheel）
cat /opt/rocm/.info/version 2>/dev/null || echo "no /opt/rocm version file"

# 4.3 从 PyTorch 官方 ROCm 索引安装（不要用清华默认源装 torch）
# 按本机 ROCm 改 rocm6.2 → rocm6.1 / rocm6.3
python3 -m pip install torch torchvision torchaudio \
  --index-url https://download.pytorch.org/whl/rocm6.2
```

**再跑步骤 3 的检测脚本。**

**通过标准（必须全部满足）：**

```text
available: True
version: 含 rocm，且不是纯 +cu130 这种
hip: 非 None
```

不满足 → **停止**，不要 clone 后乱装依赖。

---

## 步骤 5 · 获取代码

```bash
cd /workspace

# 证书错误时（你之前遇到过）
export GIT_SSL_NO_VERIFY=1

# 若旧目录是脏的，先挪走
mv qingxi-character-workshop qingxi-character-workshop.bak.$(date +%s) 2>/dev/null || true

git clone https://github.com/Marsbler/qingxi-character-workshop.git
cd qingxi-character-workshop
git checkout feature/character-workshop
git pull origin feature/character-workshop

cd character-workshop
test -f app.py && echo "CODE OK" || (echo "CODE FAIL" && exit 1)
ls -la app.py requirements.txt scripts/install_app_deps.sh
```

**通过标准：** 打印 `CODE OK`，且存在 `scripts/install_app_deps.sh`。

---

## 步骤 6 · 安全安装应用依赖（唯一推荐方式）

```bash
cd /workspace/qingxi-character-workshop/character-workshop

# 可选但推荐：装 Noto CJK 字体，保证海报中文不乱码
# （代码已有自动字体发现 + 无字体时会在海报右下角画警告，装字体是最稳路径）
sudo apt-get install -y fonts-noto-cjk 2>/dev/null || true

# 再次确认 ROCm torch（门禁）
python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip)"

# 用仓库脚本（内含 preflight / postflight，防被盖成 +cu）
bash scripts/install_app_deps.sh
```

脚本会：

1. 检查 torch 已是 ROCm 可用  
2. `pip install -r requirements.txt`（**不含**会拖 CUDA torch 的 accelerate 硬依赖）  
3. `pip install accelerate --no-deps`  
4. **再次检查** torch 仍是 ROCm  

**通过标准：** 脚本结束打印 `INSTALL APP DEPS OK`。

若失败并提示 torch 被覆盖 → 回到步骤 4，**不要**反复 `pip install -r`。

---

## 步骤 7 · 项目 device 检测

```bash
cd /workspace/qingxi-character-workshop/character-workshop
export MOCK=0

python3 - <<'PY'
from src.device import detect_device, vram_profile
d = detect_device()
print(d)
print("profile", vram_profile(d))
assert d.mock_mode is False
assert d.device_type == "cuda", d
assert d.rocm_hint or (d.torch_version and "rocm" in d.torch_version.lower())
print("PROJECT DEVICE OK")
PY
```

**通过标准：** `PROJECT DEVICE OK`，`device_type=cuda`，`vram_gb` 有值。

---

## 步骤 8 · 下载模型

```bash
cd /workspace/qingxi-character-workshop/character-workshop
export MOCK=0
# 可选 HF 镜像
# export HF_ENDPOINT=https://hf-mirror.com

bash scripts/download_models.sh
```

耗时长，保持实例不要 Destroy。PVC 可保留缓存。

---

## 步骤 9 · 真机出卡冒烟（P0）

```bash
cd /workspace/qingxi-character-workshop/character-workshop
export MOCK=0

python3 - <<'PY'
from src.orchestrator import generate_card_job, JobState
r = generate_card_job(
    user_text="青衣空系旅人，背着纸灯笼，寡言",
    affinity_pref="空",
    mock=False,
)
print(r.state, r.job_id, r.error)
print("card", r.card_path)
assert r.state == JobState.READY
assert r.card_path and r.card_path.exists()
print("CARD E2E OK")
PY
```

**通过标准：** `CARD E2E OK`，`outputs/<id>/card.png` 为真实立绘（非 MOCK 色块字）。

---

## 步骤 10 · 启动 Web

```bash
cd /workspace/qingxi-character-workshop/character-workshop
export MOCK=0
export PORT=7860
python3 app.py
```

可选公网（官方 tunnel）：

```bash
# 另开 Terminal
/var/run/secrets/frp-self-service/install
export PATH="$HOME/.local/bin:$PATH"
rc-tunnel expose --port 7860
```

用完：

```bash
rc-tunnel stop 2>/dev/null || true
# 浏览器 Destroy Instance 省额度
```

---

## 正确顺序总表（背下来）

```text
Destroy 脏实例（可选）
  → 选 ROCm 镜像 Launch
  → rocm-smi 有 GPU
  → python 里 torch 已是 ROCm 且 available=True
       （否则只装 ROCm torch，绝不装默认 torch）
  → git clone + checkout feature/character-workshop
  → bash scripts/install_app_deps.sh   ← 唯一推荐装依赖方式
  → detect_device 通过
  → download_models
  → generate_card_job MOCK=0
  → app.py
```

---

## 附录 A · PVC 残留 CUDA 包清洗

新实例一上来就是 `+cu*` 时：

```bash
python3 -c "import torch,sys; print(torch.__file__); print(sys.path)"

# 用户目录残留
rm -rf ~/.local/lib/python*/site-packages/torch*
rm -rf ~/.local/lib/python*/site-packages/nvidia*
rm -rf ~/.local/lib/python*/site-packages/cuda*

python3 -m pip uninstall -y torch torchvision torchaudio 2>/dev/null || true
# 然后重新做步骤 3～4
```

---

## 附录 B · Git 证书错误

```bash
export GIT_SSL_NO_VERIFY=1
# 或
sudo apt-get update && sudo apt-get install -y ca-certificates && sudo update-ca-certificates
```

---

## 附录 C · 本机改代码后同步云

**本机：**

```powershell
cd "D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop"
git add -A
git commit -m "your message"
git push origin feature/character-workshop
```

**云上：**

```bash
cd /workspace/qingxi-character-workshop
export GIT_SSL_NO_VERIFY=1
git pull origin feature/character-workshop
cd character-workshop
# 一般不必重装依赖；若 requirements 变了再跑 install_app_deps.sh
```

---

## 附录 D · 与旧清单章节对应

| 旧章节 | 本文件 |
|--------|--------|
| A 开实例 | 步骤 0～1 |
| B GPU 自检 | 步骤 2 |
| C clone | 步骤 5 |
| D 依赖 | 步骤 3～6（**以本文件为准**） |
| E 模型 | 步骤 8 |
| F 冒烟 | 步骤 9 |
| G Web | 步骤 10 |
