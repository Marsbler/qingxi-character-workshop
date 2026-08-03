# Radeon Cloud 逐步操作清单（命令级）

面向：**青汐灵隙角色工坊** · Task 13 真推理  
官方云：https://radeon-global.anruicloud.com/  
官方指南：https://github.com/AMD-DEV-CONTEST/Radeon-hackathon-2026-07/tree/main/Radeon-Cloud-User%20Guide  

本地代码 worktree：

```text
D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop\character-workshop
```

**远程仓库（已推送，云上 clone 用这个）：**

| 项 | 值 |
|----|-----|
| GitHub | https://github.com/Marsbler/qingxi-character-workshop |
| Clone URL | `https://github.com/Marsbler/qingxi-character-workshop.git` |
| 开发分支 | `feature/character-workshop`（含完整 `character-workshop/` 应用） |
| 默认分支 | `main`（早期文档提交；**云上请 checkout feature 分支**） |

---

## ⭐ 重来 / 第一次上云：请只跟这一份

环境弄乱、Destroy 重开、或第一次配置时，**不要从下面零散章节拼步骤**。

→ 打开并严格按顺序执行：

**[`character-workshop/docs/Radeon-Cloud从零正确开工.md`](../../.worktrees/character-workshop/character-workshop/docs/Radeon-Cloud从零正确开工.md)**  

仓库内路径（clone 后）：

```text
qingxi-character-workshop/character-workshop/docs/Radeon-Cloud从零正确开工.md
```

**核心防坑顺序（背下来）：**

```text
ROCm 镜像 Launch
  → rocm-smi 有 GPU
  → python 中 torch 已是 ROCm 且 available=True   （否则只装 ROCm torch）
  → git clone + feature/character-workshop
  → bash scripts/install_app_deps.sh              （禁止先 pip install -r 乱装）
  → detect_device 通过
  → download_models → 出卡 → app.py
```

**绝对禁止：** 未确认 ROCm torch 前执行 `pip install -r requirements.txt` / `pip install torch`（默认源会变成 `+cu*`）。

以下阶段 A～K 为分项参考；**与「从零正确开工」冲突时，以从零文档为准。**

---

## 阶段 A · 浏览器：开通实例（无 shell 命令）

### A1. 登录与权限

1. 打开 https://radeon-global.anruicloud.com/ → **Login with Email**
2. 确认已加入 AMD Developer Program（中国区：https://developer.amd.com.cn/）
3. 确认 Luma 黑客松已报名、云额度可用

### A2. 添加 Template

1. 右上角头像 → **Profile**
2. **My Templates** → **Add Template**
3. 建议填写：
   - **Title:** `qingxi-workshop`
   - **Container Image:** 选带 **PyTorch + ROCm** 的镜像（名称以平台列表为准）
   - **Storage:** **Persistent (PVC)** ← 必开，模型才不会随实例销毁丢失
   - 可选：**SSH Access** 打开（方便本机 VS Code / scp）
4. **Add Template** 保存

### A3. 启动

1. Templates 列表点 **Launch**
2. 等到 **Your workspace is ready (100%)**
3. 点 **Open Notebook** → 进入 JupyterLab

之后命令默认在 **JupyterLab → Terminal** 中执行。

---

## 阶段 B · 第一次进入：系统与 GPU 自检

```bash
# 身份与系统
whoami
uname -a
pwd
df -h
nvidia-smi 2>/dev/null || true
rocm-smi 2>/dev/null || true

# Python / Torch
python3 --version
python3 - <<'PY'
import torch
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
print("hip", getattr(torch.version, "hip", None))
if torch.cuda.is_available():
    print("device", torch.cuda.get_device_name(0))
    print("vram_gb", round(torch.cuda.get_device_properties(0).total_memory/1024**3, 2))
PY
```

**通过标准：**

- `cuda_available True`
- `hip` 非空 **或** 版本串含 `rocm`（ROCm 栈）
- 能打印 GPU 名与显存

失败：换 ROCm PyTorch 镜像，或按镜像文档重装 torch，**不要**先装 CUDA 版 torch。

---

## 阶段 C · 把项目弄上云

**推荐顺序：本机 push 到 Gitee/GitHub → 云上 `git clone`。**  
无远程时再用 C2 打包上传或 C3 scp。

### 方式 C0 · 本机：提交并推送到远程（云端之前先做）

本地开发在 worktree 分支 `feature/character-workshop`：

```text
D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop
```

#### C0.1 确认状态并提交未保存改动

```powershell
cd "D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop"

git status -sb
git branch --show-current
# 应显示 feature/character-workshop

# 如有未提交文件：
git add character-workshop
git status
git commit -m "chore: sync workshop code for Radeon Cloud"
```

#### C0.2 远程仓库状态（本项目已完成）

| 项 | 值 |
|----|-----|
| 平台 | **GitHub** |
| 仓库 | https://github.com/Marsbler/qingxi-character-workshop |
| HTTPS | `https://github.com/Marsbler/qingxi-character-workshop.git` |
| 分支 | `feature/character-workshop`（应用代码）、`main`（基础文档） |

若需在**新机器**重建 remote（已有本地仓时）：

```powershell
cd "D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop"
git remote remove origin 2>$null
git remote add origin https://github.com/Marsbler/qingxi-character-workshop.git
git remote -v
```

#### C0.3 本机再次 push（改代码后）

```powershell
cd "D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop"

git add -A
git status
git commit -m "feat: describe your change"
# 首次设置 upstream（若尚未设置）：
git push -u origin feature/character-workshop
# 之后：
git push origin feature/character-workshop
```

认证：GitHub 用户名 + **Personal Access Token**（`repo` 权限）。**不要**把 token 写进任何会 `git add` 的文件。

验证浏览器打开：

https://github.com/Marsbler/qingxi-character-workshop/tree/feature/character-workshop/character-workshop  

应能看到 `app.py`、`src/`、`configs/`。

#### C0.4 远程已用 README 初始化导致 push 被拒时

```powershell
git pull origin master --allow-unrelated-histories
# 或: git pull origin main --allow-unrelated-histories
# 解决冲突后：
git push -u origin feature/character-workshop
git push -u origin main
```

#### C0.5 以后改完代码再同步到云

```powershell
cd "D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop"
git add -A
git status
git commit -m "feat: describe your change"
git push origin feature/character-workshop
```

云上已有克隆时，在 Jupyter Terminal：

```bash
cd ~/AMDAIHackathon   # 你的克隆目录
git fetch origin
git checkout feature/character-workshop
git pull origin feature/character-workshop
cd character-workshop
```

---

### 方式 C1 · 云上：Git clone（推荐）

在 **Radeon Cloud JupyterLab → Terminal** 执行（公开仓库，一般无需 token）。  
你当前工作目录是 `/workspace`，可直接在该目录克隆。

#### C1.0 若报错：`server certificate verification failed. CAfile: none`

容器里缺 CA 证书。**先修证书，再 clone**（按顺序试）：

**方案 1 — 安装 CA（优先）**

```bash
# Debian/Ubuntu 镜像
sudo apt-get update
sudo apt-get install -y ca-certificates
sudo update-ca-certificates

# 再试
export REPO_URL="https://github.com/Marsbler/qingxi-character-workshop.git"
git clone "$REPO_URL" qingxi-character-workshop
```

**方案 2 — 临时关闭 Git SSL 校验（仅当前 shell，能下代码即可）**

```bash
export GIT_SSL_NO_VERIFY=1
# 或: git config --global http.sslVerify false

export REPO_URL="https://github.com/Marsbler/qingxi-character-workshop.git"
cd /workspace
git clone "$REPO_URL" qingxi-character-workshop
```

> 方案 2 降低安全性，**只用于黑客松临时环境拉公开代码**。拉完可恢复：  
> `git config --global --unset http.sslVerify` 或新开 Terminal（未 export 则不影响）。

**方案 3 — 用 ghproxy / 镜像（GitHub 直连 SSL 仍失败时）**

```bash
export GIT_SSL_NO_VERIFY=1
git clone https://ghproxy.com/https://github.com/Marsbler/qingxi-character-workshop.git qingxi-character-workshop
# 若 ghproxy 不可用，可换其他 GitHub 代理，或改用下方 C2 打包上传
```

**方案 4 — 仍失败：改用 C2 本机 tar 上传**（不依赖云上访问 GitHub）。

---

#### C1.1 正常克隆步骤

```bash
# 你的环境示例：/workspace
cd /workspace
# 或: cd ~

export REPO_URL="https://github.com/Marsbler/qingxi-character-workshop.git"
# 若仍有证书问题，先保留：
# export GIT_SSL_NO_VERIFY=1

# 首次克隆
git clone "$REPO_URL" qingxi-character-workshop
cd qingxi-character-workshop

# 必须检出开发分支（main 上可能没有完整应用代码）
git fetch origin
git checkout feature/character-workshop
git pull origin feature/character-workshop

cd character-workshop
pwd
ls -la app.py configs src README.md

# 确认
test -f app.py && echo "PROJECT OK"
```

之后在本机 `git push` 新提交后，云上更新：

```bash
cd /workspace/qingxi-character-workshop
# 若仍有证书问题：export GIT_SSL_NO_VERIFY=1
git checkout feature/character-workshop
git pull origin feature/character-workshop
cd character-workshop
```

若仓库改为 **private**，克隆需 PAT（**勿写入会提交的文件**）：

```bash
export GIT_SSL_NO_VERIFY=1   # 仅当仍有证书错误时
git clone "https://<GitHub用户名>:<PAT>@github.com/Marsbler/qingxi-character-workshop.git" qingxi-character-workshop
```

---

### 方式 C2 · 本机打包上传（无远程 / Git 不通时）

**本机 PowerShell：**

```powershell
cd "D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop"
# 排除大目录与缓存
tar -czf "$env:TEMP\qingxi-workshop.tgz" `
  --exclude=character-workshop/.venv `
  --exclude=character-workshop/outputs `
  --exclude=character-workshop/models `
  --exclude=character-workshop/.pytest_cache `
  character-workshop
Write-Host "Archive: $env:TEMP\qingxi-workshop.tgz"
```

JupyterLab 左侧 **Upload** 上传 `qingxi-workshop.tgz`，云上：

```bash
cd ~
mkdir -p work && cd work
# 若上传到 $HOME：
tar -xzf ~/qingxi-workshop.tgz
# 或: tar -xzf /path/to/qingxi-workshop.tgz
cd character-workshop
ls -la app.py
```

### 方式 C3 · SSH + scp（Template 已开 SSH）

Profile 里粘贴本机公钥后 Launch，复制页面上的：

```bash
ssh <user>@<host> -p <port>
```

本机：

```powershell
scp -P <port> -r "D:\APP\Obsidian\Documents\Marbler\主业副业\AMDAIHackathon\.worktrees\character-workshop\character-workshop" <user>@<host>:~/character-workshop
```

---

## 阶段 D · Python 依赖

> **完整防坑步骤见：`docs/Radeon-Cloud从零正确开工.md` 步骤 3～7。**  
> 下面是摘要；冲突时以从零文档为准。

### 唯一推荐装依赖命令

```bash
cd /workspace/qingxi-character-workshop/character-workshop

# 门禁：必须先 ROCm torch available=True
python3 -c "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.hip)"

# 安全脚本（preflight + requirements + accelerate --no-deps + postflight）
bash scripts/install_app_deps.sh

export MOCK=0
python3 -c "from src.device import detect_device; d=detect_device(); print(d); assert d.device_type=='cuda'"
```

### 禁止

```bash
pip install torch                          # 默认源 = 常变成 +cu*
pip install -r requirements.txt            # 在 torch 未就绪时禁止
pip install accelerate                     # 禁止不带 --no-deps
```

---

## 阶段 E · 下载模型（耗时长、占 PVC）

```bash
cd ~/work/character-workshop   # 改成你的路径
export MOCK=0

# Hugging Face 网络不稳时可设镜像（若环境允许）
# export HF_ENDPOINT=https://hf-mirror.com

bash scripts/download_models.sh
```

脚本会拉（见 `configs/models.yaml`）：

- LLM: `Qwen/Qwen2.5-7B-Instruct`
- Image: `cagliostrolab/animagine-xl-3.1`

**显存紧（约 16GB）时** 可先改小 LLM，编辑 `configs/models.yaml`：

```yaml
llm:
  model_id: Qwen/Qwen2.5-3B-Instruct
```

再只下 LLM：

```bash
python3 - <<'PY'
from huggingface_hub import snapshot_download
snapshot_download("Qwen/Qwen2.5-3B-Instruct")
print("llm ok")
PY
```

检查缓存体积：

```bash
du -sh ~/.cache/huggingface 2>/dev/null || du -sh $HOME/.cache/huggingface
```

---

## 阶段 F · 分步冒烟（先于 Web）

始终在项目根目录，`export MOCK=0`。

### F1. MOCK 回归（确认代码未坏）

```bash
export MOCK=1
python3 -m pytest -q
python3 scripts/run_eval_json.py
export MOCK=0
```

### F2. 仅 LLM → JSON

```bash
export MOCK=0
python3 - <<'PY'
from src.llm_role import generate_character
card = generate_character("青衣空系旅人，背着纸灯笼", affinity_pref="空", mock=False)
print(card.name, card.primary_affinity)
print(card.image_prompt[:200])
print("LLM OK")
PY
```

**失败：** OOM → 换 3B；JSON 坏 → 看报错，检查 prompts；权限/下载 → 重跑 download。

### F3. 仅出图（耗显存）

```bash
export MOCK=0
python3 - <<'PY'
from pathlib import Path
from src.llm_role import generate_character
from src.image_gen import generate_portrait, unload_image_models
card = generate_character("红发质系武士", affinity_pref="质", mock=False)
out = Path("outputs/_smoke_portrait.png")
out.parent.mkdir(exist_ok=True)
p = generate_portrait(card, out, mock=False)
print("portrait", p, p.stat().st_size)
unload_image_models()
print("IMAGE OK")
PY
```

**OOM：** 在 `configs/models.yaml` 的 `image` 中降 `width/height`（如 768×1024）、`steps: 24` 后重试。

### F4. 完整出卡（P0 硬门槛）

```bash
export MOCK=0
python3 - <<'PY'
from src.orchestrator import generate_card_job, JobState
r = generate_card_job(
    user_text="青衣空系旅人，背着纸灯笼，寡言",
    affinity_pref="空",
    mock=False,
)
print(r.state, r.job_id, r.error)
print(r.card_path, r.portrait_path)
assert r.state == JobState.READY
assert r.card_path and r.card_path.exists()
print("CARD E2E OK")
PY
```

下载/预览：Jupyter 文件树打开 `outputs/<job_id>/card.png`。

### F5. 修订路径

```bash
export MOCK=0
python3 - <<'PY'
from src.orchestrator import generate_card_job, revise_job, JobState
base = generate_card_job("黑发念系少年", affinity_pref="念", mock=False)
assert base.state == JobState.READY
r = revise_job(base.job_id, "改成金色短发", mock=False)
print(r.state, r.error)
assert r.state == JobState.READY
print("REVISE OK", r.card_path)
PY
```

### F6. 视频（P1，允许失败）

```bash
export MOCK=0
python3 - <<'PY'
from src.orchestrator import generate_card_job, animate_job, JobState
base = generate_card_job("空系纸灯笼旅人", affinity_pref="空", mock=False)
assert base.state == JobState.READY
r = animate_job(base.job_id, mock=False)
print("state", r.state, "video", r.video_path, "err", r.error)
# 成功或软失败都可；卡片必须仍在
assert base.card_path.exists() or (r.card_path and r.card_path.exists())
print("ANIMATE DONE (check video or soft-fail)")
PY
```

若 `NotImplementedError` / ROCm 不支持：

```bash
# 临时关视频，保 P0
# 编辑 configs/models.yaml → video.enabled: false
```

真要接 I2V：在云上按 Diffusers/CogVideoX 文档改 `src/video_gen.py` 的 `_cogvideox`，再重跑 F6。

---

## 阶段 G · 启动 Web UI

```bash
cd ~/work/character-workshop
export MOCK=0
export PORT=7860

# 后台跑（可选）
# nohup python3 app.py > /tmp/qingxi-app.log 2>&1 &

python3 app.py
```

### G1. 仅 Notebook 内访问

浏览器打开 Jupyter 提供的代理 URL，或平台说明的 port 映射。  
若只能本机端口：在 Terminal 确认：

```bash
curl -s -o /dev/null -w "%{http_code}\n" http://127.0.0.1:7860/
```

期望 `200` 或 `302`。

### G2. 公网 tunnel（录屏/外链演示）

```bash
# 官方 rc-tunnel（Notebook 内）
/var/run/secrets/frp-self-service/install
export PATH="$HOME/.local/bin:$PATH"
rc-tunnel version

# app 必须已在 127.0.0.1:7860 监听
rc-tunnel expose --port 7860
# 记下 https://rc-xxxx.radeon.firstdg.ai

rc-tunnel status
# 结束：
# rc-tunnel stop
```

**注意：** 公网 URL 勿长时间裸奔；演示完 `stop` 并 **Destroy Instance** 省额度。

### G3. UI 真机检查清单

- [ ] 中文设定 → 生成角色卡 → 真立绘（非 MOCK 色块字）
- [ ] 下载/查看 `card.png`
- [ ] 参考图可选测一张
- [ ] 改一版
- [ ] 能力动画：成功或友好失败且卡仍在
- [ ] 顶栏 device 非 mock

---

## 阶段 H · 演示录制素材命令

```bash
# 录屏前另开 Terminal
rocm-smi
# 或
python3 - <<'PY'
from src.device import detect_device
print(detect_device())
PY
```

建议录屏顺序：

1. `rocm-smi` / device 打印
2. `export MOCK=0 && python3 app.py`
3. UI 生成卡（case01）
4. 改一版或参考图
5. 动画或说明 P1 限制
6. 展示 `outputs/.../card.png`

---

## 阶段 I · 省额度与持久化

```bash
# 工作结束前确认重要输出在 PVC 路径
pwd
ls outputs/

# 停止 tunnel
rc-tunnel stop 2>/dev/null || true

# 停 app：Ctrl+C 或
pkill -f "python3 app.py" 2>/dev/null || true
```

浏览器：**Profile → Active Instance → Destroy Instance**  
下次 Launch 同一 Template + PVC，代码与 HF 缓存应仍在（视平台 PVC 挂载点而定）。

---

## 阶段 J · 故障速查

| 现象 | 命令/处理 |
|------|-----------|
| `cuda_available False` | 换 ROCm 镜像；勿装 NVIDIA 轮子 |
| HF 下载超时 | `HF_ENDPOINT` 镜像；重试 `download_models.sh` |
| LLM/出图 OOM | 3B LLM；降分辨率/步数；`unload` 后分阶段跑 |
| `No module named src` | `cd` 到含 `app.py` 的目录 |
| Gradio 打不开 | `curl 127.0.0.1:7860`；检查 `PORT`；用 `rc-tunnel` |
| 仍是 MOCK 图 | `echo $MOCK` 必须为空或 `0`；重启 app |
| 视频永远 NotImplemented | 预期直至接线；`video.enabled: false` |
| 额度没了 | Destroy；本地 MOCK 继续改代码 |

---

## 阶段 K · 最小「今日完成」路径（压缩版）

```bash
# 1) 自检 GPU
python3 -c "import torch; print(torch.cuda.is_available(), getattr(torch.version,'hip',None))"

# 2) 进项目、装依赖
cd ~/work/character-workshop
pip install -r requirements.txt

# 3) 下模型
bash scripts/download_models.sh

# 4) 真机出卡
export MOCK=0
python3 -c "from src.orchestrator import generate_card_job,JobState as S;r=generate_card_job('青衣空系旅人',affinity_pref='空',mock=False);print(r.state,r.card_path);assert r.state==S.READY"

# 5) Web
python3 app.py
# 可选: rc-tunnel expose --port 7860
```

**今日成功定义：** 步骤 4 打出 `READY` 且 `card.png` 为真实立绘合成卡。

---

## 与提交的关系（Task 14 预告）

真机跑通后准备：

- 演示视频 3–5 分钟
- 英文 PDF（架构 + ROCm 适配）
- PPT/海报
- PR：`Track 1, <Team>, Qingxi Character Workshop`

---

## 相关文档

- 本地 MOCK 测试：`docs/本地测试指南.md`
- 英文 README：`README.md`
- 设计规格：仓库 `docs/superpowers/specs/2026-07-26-amd-hackathon-character-workshop-design.md`
- 实现计划 T13：`docs/superpowers/plans/2026-07-26-character-workshop-implementation.md`
