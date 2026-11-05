# RoboDuet 对齐版 — 训练/评估代码 + 视频生成

> 「对齐版」= 在 V8 基线上补上 **域随机化（DR）+ 观测历史 + 非对称 Actor-Critic** 三项对齐后的策略。
> 论文口径下精度略降，但稳定性明显改善。

这是 [RoboDuet-Reproduction](https://github.com/Loober107/RoboDuet-Reproduction) 里
「对齐版」这一支的独立打包：**只要代码 + 视频生成脚本**，不含复现报告。

---

## 目录

```
├── code/                  对齐版代码
│   ├── train_hist.py          对齐版训练（AsymActorCritic：历史编码器 + 非对称 critic）
│   ├── train_stage1.py        基座网络定义（ActorCritic，被视频脚本依赖）
│   ├── eval_hist.py           对齐版评估
│   ├── eval_robust.py         DR 鲁棒性评估（V8 网络）
│   ├── eval_robust_hist.py    DR 鲁棒性评估（对齐版网络）
│   ├── model_io.py            模型载入（规避 MuJoCo 中文路径问题）
│   ├── envs/
│   │   ├── go1_env.py         向量化环境：50 Hz 策略 / 200 Hz 物理
│   │   └── go1_env_dr.py      域随机化子类
│   └── models/
│       ├── arx5p2Go1.xml      MuJoCo 模型（实际使用：18 执行器 / 27 qpos / 26 dof）
│       ├── build_mjcf.py      从官方 URDF 生成上述模型（质量/惯量/限位均取原文件）
│       ├── go1_arm.xml        早期 6 DoF 替身版（18 执行器 / 25 qpos / 24 dof）
│       ├── build_model.py     从 Menagerie Go1 + 6 DoF 臂拼装出替身版
│       └── vendor/go1_menagerie.xml
├── video_gen/
│   └── make_video.py      视频生成（MuJoCo 离屏渲染 + 逐帧读数叠加）
├── video/                 已生成的 5 段视频
├── checkpoints/
│   ├── ckpt_600.pt        对齐版权重（第 600 次训练迭代的快照）
│   └── v8_ckpt_100.pt     V8 基线权重（对比用）
└── README.md
```

---

## 跑起来

依赖：**Python 3.12 + mujoco 3.x + torch + numpy + pillow + glfw + imageio**

```bash
pip install mujoco torch numpy pillow glfw "imageio[ffmpeg]"
```

> `imageio` 只用来写 mp4。没装也能跑，但只会输出 `.gif`——脚本里写 mp4 那段有
> `try/except`，失败不报错，只打印「MP4 跳过」。

> 原项目为了绕开 Windows 上的 pip 文件锁，把依赖解压到 `.pylibs/` 并通过
> `PYTHONPATH` 注入。用普通 `pip install` 也一样，只要 `import mujoco` 能过。

### 生成视频

```bash
cd video_gen
python make_video.py
```

会在 `../video/` 下重新生成 **5 段视频**（每段同时输出 `.mp4` 和 `.gif`）：

| 文件 | 内容 |
|---|---|
| `video_align_0.5.mp4` | 对齐版 @ vx = +0.5 m/s |
| `video_align_1.0.mp4` | 对齐版 @ vx = +1.0 m/s（高速档，仍未追上指令） |
| `video_align_stand.mp4` | 对齐版 @ vx = 0（零速锁时钟，四足着地） |
| `video_v8_1.0.mp4` | V8 基线 @ +1.0 m/s（能加速到约 0.9，但频繁摔倒） |
| `video_compare_1.0.mp4` | V8（左）vs 对齐版（右），同指令并排对比 |

视频里的画面设计：地面每 0.5 m 一个标记、相机跟随基座、每帧叠加指令/实测速度/位移/时间读数。

### 训练

```bash
cd code
python train_hist.py
```

权重默认写到 `code/runs/<tag>_<时间戳>/`，每次迭代存一个 `ckpt_<iter>.pt`，
结束再存 `final.pt` 和 `summary.json`。评估用 `eval_hist.py`。**注意**：两个脚本里的
checkpoint 路径按原机器目录硬编码过（`runs/hist_dr_*/`），换机器需自己改。

---

## 对齐版到底对齐了什么

| 项 | V8 基线 | 对齐版 |
|---|---|---|
| 域随机化 | 无 | 摩擦 U[0.1, 3.0]、弹性 [0, 0.4]、负载 ±2 kg、电机强度 [0.9, 1.1]、电机零位 ±0.02、重力 ±1 |
| 观测历史 | 单帧（63 维） | **30 帧堆叠**（历史编码器） |
| Actor-Critic | 对称 | **非对称**（critic 用特权信息） |
| 速度跟踪精度 | 较高 | **略降**（精度换稳定） |
| 抗扰动 / 鲁棒性 | 较弱 | **明显改善** |

> 注意维度：模型有 **18 个执行器**，但 Stage 1 的策略只输出 **12 维**（四条腿），
> 机械臂锁在默认位姿 `[0, 0.8, 0.8, 0, 0, 0]` 不动。`NUM_ACT = 12`，别和 18 混了。

即报告第 5.4 节说的「**精度换稳定**」。完整数据见
[复现报告](https://github.com/Loober107/RoboDuet-Reproduction/blob/main/report/RoboDuet_%E5%A4%8D%E7%8E%B0%E6%8A%A5%E5%91%8A_V8_%E4%BF%AE%E6%AD%A3.docx)。

---

## 改动说明（相对原目录）

唯一改动是 `video_gen/make_video.py` 的路径常量，让它能在这个新目录结构下找到依赖：

```python
HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # 仓库根目录
sys.path.insert(0, str(ROOT / "code"))   # 依赖 code/ 下的模块

RES       = ROOT / "video"
MODEL_XML = ROOT / "code" / "models" / "arx5p2Go1.xml"
V8_CKPT   = ROOT / "checkpoints" / "v8_ckpt_100.pt"
AL_CKPT   = ROOT / "checkpoints" / "ckpt_600.pt"
```

并把 `torch.load(HERE / ckpt_path)` 改成 `torch.load(pathlib.Path(ckpt_path))`。
算法本身一行没动。

---

## 参考

- 论文：[arXiv:2403.17367](https://arxiv.org/abs/2403.17367)
- 官方实现：[locomanip-duet/RoboDuet](https://github.com/locomanip-duet/RoboDuet)
- 完整复现（报告 + 全部代码）：[Loober107/RoboDuet-Reproduction](https://github.com/Loober107/RoboDuet-Reproduction)

个人学习性质，非官方实现。论文与官方代码版权归原作者所有。
