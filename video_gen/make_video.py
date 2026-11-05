"""Render MuJoCo motion clips of the two policies.

Visual design:
  * a row of ground markers every 0.5 m plus a centre line, so forward travel is
    legible even though the camera follows the robot;
  * the camera follows the base at a fixed distance, keeping the robot large;
  * each frame carries a live read-out: command, measured vx, displacement, time.

Outputs (video/):
    video_align_0.5.mp4/.gif      aligned model at 0.5 m/s
    video_align_1.0.mp4/.gif      aligned model at 1.0 m/s
    video_v8_1.0.mp4/.gif         V8 baseline at 1.0 m/s
    video_align_stand.mp4/.gif    aligned model, zero command
    video_compare_1.0.mp4/.gif    V8 vs aligned, side by side

Run:  python make_video.py
"""
import pathlib
import sys

import numpy as np
import torch

HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent                       # 仓库根目录
sys.path.insert(0, str(ROOT / "code"))   # 依赖 code/ 下的环境、模型与网络定义

import mujoco                                              # noqa: E402
from PIL import Image, ImageDraw, ImageFont                # noqa: E402

from model_io import _in                                   # noqa: E402
from envs.go1_env import Go1Env, NUM_OBS, NUM_ACT          # noqa: E402
from train_stage1 import ActorCritic                       # noqa: E402
from train_hist import AsymActorCritic                     # noqa: E402

RES = ROOT / "video"
MODEL_XML = ROOT / "code" / "models" / "arx5p2Go1.xml"
W, H = 720, 540
FPS = 25
STEPS = 250                     # 5 s at 50 Hz
FONT_PATH = "C:/Windows/Fonts/msyh.ttc"

V8_CKPT = ROOT / "checkpoints" / "v8_ckpt_100.pt"
AL_CKPT = ROOT / "checkpoints" / "ckpt_600.pt"


def font(size):
    try:
        return ImageFont.truetype(FONT_PATH, size)
    except Exception:
        return ImageFont.load_default()


def load_marked_model():
    """Compile the robot plus visual-only ground markers.

    The markers are collision-free (contype/conaffinity = 0) so they cannot
    change the dynamics — they exist purely so the eye can see translation.
    """
    xml = MODEL_XML.read_text(encoding="utf-8")
    marks = []
    for i in range(-4, 17):
        marks.append(
            f'<geom name="mk{i}" type="box" size="0.025 0.20 0.002" '
            f'pos="{i * 0.5} 0 0.002" rgba="0.22 0.52 0.32 1" '
            f'contype="0" conaffinity="0"/>')
    marks.append(
        '<geom name="mk_line" type="box" size="8.0 0.012 0.002" pos="3.0 0 0.002" '
        'rgba="0.20 0.30 0.55 1" contype="0" conaffinity="0"/>')
    xml = xml.replace("</worldbody>", "\n".join(marks) + "\n</worldbody>")
    with _in(MODEL_XML.parent):
        m = mujoco.MjModel.from_xml_string(xml)
    m.vis.global_.offwidth = 1280
    m.vis.global_.offheight = 1280
    m.vis.headlight.ambient[:] = [0.55] * 3
    m.vis.headlight.diffuse[:] = [0.75] * 3
    m.vis.headlight.specular[:] = [0.28] * 3
    return m


class Rig:
    """One robot in one MuJoCo world, driven by a policy."""

    def __init__(self, model, kind, ckpt_path, vx, seed=5):
        self.env = Go1Env(model, 1, num_threads=1, seed=seed)
        self.env.reset(range(1))
        # reset randomises yaw; pin it so body +x == world +x and the robot
        # visibly walks towards the camera-right instead of off to one side
        d = self.env.data[0]
        d.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        d.qvel[0:6] = 0.0
        mujoco.mj_forward(self.env.models[0], d)
        self.env.cmd[:] = [vx, 0.0, 0.0]
        self.kind, self.vx = kind, vx
        self.falls = 0
        self.net = ActorCritic(NUM_OBS, NUM_ACT) if kind == "v8" else AsymActorCritic()
        ck = torch.load(pathlib.Path(ckpt_path), map_location="cpu", weights_only=False)
        self.net.load_state_dict(ck["model"])
        self.net.eval()

    def step(self):
        if self.kind == "v8":
            with torch.no_grad():
                mu, _ = self.net(torch.as_tensor(self.env.obs(), dtype=torch.float32))
        else:
            hist = torch.as_tensor(self.env.obs_hist, dtype=torch.float32)
            with torch.no_grad():
                mu = self.net.mean_action(hist)
        _, _, done, _ = self.env.step(mu.numpy())
        if done.any():
            self.falls += int(done.sum())
            self.env.reset(np.where(done)[0])
            self.env.cmd[:] = [self.vx, 0.0, 0.0]

    @property
    def data(self):
        return self.env.data[0]

    @property
    def vx_now(self):
        return float(self.env.base_lin_vel[0, 0])

    def close(self):
        self.env.close()


def follow_camera(d, distance=2.9, azimuth=90, elevation=-12):
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [float(d.qpos[0]), float(d.qpos[1]), 0.26]
    cam.distance = distance
    cam.elevation = elevation
    cam.azimuth = azimuth
    return cam


def capture(rigs, renderer, steps=STEPS):
    frames = [[] for _ in rigs]
    tele = [[] for _ in rigs]
    for k in range(steps):
        for r in rigs:
            r.step()
        if k % 2 == 0:
            for i, r in enumerate(rigs):
                renderer.update_scene(r.data, camera=follow_camera(r.data))
                frames[i].append(Image.fromarray(renderer.render()))
                tele[i].append((r.vx_now, float(r.data.qpos[0]), k * 0.02,
                                r.falls))
    return frames, tele


def overlay(img, title, subtitle, bottom, accent=(255, 255, 255)):
    d = ImageDraw.Draw(img, "RGBA")
    d.rectangle([0, 0, img.width, 62], fill=(15, 20, 30, 170))
    d.text((14, 9), title, font=font(23), fill=accent)
    d.text((14, 38), subtitle, font=font(14), fill=(205, 212, 225))
    d.rectangle([0, img.height - 42, img.width, img.height], fill=(15, 20, 30, 170))
    d.text((14, img.height - 33), bottom, font=font(17), fill=(235, 240, 250))
    return img


def save_clip(frames, name, title, subtitle, texts, accent=(255, 255, 255)):
    RES.mkdir(parents=True, exist_ok=True)
    tagged = [overlay(f.copy(), title, subtitle, s, accent=accent)
              for f, s in zip(frames, texts)]

    gif = RES / f"{name}.gif"
    tagged[0].save(gif, save_all=True, append_images=tagged[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"  {gif.name}  ({len(tagged)} 帧)")

    try:
        import imageio.v2 as iio
        mp4 = RES / f"{name}.mp4"
        iio.mimsave(mp4, [np.asarray(f) for f in tagged], fps=FPS,
                    quality=8, macro_block_size=1)
        print(f"  {mp4.name}")
    except Exception as e:
        print(f"  （MP4 跳过：{type(e).__name__}: {e}）")


def single(model, renderer, kind, ckpt, vx, name, title, subtitle, accent):
    rig = Rig(model, kind, ckpt, vx)
    frames, tele = capture([rig], renderer)
    rig.close()
    texts = [f"指令 vx = {vx:+.1f} m/s　实测 {v:+.2f} m/s　"
             f"位移 {x:+.2f} m　摔倒 {f} 次　t = {t:.1f} s"
             for v, x, t, f in tele[0]]
    save_clip(frames[0], name, title, subtitle, texts, accent=accent)


def write_plain(frames, name):
    """Write frames as-is (they already carry their own layout)."""
    RES.mkdir(parents=True, exist_ok=True)
    gif = RES / f"{name}.gif"
    frames[0].save(gif, save_all=True, append_images=frames[1:],
                   duration=int(1000 / FPS), loop=0, optimize=True)
    print(f"  {gif.name}  ({len(frames)} 帧)")
    try:
        import imageio.v2 as iio
        mp4 = RES / f"{name}.mp4"
        iio.mimsave(mp4, [np.asarray(f) for f in frames], fps=FPS,
                    quality=8, macro_block_size=1)
        print(f"  {mp4.name}")
    except Exception as e:
        print(f"  （MP4 跳过：{type(e).__name__}: {e}）")


def compare(model, renderer):
    """V8 (left) vs aligned (right), same command, fixed side-by-side framing."""
    a = Rig(model, "v8", V8_CKPT, 1.0)
    b = Rig(model, "align", AL_CKPT, 1.0)
    frames, tele = capture([a, b], renderer)
    a.close()
    b.close()

    half = ((W - 10) // 2) // 2 * 2          # even, for H.264
    tile_h = (int(H * half / W)) // 2 * 2
    head, foot = 34, 46
    canvas_frames = []
    for la, lb in zip(frames[0], frames[1]):
        la = la.resize((half, tile_h))
        lb = lb.resize((half, tile_h))
        canvas = Image.new("RGB", (W, head + tile_h + foot), (12, 16, 24))
        canvas.paste(la, (2, head))
        canvas.paste(lb, (half + 8, head))
        d = ImageDraw.Draw(canvas)
        d.text((10, 6), "V8（无 DR / 无历史）", font=font(19), fill=(130, 200, 255))
        d.text((half + 16, 6), "对齐版（DR + 历史 + 非对称）", font=font(19),
               fill=(255, 150, 150))
        canvas_frames.append(canvas)

    for (va, xa, tt, fa), (vb, xb, _, fb), canvas in zip(tele[0], tele[1],
                                                         canvas_frames):
        d = ImageDraw.Draw(canvas)
        d.rectangle([0, head + tile_h, W, canvas.height], fill=(12, 16, 24))
        d.text((12, head + tile_h + 4),
               f"指令 vx = +1.0 m/s　t = {tt:.1f} s",
               font=font(17), fill=(235, 240, 250))
        d.text((12, head + tile_h + 24),
               f"左 V8：实测 {va:+.2f} m/s，摔倒 {fa} 次　　"
               f"右 对齐版：实测 {vb:+.2f} m/s，摔倒 {fb} 次",
               font=font(15), fill=(200, 210, 225))

    write_plain(canvas_frames, "video_compare_1.0")


def main():
    model = load_marked_model()
    renderer = mujoco.Renderer(model, height=H, width=W)
    RES.mkdir(parents=True, exist_ok=True)

    print("对齐版 0.5 m/s ...")
    single(model, renderer, "align", AL_CKPT, 0.5, "video_align_0.5",
           "对齐版　指令 vx = +0.5 m/s",
           "DR + 30 帧观测历史 + 非对称 actor-critic", (150, 220, 255))

    print("对齐版 1.0 m/s ...")
    single(model, renderer, "align", AL_CKPT, 1.0, "video_align_1.0",
           "对齐版　指令 vx = +1.0 m/s", "同一策略的高速档", (150, 220, 255))

    print("V8 1.0 m/s（对照）...")
    single(model, renderer, "v8", V8_CKPT, 1.0, "video_v8_1.0",
           "V8　指令 vx = +1.0 m/s", "能加速到约 0.9，但频繁摔倒", (255, 170, 120))

    print("对齐版站立 ...")
    single(model, renderer, "align", AL_CKPT, 0.0, "video_align_stand",
           "对齐版　指令 vx = 0（静止）", "零速锁时钟生效，四足全部着地",
           (180, 255, 190))

    print("对比视频 ...")
    compare(model, renderer)

    renderer.close()
    print("\n输出目录:", RES)


if __name__ == "__main__":
    main()
