# MuJoCo 板球平衡：圆周轨迹与任意形状（Fourier 一笔画）

用 MuJoCo 搭建一个**真实接触动力学**的板球系统：球是带 free joint 的刚体，完全被动；板由两个铰链关节做俯仰/滚转，两个电机输出力矩。控制器根据球在板面内的位置/速度，解算需要的板倾角。

在这个基础模型上实现了两类目标：

1. 从板中心平滑起步，稳定跟踪半径 0.16 m、周期 4 s 的圆周；
2. 把任意闭合一笔画分解成傅里叶描述子，让球在板上画出 **Joseph Fourier 版画的单笔剪影**；
3. 提供 `auto_draw.py`：输入**任意图片**，自动提取主轮廓并驱动 MuJoCo 完成绘图。

![preview](outputs/preview.png)

- 完整视频：[`outputs/ball_on_plate_circle.mp4`](outputs/ball_on_plate_circle.mp4)（20 s，900×900，60 fps）
- 循环预览：[`outputs/preview.gif`](outputs/preview.gif)（8 s，540×540）
- 结果曲线：[`outputs/tracking.png`](outputs/tracking.png)

## 项目背景

这是我在 **DeepSeek Harness** 上完成的第一个项目。它不只是生成了一段仿真代码，而是从 MuJoCo 的 MJCF 建模、接触与滚动约束、控制算法推导，一路做到可视化、指标评估和回归测试。最终球在真实接触仿真中达到了约 1 mm 的圆周跟踪误差，并进一步扩展到用傅里叶描述子画出一幅 Fourier 一笔画，也让我直观感受到 DeepSeek Harness 对物理仿真器这类任务的理解能力。

## 圆周跟踪

稳定段（12.4–20 s）的跟踪结果：

| 指标 | 数值 |
|---|---:|
| 圆周半径 | 0.160 m |
| 位置 RMS 误差 | **1.02 mm**（半径的 0.64%） |
| 位置最大误差 | 1.03 mm |
| 径向 / 切向 RMS | 0.49 / 0.90 mm |
| 相位滞后 | 0.32° |
| 最大板倾角 | 3.56° |
| 最大关节力矩 | 0.137 N·m（电机限幅 ±6 N·m） |
| 接触滑移速度 RMS | 0.79 mm/s |
| 接触法向力 | 约 0.44 N（球重 0.045 kg） |

> 这里的误差是 MuJoCo 中球的实际位置与参考位置之差。球没有被“绑定”到轨迹上：所有运动都来自板面接触、摩擦和滚动约束。

## 任意形状：Fourier 一笔画

同一套物理模型和控制器可以直接跟踪任意闭合一笔画。示例使用公有领域的 Fourier 版画 **“Joseph Fourier”, A. F. B. Geille after J. Boilly**（Wikimedia Commons），用 [`extract_portrait.py`](extract_portrait.py) 提取人物外轮廓，再用 40 个傅里叶谐波重建，让球沿轮廓画出来。

> **关于“哪个 Fourier 头像”**：用的是下面这张公有领域版画，来源为 [Wikimedia Commons: File:Joseph Fourier.jpg](https://commons.wikimedia.org/wiki/File:Joseph_Fourier.jpg)。因为球的轨迹必须是一条连续闭合曲线，这里画的是**外轮廓剪影**，不包含版画内部的五官和排线；它比之前手工构造的轮廓有真实历史图像依据。

![source and reconstruction](outputs/fourier/portrait_reference.png)

左侧：原版画和红色提取轮廓；右侧：40 阶傅里叶重建与球的实际轨迹。

![Fourier portrait](outputs/fourier/preview.png)

- 4× 双视图视频：[`outputs/fourier/ball_on_plate_fourier.mp4`](outputs/fourier/ball_on_plate_fourier.mp4)（约 9.2 s，1800×900，60 fps）
- 4× 循环预览：[`outputs/fourier/preview.gif`](outputs/fourier/preview.gif)
- 跟踪曲线：[`outputs/fourier/tracking.png`](outputs/fourier/tracking.png)
- 傅里叶分析：[`outputs/fourier/fourier_analysis.png`](outputs/fourier/fourier_analysis.png)

视频左侧是俯视图（带参考轮廓和实际笔画），右侧是同时渲染的 3D 台子视图，可以直接看到板、球和台体的空间关系。物理时间约 34.4 s，视频按 4× 速度播放。

绘制结果：

| 指标 | 数值 |
|---|---:|
| 傅里叶谐波数 | 40 |
| 一笔画长度 | 1.421 m |
| 绘制时长 | 34.4 s |
| 位置 RMS 误差 | **0.42 mm** |
| 位置最大误差 | 1.95 mm |
| 95% 分位误差 | 0.79 mm |
| 最大板倾角 | 2.19° |
| 最大关节力矩 | 1.826 N·m |
| 接触滑移速度 RMS | 0.39 mm/s |
| 40 阶谐波能量占比 | 99.999% |

## 可复用管线：任意图片 → 自动画线

`auto_draw.py` 把整条链路封装成一条命令：

```bash
python auto_draw.py path/to/image.png
```

流程是：

1. **前景分割**：优先使用 PNG alpha；否则用图像边界估计背景色，自动选择前景极性；也可以用 `--mode threshold`、`--threshold`、`--mask` 手动指定；
2. **轮廓提取**：做相对尺度的形态学闭运算，把线稿/版画的断线连成实体，取最大外轮廓并做 RDP 简化；
3. **归一化与傅里叶拟合**：轮廓转成复数序列，FFT 后保留 K 个谐波；
4. **物理可执行化**：按曲率生成限速剖面，送入原有的串级板球控制器；
5. **输出**：4× 双视图视频、源图 + 轮廓图、对比图、跟踪曲线、指标和 `.npz` 日志。

示例（五角星）：

```bash
python auto_draw.py assets/demo_star.png --name "Star" --harmonics 24
```

![auto pipeline demo](outputs/auto/star/comparison.png)

- 五角星 4× 视频：[`outputs/auto/star/stroke.mp4`](outputs/auto/star/stroke.mp4)
- 五角星预览：[`outputs/auto/star/preview.gif`](outputs/auto/star/preview.gif)

输出目录默认是 `outputs/auto/<图片名>/`，包含：

| 文件 | 内容 |
|---|---|
| `source.png` | 处理后的输入图 |
| `mask.png` | 自动/手动得到的前景 mask |
| `source_contour.png` | 原图 + 红色提取轮廓 |
| `stroke.mp4` | 4× 双视图（左俯视 / 右 3D） |
| `comparison.png` | 原图、轮廓、傅里叶路径与球轨迹对照 |
| `tracking.png` / `fourier_analysis.png` | 误差、板角、力矩、频谱和速度 |
| `contour_points.npy` | 可直接复用的闭合轮廓点 |
| `metrics.json` | 全部指标 |

常用参数：

```bash
# 复杂背景：显式提供 mask（白色=前景）
python auto_draw.py photo.jpg --mask photo_mask.png

# 白底线稿/版画：手动阈值
python auto_draw.py engraving.jpg --mode threshold --threshold 220

# 只验证提取与控制，不渲染
python auto_draw.py image.png --no-video

# 调整傅里叶阶数/绘制尺寸/播放倍率
python auto_draw.py image.png --harmonics 60 --span 0.44 --speed 4
```

**边界说明**：当前管线画的是**单条闭合外轮廓**，因此不会自动画出物体内部的眼睛、文字或排线。复杂背景的分割也不是万能的；遇到分割失败时，用 `--mask` 是最稳的方式。如果需要内部细节，可以把外轮廓和多条内部轮廓串成一条带“提笔/落笔”标记的路径，这是下一步扩展方向。

### 怎么把一笔画变成参考轨迹

`drawing.py` 把闭合轮廓采样成复数序列 \(z_n=x_n+iy_n\)，做 FFT 得到傅里叶描述子：

\[
c_k=\frac{1}{N}\sum_{n=0}^{N-1} z_n e^{-i2\pi kn/N},
\qquad
z(u)=\sum_{k=-K}^{K} c_k e^{i2\pi ku}
\]

只保留 \(K=45\) 个正频率及其共轭项，轮廓就被低通成一个光滑、可微的单笔曲线。随后根据曲率 \(\kappa(s)\) 和期望的侧向加速度上限生成速度剖面：

\[
v_{\max}(s)=\min\left(v_{\text{cap}},\sqrt{\frac{a_{\text{lat}}}{\kappa(s)}}\right)
\]

再对切向加速度限幅、对速度剖面做滑动最小值和低通平滑，最后积分成 \(p_d(t)\)。尖角处球会自动减速，长直段则保持较快的线速度。

这段参考仍然送入完全相同的串级控制器：外环把 \(a_d\) 反解成板倾角，内环用关节 PD 输出力矩。球依然只是被动滚动，没有任何脚本化的位置。

要换成别的形状，只需提供任意闭合二维点列：

```python
from drawing import make_fourier_path, FourierDrawing

path = make_fourier_path(my_closed_points, harmonics=60, target_span=0.40)
reference = FourierDrawing(path)
```

## 快速开始

```bash
cd ball_on_plate_mujoco
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# 圆周跟踪：20 s 仿真，生成视频、GIF、曲线和 metrics
python simulate.py

# 任意形状 / Fourier 一笔画：34 s 绘制，输出 4× 双视图视频
python draw_fourier.py
python draw_fourier.py --speed 4 --width 900 --height 900

# 任意图片 → 自动轮廓 → MuJoCo 画线
python auto_draw.py assets/demo_star.png --name "Star" --harmonics 24

# 只跑物理和控制，不渲染
python simulate.py --no-video
python draw_fourier.py --no-video

# 自检
python -m unittest -v test_smoke.py test_fourier.py test_image_to_stroke.py
```

macOS 上程序会自动设置 `MUJOCO_GL=cgl`。Linux 无显示器时，渲染前可设置：

```bash
MUJOCO_GL=egl python simulate.py
```

如果机器没有可用的 OpenGL，使用 `--no-video`，物理仿真、指标和曲线仍会正常生成。

## 物理模型

模型文件：[`ball_on_plate.xml`](ball_on_plate.xml)

```
world
└── base / 立柱
    └── pitch_body        hinge: pitch，绕世界 y 轴
        └── plate         hinge: roll，绕俯仰后的局部 x 轴
            ├── 板面      0.60 × 0.60 × 0.016 m，质量 1.20 kg
            ├── 四周挡边
            └── 参考圆点（纯视觉，无碰撞）
ball                       free joint，半径 0.025 m，质量 0.045 kg
```

关键点：

- 板不是 mocap，也不是运动学 body；它由两个**力矩电机**驱动。
- 球是 `freejoint` 刚体，**没有任何执行器**；它只受到重力和板面接触力。
- 接触使用 `condim="6"`，同时包含滑动、扭转和滚动摩擦：
  - 滑动摩擦系数 1.1
  - 扭转摩擦 0.0005
  - 滚动摩擦 0.00005
- 物理步长 0.5 ms（2 kHz），控制器 1 kHz。MuJoCo 使用 `implicitfast` 积分器和椭圆摩擦锥。
- 球在仿真中的滑移速度约 0.8 mm/s，说明 no-slip 滚动约束基本成立。

这是一个“板动、球被动”的真实动力学系统：板倾斜产生的重力分量驱动球滚动，球的惯性反作用力和接触力又作用回板，由关节电机闭环维持姿态。

## 控制算法

控制代码：[`controller.py`](controller.py)。采用串级结构。

### 1. 外环：滚动球动力学的反馈线性化

对均质实心球 \(I=\frac25 mr^2\)，无滑动滚动时：

\[
m a = m g_t - f,\qquad I\alpha = rf,\qquad a=r\alpha
\]

消去摩擦力得到：

\[
a=\frac57 g_t
\]

板姿态取 \(R=R_y(\beta)R_x(\alpha)\)，重力在板面内的分量为：

\[
g_x=g\sin\beta,\qquad g_y=-g\sin\alpha\cos\beta
\]

因此球在板面坐标下的加速度为：

\[
\ddot p_x=\frac57 g\sin\beta,\qquad
\ddot p_y=-\frac57 g\sin\alpha\cos\beta
\]

外环先由参考轨迹和 PD 反馈得到期望加速度：

\[
a_{cmd}=a_d+K_p(p_d-p)+K_d(v_d-v)
\]

再解析反解出板角：

\[
\beta_d=\arcsin\!\left(\frac{7a_{cmd,x}}{5g}\right),\qquad
\alpha_d=-\arcsin\!\left(\frac{7a_{cmd,y}}{5g\cos\beta_d}\right)
\]

这就是对滚动约束的反馈线性化，而不是直接给球写位置。控制量还做了倾角/加速度限幅，避免初始阶段或扰动时给出不可实现的姿态。

### 2. 内环：板姿态的关节 PD

\[
\tau = K_{p,j}(\theta_d-\theta)-K_{d,j}\dot\theta
\]

两个力矩再经过 ±6 N·m 饱和后送入 MuJoCo 的 `motor` 执行器。外环带宽远低于内环，因此可以认为板快速跟随期望倾角。

### 3. 参考轨迹

半径用五次 smootherstep 从 0 平滑增加到 0.16 m，持续 4 s；其一阶、二阶导数在起点和终点都为零，因此 \(t=0\) 时球从板中心静止出发，参考速度和加速度均为零：

\[
r(t)=R_0\,s(t/T_r),\qquad s(u)=6u^5-15u^4+10u^3
\]

\[
p_d=r\begin{bmatrix}\cos\omega t\\ \sin\omega t\end{bmatrix},\qquad
\omega=\frac{2\pi}{T}
\]

之后保持半径 0.16 m、周期 4 s 的逆时针圆周运动。

### 控制参数

| 参数 | 值 |
|---|---:|
| 外环 \(K_p\) | 60 s⁻² |
| 外环 \(K_d\) | 18 s⁻¹ |
| 内环 \(K_{p,j}\) | 140 N·m/rad |
| 内环 \(K_{d,j}\) | 18 N·m·s/rad |
| 最大倾角 | 0.30 rad |
| 最大指令加速度 | 2.0 m/s² |
| 力矩限幅 | ±6 N·m |

外环没有使用积分项。圆周跟踪中的主要稳态误差来自滚动摩擦造成的微小相位滞后（0.32°），通过提高内环刚度/带宽控制在毫米级。

## 输出文件

运行 `python simulate.py` 后：

| 文件 | 内容 |
|---|---|
| [`outputs/ball_on_plate_circle.mp4`](outputs/ball_on_plate_circle.mp4) | 20 s 三维仿真视频，含实时误差 HUD 和球轨迹拖尾 |
| [`outputs/preview.gif`](outputs/preview.gif) | 8 s 循环预览 |
| [`outputs/preview.png`](outputs/preview.png) | 单帧预览 |
| [`outputs/tracking.png`](outputs/tracking.png) | 平面轨迹、跟踪误差、板角、电机力矩 |
| [`outputs/trajectory.npz`](outputs/trajectory.npz) | 200 Hz 完整日志（位置、速度、角度、力矩、滑移等） |
| [`outputs/metrics.json`](outputs/metrics.json) | 全部指标 |

运行 `python draw_fourier.py` 后：

| 文件 | 内容 |
|---|---|
| [`outputs/fourier/ball_on_plate_fourier.mp4`](outputs/fourier/ball_on_plate_fourier.mp4) | 4× 双视图视频（左俯视 / 右 3D） |
| [`outputs/fourier/preview.gif`](outputs/fourier/preview.gif) | 4× 双视图循环预览 |
| [`outputs/fourier/preview.png`](outputs/fourier/preview.png) | 完成后的双视图单帧 |
| [`outputs/fourier/tracking.png`](outputs/fourier/tracking.png) | 绘制轨迹、误差、板角、力矩 |
| [`outputs/fourier/fourier_analysis.png`](outputs/fourier/fourier_analysis.png) | 傅里叶重建、频谱、收敛、速度剖面 |
| [`outputs/fourier/portrait_reference.png`](outputs/fourier/portrait_reference.png) | 原版画、提取轮廓与球轨迹对照 |
| [`outputs/fourier/source_contour.png`](outputs/fourier/source_contour.png) | 原版画上的红色提取轮廓 |
| [`outputs/fourier/trajectory.npz`](outputs/fourier/trajectory.npz) | 绘制过程完整日志 |
| [`outputs/fourier/metrics.json`](outputs/fourier/metrics.json) | 全部指标 |

`auto_draw.py` 的每个输入图会在 `outputs/auto/<name>/` 下生成自己的 `stroke.mp4`、`comparison.png`、`mask.png` 和指标。

仿真日志中的位置/速度都在**板坐标系**下，和控制器使用的坐标一致。

## 代码结构

```
ball_on_plate_mujoco/
├── ball_on_plate.xml     # MuJoCo 物理模型（铰链、板、自由球、接触、电机）
├── controller.py         # 状态读取、串级控制器
├── simulate.py           # 圆周跟踪主循环、日志、视频/GIF
├── plotting.py           # 圆周结果曲线
├── drawing.py            # Fourier 描述子、任意一笔画、曲率限速
├── extract_portrait.py   # Fourier 版画的专用提取脚本
├── image_to_stroke.py    # 任意图片：前景分割 + 轮廓提取
├── auto_draw.py          # 任意图片 → MuJoCo 画线的一键管线
├── draw_fourier.py       # Fourier 绘制主循环、俯视 + 3D 双视图
├── drawing_plots.py      # 绘制结果与傅里叶分析图
├── assets/               # 公有领域原图与提取的轮廓点
├── test_smoke.py         # 圆周端到端自检
├── test_fourier.py       # Fourier 参考快速自检
├── test_image_to_stroke.py # 图片轮廓提取自检
├── requirements.txt
├── outputs/              # 圆周结果
└── outputs/fourier/      # Fourier 一笔画结果
```

## 实现说明与可扩展方向

- 外环使用的是准静态滚动模型，忽略了板角速度和角加速度带来的 Coriolis/惯性耦合项。本轨迹下板角速度峰值约 0.12 rad/s、倾角约 3.6°，这些项较小；1 mm 的跟踪误差也验证了该近似在这个工作点足够。
- 如果需要更激进的轨迹（更高频率、更大半径、快速换向），可以把外环替换为基于完整刚体动力学的 MPC / 计算力矩控制，或把板的角速度项加入前馈。
- 迁移到真机时，需要辨识滚动阻力、接触摩擦、电机带宽和传感器延迟；可以在此基础上加入积分项、扰动观测器或自适应摩擦补偿。
- 当前球和板都是刚体，接触是点接触。若研究软球/软板，可替换为 MuJoCo 的 flex/soft 模型或外部 FEM/MPM 耦合。
- 一笔画的目标是**手工设计的 Fourier 侧面轮廓**，不是从照片自动描边；如果要从图片生成，可先用边缘/轮廓提取得到闭合点列，再传给 `make_fourier_path`。
