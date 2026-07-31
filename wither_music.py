# -*- coding: utf-8 -*-
"""
凋零BOSS风格绝望纯音乐合成器
- 5分钟（300秒）无缝循环
- 黑暗 ambient / drone / 不和谐钟声 / 心跳鼓
- D 减七和弦（含三全音魔鬼音程）
输出: wither_despair.wav  (后续可转 mp3)
"""
import numpy as np
import wave

SR = 44100                # 采样率
DURATION = 300            # 最终时长（秒）= 5 分钟
OVERLAP = 2               # 首尾交叉淡化长度（秒）
TOTAL = DURATION + OVERLAP
N = int(TOTAL * SR)       # 总采样点数
t = np.arange(N) / SR     # 时间轴

# ---------- 通用工具 ----------
def midi_to_freq(m):
    return 440.0 * 2.0 ** ((m - 69) / 12.0)

def adsr(n, a, d, s, r, sus_level=1.0):
    """生成 ADSR 包络。a/d/r 为秒。对任意总长 n 健壮（不足时按比例缩短）。"""
    total = n
    a_s = int(a * SR); d_s = int(d * SR); r_s = int(r * SR)
    need = a_s + d_s + r_s
    if need > total and need > 0:
        scale = total / need
        a_s = int(a_s * scale)
        d_s = int(d_s * scale)
        r_s = max(0, total - a_s - d_s)
    sus_s = total - a_s - d_s - r_s
    if sus_s < 0:
        sus_s = 0
        r_s = max(0, total - a_s - d_s)
    env = np.zeros(total)
    i = 0
    if a_s > 0:
        env[i:i+a_s] = np.linspace(0, 1, a_s, endpoint=False); i += a_s
    if d_s > 0:
        env[i:i+d_s] = np.linspace(1, sus_level, d_s, endpoint=False); i += d_s
    if sus_s > 0:
        env[i:i+sus_s] = sus_level; i += sus_s
    if r_s > 0:
        env[i:i+r_s] = np.linspace(sus_level, 0, r_s, endpoint=False)
    return env

def saw(f, n, phase=0.0):
    """带限锯齿波（叠加前若干正弦谐波近似）。"""
    x = np.arange(n) / SR
    sig = np.zeros(n)
    harmonics = min(40, int(SR/2/f) - 1)
    for k in range(1, harmonics+1):
        sig += np.sin(2*np.pi*f*k*x + phase) / k
    return sig * 2 / np.pi

def triangle(f, n, phase=0.0):
    x = np.arange(n) / SR
    sig = np.zeros(n)
    harmonics = min(40, int(SR/2/f) - 1)
    for k in range(1, harmonics+1, 2):
        sig += ((-1)**((k-1)//2)) * np.sin(2*np.pi*f*k*x + phase) / (k*k)
    return sig * 8 / (np.pi*np.pi)

def sine(f, n, phase=0.0):
    x = np.arange(n) / SR
    return np.sin(2*np.pi*f*x + phase)

# 简单一阶低通（用于噪声/弦乐柔化）
def lowpass(sig, alpha=0.1):
    out = np.empty_like(sig)
    out[0] = sig[0]
    for i in range(1, len(sig)):
        out[i] = out[i-1] + alpha * (sig[i] - out[i-1])
    return out

# 用 cumsum 做移动平均低通（O(n)，比 np.convolve 快很多），边缘 reflect 补齐
def lowpass_fast(sig, window=256):
    if window <= 1:
        return sig
    w = int(window)
    pad = w // 2
    padded = np.pad(sig, pad, mode='reflect')
    csum = np.cumsum(padded, dtype=np.float64)
    L = len(sig)
    # window_sum[i] = csum[i+w] - csum[i]
    window_sum = csum[w:w+L] - csum[0:L]
    out = (window_sum / w)
    return out.astype(np.float32)

# ---------- 主输出缓冲（立体声） ----------
out_L = np.zeros(N, dtype=np.float32)
out_R = np.zeros(N, dtype=np.float32)

def add_layer(sig, gain=1.0, pan=0.0, dest_L=None, dest_R=None):
    """把单声道信号叠加到输出。pan: -1(左) ~ +1(右)。"""
    L = dest_L if dest_L is not None else out_L
    R = dest_R if dest_R is not None else out_R
    gL = gain * (1.0 - max(0.0, pan))   # pan=+1 -> 右声道
    gR = gain * (1.0 - max(0.0, -pan))  # pan=-1 -> 左声道
    # 等价于: gL=gain*(0.5-pan/2)*2, 简化为:
    gL = gain * (1.0 - pan) * 0.5 + gain * 0.5 * 0  # 下面用更直观方式
    # 直观公式
    gL = gain * np.clip(1.0 - pan, 0.0, 1.0)
    gR = gain * np.clip(1.0 + pan, 0.0, 1.0)
    L[:len(sig)] += (sig * gL).astype(np.float32)
    R[:len(sig)] += (sig * gR).astype(np.float32)

# =====================================================================
# 层 1: 低频持续 drone（贯穿全曲，缓慢起伏）
# 音: D1(36.71) + D2(73.42) + 微失谐，营造深渊般的低频压顶感
# =====================================================================
print("[1/6] 合成低频 drone...")
drone = np.zeros(N, dtype=np.float32)
f_d1 = midi_to_freq(26)   # D1 = 36.71
f_d2 = midi_to_freq(38)   # D2 = 73.42
# 用正弦+轻微失谐叠加，模拟厚重的持续低音
drone += 0.6 * sine(f_d1, N)
drone += 0.3 * sine(f_d1 * 1.003, N)   # 失谐，产生缓慢拍音
drone += 0.4 * sine(f_d2, N)
drone += 0.2 * sine(f_d2 * 1.005, N)
# 加入一点低八度的三角波增加"嗡"感
drone += 0.15 * triangle(f_d1, N)
# 缓慢音量起伏（LFO ~ 0.05Hz，约20秒一个周期），保证开头结尾都接近中等响度
lfo = 0.5 + 0.5 * np.sin(2*np.pi*0.05*t + np.pi*0.3)
# 整体淡入（开头4秒）和淡出（结尾保持，由交叉淡化处理）
fade_in = np.minimum(1.0, t / 4.0)
drone_env = fade_in * (0.55 + 0.45 * lfo)
drone = drone * drone_env
# drone 低通一下，去掉可能的数字味
drone = lowpass_fast(drone, window=128)
add_layer(drone, gain=0.55, pan=0.0)

# =====================================================================
# 层 2: 减和弦弦乐 pad（D-F-Ab-B 减七和弦）
# 多层失谐锯齿 + 低通 + 缓慢呼吸式包络
# =====================================================================
print("[2/6] 合成减和弦 pad...")
# 和弦音 (D 减七): D3 F3 Ab3 B3
chord_notes = [50, 53, 56, 59]
# 不同的和弦变化段落（缓慢下行/变换，营造下沉的绝望感）
# 每段 60 秒，共 5 段 + overlap
sections = [
    # (起始秒, 持续秒, midi列表, gain)
    (0,   60, [50, 53, 56, 59], 0.18),   # D dim7
    (60,  60, [49, 52, 55, 58], 0.18),   # C# dim7 (半音下行)
    (120, 60, [50, 53, 56, 59], 0.20),   # 回到 D dim7
    (180, 60, [48, 51, 54, 57], 0.18),   # C dim7 (继续下行)
    (240, 62, [50, 53, 56, 59], 0.16),   # 回归 D dim7（含 overlap）
]
for start, dur, notes, g in sections:
    s0 = int(start * SR)
    s1 = min(N, int((start + dur) * SR))
    seg_n = s1 - s0
    if seg_n <= 0:
        continue
    # pad 段：长淡入淡出（呼吸感）
    a = 12.0; r = 12.0
    env = adsr(seg_n, a=a, d=2.0, s=0.0, r=r, sus_level=0.85)
    seg = np.zeros(seg_n, dtype=np.float32)
    for m in notes:
        f = midi_to_freq(m)
        # 三个失谐锯齿叠加，做厚 pad
        seg += 0.33 * saw(f, seg_n, phase=0.0)
        seg += 0.33 * saw(f*1.004, seg_n, phase=1.3)
        seg += 0.33 * saw(f*0.996, seg_n, phase=2.7)
    seg = seg * env
    # 低通柔化
    seg = lowpass_fast(seg, window=512)
    # 加入一点三角波增加圆润
    tri_layer = np.zeros(seg_n, dtype=np.float32)
    for m in notes:
        tri_layer += 0.25 * triangle(midi_to_freq(m), seg_n)
    tri_layer = lowpass_fast(tri_layer, window=512) * env
    seg = seg * 0.7 + tri_layer * 0.3
    # 立体声轻微扩散：左右轻微失谐
    out_L[s0:s1] += (seg * g * 1.0).astype(np.float32)
    out_R[s0:s1] += (seg * g * 0.92).astype(np.float32)  # 右声道稍弱，制造宽立体感

# =====================================================================
# 层 3: 下行低音线条（每 8 秒一个音，缓慢下行制造"坠落"感）
# 音色: 低频锯齿+正弦，带包络
# =====================================================================
print("[3/6] 合成下行低音线条...")
bass_pattern = [38, 37, 36, 35, 34, 33, 32, 31, 30, 38]  # D2 下行到 D1 再回
bass_note_dur = 8.0
idx = 0
pos = 0.0
# 开头留 4 秒给 drone 单独出现；只生成完整长度的音，避免末尾截断影响循环
pos = 4.0
while pos + bass_note_dur <= TOTAL:
    m = bass_pattern[idx % len(bass_pattern)]
    f = midi_to_freq(m)
    n_seg = int(bass_note_dur * SR)
    s0 = int(pos * SR)
    s1 = s0 + n_seg
    seg_n = s1 - s0
    # 包络：快速淡入，缓慢淡出，连绵
    env = adsr(seg_n, a=0.8, d=1.5, s=0.0, r=3.0, sus_level=0.7)
    seg = (0.6 * saw(f, seg_n) + 0.5 * sine(f, seg_n) + 0.3 * triangle(f, seg_n))
    seg = seg * env
    seg = lowpass_fast(seg, window=256)
    out_L[s0:s1] += (seg * 0.16).astype(np.float32)
    out_R[s0:s1] += (seg * 0.16).astype(np.float32)
    pos += bass_note_dur
    idx += 1

# =====================================================================
# 层 4: 不和谐金属钟声 / 铜锣（不规则间隔，制造警示与绝望）
# 用非谐波泛音叠加 + FM，长衰减
# =====================================================================
print("[4/6] 合成钟声/铜锣...")
def bell_strike(freq, n, gain=1.0):
    """金属钟声：基频 + 非谐波泛音，指数衰减。"""
    x = np.arange(n) / SR
    sig = np.zeros(n)
    # 非谐波泛音比例（类似真实钟/锣）
    partials = [(1.0, 1.0, 6.0),    # (频率倍数, 振幅, 衰减速率)
                (2.0, 0.5, 4.0),
                (2.4, 0.35, 3.5),
                (3.1, 0.25, 2.8),
                (4.2, 0.18, 2.2),
                (5.4, 0.12, 1.8),
                (6.7, 0.08, 1.4)]
    for mult, amp, decay in partials:
        f = freq * mult
        if f > SR/2:
            continue
        decay_env = np.exp(-x * decay)
        sig += amp * np.sin(2*np.pi*f*x) * decay_env
    # 初始的金属"铛"声（高频噪声短脉冲）
    noise_len = int(0.05 * SR)
    noise = (np.random.rand(noise_len) * 2 - 1) * np.exp(-np.arange(noise_len)/SR * 40)
    sig[:noise_len] += noise * 0.3
    return sig * gain

# 钟声事件：(时间秒, midi, 持续秒, gain, pan)
bell_events = [
    (8.0,   74, 9.0, 0.35, -0.3),   # D5 远处钟
    (24.0,  69, 8.0, 0.30,  0.3),   # A4
    (45.0,  77, 10.0, 0.38, -0.2),  # F5 (减和弦音)
    (70.0,  80, 9.0, 0.32, 0.25),   # Ab5 (三全音强调，最不和谐)
    (95.0,  74, 8.0, 0.30, -0.3),
    (118.0, 71, 9.0, 0.34, 0.2),    # B4 (减七)
    (140.0, 77, 10.0, 0.40, -0.25),
    (162.0, 69, 8.0, 0.28, 0.3),
    (185.0, 80, 10.0, 0.42, -0.2),  # 高潮点1
    (210.0, 74, 8.0, 0.32, 0.3),
    (235.0, 77, 9.0, 0.36, -0.3),
    (258.0, 71, 9.0, 0.30, 0.25),
    (278.0, 74, 10.0, 0.34, -0.2),  # 结尾前，与开头8秒处的钟呼应（循环衔接）
]
for (ts, m, dur, g, pan) in bell_events:
    s0 = int(ts * SR)
    s1 = min(N, s0 + int(dur * SR))
    seg_n = s1 - s0
    if seg_n <= 0:
        continue
    seg = bell_strike(midi_to_freq(m), seg_n, gain=1.0)
    # 轻微低通过滤，模拟"远处"感
    seg = lowpass_fast(seg, window=64)
    out_L[s0:s1] += (seg * g * (1.0 - pan)).astype(np.float32)
    out_R[s0:s1] += (seg * g * (1.0 + pan)).astype(np.float32)

# =====================================================================
# 层 5: 慢速心跳鼓声（lub-dub 双击，约 30 BPM）
# 低频 boom + 少量噪声 click
# =====================================================================
print("[5/6] 合成心跳鼓...")
def heartbeat(n, freq=55.0):
    """单次低频心跳。"""
    x = np.arange(n) / SR
    # 低频正弦快速下滑（boom）
    f_env = freq * (1.0 + 0.8 * np.exp(-x * 30))  # 频率从高快速下滑
    phase = 2 * np.pi * np.cumsum(f_env) / SR
    boom = np.sin(phase) * np.exp(-x * 5.0)
    # 加一点噪声点击增加"皮"感
    click_len = int(0.02 * SR)
    click = (np.random.rand(click_len) * 2 - 1) * np.exp(-np.arange(click_len)/SR * 80) * 0.4
    boom[:click_len] += click
    return boom

beat_interval = 4.0   # 每 4 秒一组心跳
t_start = 12.0        # 12 秒后开始（让前面铺垫）
pos = t_start
# 只生成不跨越 DURATION 边界的心跳，保证结尾 overlap 区只剩 drone，循环无缝
while pos + 1.0 <= DURATION:
    # lub
    n1 = int(0.6 * SR)
    s0 = int(pos * SR)
    s1 = min(N, s0 + n1)
    seg = heartbeat(s1 - s0, freq=55.0)
    out_L[s0:s1] += (seg * 0.22).astype(np.float32)
    out_R[s0:s1] += (seg * 0.22).astype(np.float32)
    # dub (稍弱，0.18 秒后)
    s0b = int((pos + 0.18) * SR)
    s1b = min(N, s0b + n1)
    seg2 = heartbeat(s1b - s0b, freq=48.0)
    out_L[s0b:s1b] += (seg2 * 0.14).astype(np.float32)
    out_R[s0b:s1b] += (seg2 * 0.14).astype(np.float32)
    pos += beat_interval

# =====================================================================
# 层 6: 氛围风声 / 低通噪声（持续，缓慢起伏）
# =====================================================================
print("[6/6] 合成氛围风声...")
noise = np.random.rand(N).astype(np.float32) * 2 - 1
noise = lowpass_fast(noise, window=2048)  # 重低通 -> 风声
# 额外再低通一次更柔和
noise = lowpass_fast(noise, window=1024)
# 缓慢起伏
lfo2 = 0.5 + 0.5 * np.sin(2*np.pi*0.03*t + 1.2)
fade_in2 = np.minimum(1.0, t / 6.0)
noise_env = fade_in2 * (0.4 + 0.6 * lfo2)
noise = noise * noise_env
out_L += (noise * 0.06).astype(np.float32)
out_R += (noise * 0.055).astype(np.float32)

# =====================================================================
# 交叉淡化 -> 无缝 5 分钟循环
# 把原始 302 秒末尾的 OVERLAP 秒叠加到开头的 OVERLAP 秒
# =====================================================================
print("交叉淡化处理无缝循环...")
ov = int(OVERLAP * SR)
dur_n = int(DURATION * SR)
loop_L = np.zeros(dur_n, dtype=np.float32)
loop_R = np.zeros(dur_n, dtype=np.float32)
fade_out = np.linspace(1.0, 0.0, ov, dtype=np.float32)  # 给末尾
fade_in  = np.linspace(0.0, 1.0, ov, dtype=np.float32)  # 给开头
# 开头 overlap 段 = 原始开头 * fadeIn + 原始末尾 * fadeOut
loop_L[:ov] = out_L[:ov] * fade_in + out_L[dur_n:dur_n+ov] * fade_out
loop_R[:ov] = out_R[:ov] * fade_in + out_R[dur_n:dur_n+ov] * fade_out
# 中段不变
loop_L[ov:dur_n] = out_L[ov:dur_n]
loop_R[ov:dur_n] = out_R[ov:dur_n]

# =====================================================================
# 后期处理：立体声合并、归一化、软限幅
# =====================================================================
stereo = np.stack([loop_L, loop_R], axis=1)
# 归一化到 -3 dBFS peak
peak = np.max(np.abs(stereo))
if peak > 0:
    target = 0.707  # -3 dB
    stereo = stereo * (target / peak)
# 软限幅（tanh）防止削波
stereo = np.tanh(stereo * 1.1) * 0.9
# 最终 int16
audio_int = (stereo * 32767 * 0.95).astype(np.int16)

# =====================================================================
# 写入 WAV
# =====================================================================
out_path = "/workspace/wither_despair.wav"
with wave.open(out_path, 'w') as wf:
    wf.setnchannels(2)
    wf.setsampwidth(2)
    wf.setframerate(SR)
    wf.writeframes(audio_int.tobytes())

actual_dur = audio_int.shape[0] / SR
print(f"完成: {out_path}")
print(f"时长: {actual_dur:.2f} 秒 ({actual_dur/60:.2f} 分钟)")
print(f"采样率: {SR} Hz, 立体声, 16-bit")
