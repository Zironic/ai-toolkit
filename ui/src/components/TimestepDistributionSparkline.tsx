'use client';

import { useEffect, useMemo, useRef } from 'react';

// ── sampling math (mirrors custom_flowmatch_sampler.py) ───────────────────

function randn(): number {
  const u1 = Math.max(1e-10, Math.random());
  return Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * Math.random());
}

function sigmoid(x: number) {
  return 1 / (1 + Math.exp(-x));
}

function timeShift(mu: number, t: number): number {
  const tc = Math.max(1e-10, Math.min(1 - 1e-10, t));
  return Math.exp(mu) / (Math.exp(mu) + (1 / tc - 1));
}

function calcShiftMu(
  seqLen: number,
  baseSeqLen = 256,
  maxSeqLen = 1024,
  baseShift = 0.5,
  maxShift = 0.85,
  minShift = 0.33,
): number {
  const m = (maxShift - baseShift) / (maxSeqLen - baseSeqLen);
  const b = baseShift - m * baseSeqLen;
  return Math.max(minShift, seqLen * m + b);
}

// Returns N noise fractions in [0,1] sorted descending (index 0 = most noisy).
// Mirrors set_train_timesteps() in the custom scheduler.
function buildSchedule(type: string, n: number, seqLen: number): number[] {
  if (type === 'sigmoid') {
    // t = sigmoid(randn()), timestep = (1-t)*1000 → noise_fraction = 1-t
    const arr = Array.from({ length: n }, () => 1 - sigmoid(randn()));
    return arr.sort((a, b) => b - a);
  }

  if (type === 'lognorm_blend') {
    // 75% LogNormal(0, 0.333) reversed+scaled, 25% uniform
    const nLog = Math.floor(n * 0.75);
    const logSamples = Array.from({ length: nLog }, () => Math.exp(0.333 * randn()));
    const logMax = Math.max(...logSamples);
    const logPart = logSamples.map(t => 1 - t / logMax);
    const linPart = Array.from({ length: n - nLog }, (_, i) => 1 - i / Math.max(1, n - nLog - 1));
    return [...logPart, ...linPart].sort((a, b) => b - a);
  }

  if (type === 'shift' || type === 'flux_shift' || type === 'lumina2_shift') {
    const mu = calcShiftMu(seqLen);
    // linspace from ~1 to ~0, then time_shift applied
    return Array.from({ length: n }, (_, i) => {
      const sigma = 1 - i / Math.max(1, n - 1);
      return timeShift(mu, Math.max(1e-10, sigma));
    });
  }

  // 'linear', 'weighted', 'weighted_low' — uniform linspace
  return Array.from({ length: n }, (_, i) => 1 - i / Math.max(1, n - 1));
}

// Samples M noise fractions by picking biased indices into the schedule.
// Mirrors the content_or_style branching in BaseSDTrainProcess.py.
function sampleDistribution(
  type: string,
  bias: string,
  seqLen: number,
  M = 2500,
): Float32Array {
  const schedule = buildSchedule(type, 1000, seqLen);
  const N = schedule.length;
  const out = new Float32Array(M);
  for (let i = 0; i < M; i++) {
    const u = Math.random();
    let idx: number;
    if (bias === 'content') {
      // High Noise: cubic bias toward index 0 (highest noise)
      idx = Math.floor(u ** 3 * N);
    } else if (bias === 'style') {
      // Low Noise: cubic bias toward last index (lowest noise)
      idx = Math.floor((1 - u ** 3) * N);
    } else {
      // balanced: uniform index
      idx = Math.floor(u * N);
    }
    out[i] = schedule[Math.min(N - 1, Math.max(0, idx))];
  }
  return out;
}

// ── component ─────────────────────────────────────────────────────────────

interface Props {
  timestepType: string;
  contentOrStyle: string;
  /** Pixel resolution of training images — only affects the 'shift' schedule. */
  resolution?: number;
  className?: string;
}

const BINS = 44;
const SAMPLES = 2500;

export default function TimestepDistributionSparkline({
  timestepType,
  contentOrStyle,
  resolution = 1024,
  className,
}: Props) {
  const canvasRef = useRef<HTMLCanvasElement>(null);

  // Token count for the shift formula: stride = VAE_spatial(8) × patch(2) = 16px
  const seqLen = useMemo(
    () => Math.round((resolution / 16) ** 2),
    [resolution],
  );

  const hist = useMemo(() => {
    const samples = sampleDistribution(timestepType, contentOrStyle, seqLen, SAMPLES);
    const bins = new Float32Array(BINS);
    for (let i = 0; i < samples.length; i++) {
      // noise fraction: 0 = clean (left), 1 = noisy (right)
      const b = Math.min(BINS - 1, Math.floor(samples[i] * BINS));
      bins[b]++;
    }
    return bins;
  }, [timestepType, contentOrStyle, seqLen]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const dpr = window.devicePixelRatio ?? 1;
    const w = canvas.offsetWidth;
    const h = canvas.offsetHeight;
    if (w === 0 || h === 0) return;
    canvas.width = w * dpr;
    canvas.height = h * dpr;
    const ctx = canvas.getContext('2d')!;
    ctx.scale(dpr, dpr);
    ctx.clearRect(0, 0, w, h);

    const maxVal = Math.max(...hist);
    if (maxVal === 0) return;

    const padL = 2, padR = 2, padT = 3, padB = 16;
    const gW = w - padL - padR;
    const gH = h - padT - padB;
    const barW = gW / BINS;

    // Filled area
    const grad = ctx.createLinearGradient(padL, 0, padL + gW, 0);
    grad.addColorStop(0, 'rgba(59, 130, 246, 0.25)');
    grad.addColorStop(1, 'rgba(168, 85, 247, 0.45)');

    ctx.beginPath();
    ctx.moveTo(padL, padT + gH);
    for (let i = 0; i < BINS; i++) {
      const x = padL + i * barW;
      const barH = (hist[i] / maxVal) * gH;
      ctx.lineTo(x, padT + gH - barH);
      ctx.lineTo(x + barW, padT + gH - barH);
    }
    ctx.lineTo(padL + gW, padT + gH);
    ctx.closePath();
    ctx.fillStyle = grad;
    ctx.fill();

    // Top outline
    const lineGrad = ctx.createLinearGradient(padL, 0, padL + gW, 0);
    lineGrad.addColorStop(0, 'rgba(96, 165, 250, 0.85)');
    lineGrad.addColorStop(1, 'rgba(192, 132, 252, 0.85)');
    ctx.beginPath();
    for (let i = 0; i < BINS; i++) {
      const x = padL + i * barW;
      const barH = (hist[i] / maxVal) * gH;
      if (i === 0) ctx.moveTo(x, padT + gH - barH);
      else ctx.lineTo(x, padT + gH - barH);
      ctx.lineTo(x + barW, padT + gH - barH);
    }
    ctx.strokeStyle = lineGrad;
    ctx.lineWidth = 1.5;
    ctx.stroke();

    // Baseline
    ctx.beginPath();
    ctx.moveTo(padL, padT + gH + 0.5);
    ctx.lineTo(padL + gW, padT + gH + 0.5);
    ctx.strokeStyle = 'rgba(255,255,255,0.12)';
    ctx.lineWidth = 1;
    ctx.stroke();

    // Labels
    ctx.font = `${9 * Math.min(1, w / 160)}px system-ui, sans-serif`;
    ctx.fillStyle = 'rgba(255,255,255,0.4)';
    ctx.textAlign = 'left';
    ctx.fillText('Clean', padL, h - 3);
    ctx.textAlign = 'right';
    ctx.fillText('Noisy', w - padR, h - 3);
    if (timestepType === 'shift' || timestepType === 'flux_shift' || timestepType === 'lumina2_shift') {
      ctx.textAlign = 'center';
      ctx.fillText(`${resolution}px`, w / 2, h - 3);
    }
  }, [hist, timestepType, resolution]);

  return (
    <canvas
      ref={canvasRef}
      className={className}
      style={{ width: '100%', height: '64px', display: 'block' }}
    />
  );
}
