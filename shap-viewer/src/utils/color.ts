import { interpolateRdBu } from "d3-scale-chromatic";

export type ShapValueMode = "discrete" | "continuous";

export const SHAP_DISCRETE_LABELS = [
  "Very Negative",
  "Negative",
  "Neutral",
  "Positive",
  "Very Positive",
] as const;

type DiscreteBucket = {
  index: number;
  label: (typeof SHAP_DISCRETE_LABELS)[number];
  centerRatio: number;
};

function clamp(value: number, min: number, max: number): number {
  return Math.min(max, Math.max(min, value));
}

function getDiscreteBucket(value: number, maxAbs: number): DiscreteBucket {
  const safeMax = maxAbs === 0 ? 1 : maxAbs;
  const ratio = clamp(value / safeMax, -1, 1);

  // 5 equal-width bins across [-1, 1]
  // [-1, -0.6), [-0.6, -0.2), [-0.2, 0.2), [0.2, 0.6), [0.6, 1]
  let index = 0;
  if (ratio < -0.6) index = 0;
  else if (ratio < -0.2) index = 1;
  else if (ratio < 0.2) index = 2;
  else if (ratio < 0.6) index = 3;
  else index = 4;

  const centerRatios = [-0.8, -0.4, 0, 0.4, 0.8] as const;
  return {
    index,
    label: SHAP_DISCRETE_LABELS[index],
    centerRatio: centerRatios[index],
  };
}

export function getGlobalMaxAbs(values: number[]): number {
  const max = Math.max(...values.map((v) => Math.abs(v)), 0);
  return max === 0 ? 1 : max;
}

export function shapToColor(
  value: number,
  maxAbs: number,
  mode: ShapValueMode = "continuous",
): string {
  const safeMax = maxAbs === 0 ? 1 : maxAbs;

  if (mode === "discrete") {
    const bucket = getDiscreteBucket(value, safeMax);
    const t = (bucket.centerRatio + 1) / 2;
    return interpolateRdBu(t);
  }

  const normalized = (value + safeMax) / (2 * safeMax);
  return interpolateRdBu(normalized);
}

export function getTextColorForBackground(
  value: number,
  maxAbs: number,
  mode: ShapValueMode = "continuous",
): string {
  const safeMax = maxAbs === 0 ? 1 : maxAbs;
  const intensity =
    mode === "discrete"
      ? Math.abs(getDiscreteBucket(value, safeMax).centerRatio)
      : Math.abs(value) / safeMax;
  return intensity > 0.55 ? "#ffffff" : "#0f172a";
}

export function formatShap(value: number): string {
  return value.toFixed(4);
}

export function formatShapDiscrete(value: number, maxAbs: number): string {
  return getDiscreteBucket(value, maxAbs).label;
}

export function getShapDiscreteColors(): string[] {
  // Colors correspond to the bucket centers in getDiscreteBucket.
  return [-0.8, -0.4, 0, 0.4, 0.8].map((ratio) => interpolateRdBu((ratio + 1) / 2));
}
