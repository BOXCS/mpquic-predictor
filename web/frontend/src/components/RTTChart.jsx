/**
 * RTTChart.jsx — RTT Line Chart (Recharts)
 *
 * Renders a responsive LineChart showing RTT (ms) over time for both paths
 * simultaneously. All colours come from CSS custom-property tokens — no raw
 * hex values inside this file.
 *
 * Props:
 *   metrics          {object}  { 1: MetricPoint[], 2: MetricPoint[] }
 *                              Rolling window from useMetrics().
 *   prediction       {object|null}  { degradation_detected, confidence }
 *                              Used to shade the chart background when
 *                              degradation is detected.
 *   degradationThreshold  {number}  RTT ms above which the dashed reference
 *                              line is drawn (default: 150).
 *   height           {number}  Chart height in px (default: 260).
 *
 * Invariants:
 *   - No internal data fetching — receives all data as props.
 *   - Reads CSS var colours at render time via getComputedStyle on <html>.
 *   - Gracefully renders an empty state when both path arrays are empty.
 */

import { useMemo } from 'react'
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ReferenceLine,
} from 'recharts'
import { Activity } from 'lucide-react'

// ── CSS-var reader ──────────────────────────────────────────────────────────
// Called at render time so the correct light/dark value is used.
function cssVar(name) {
  return getComputedStyle(document.documentElement)
    .getPropertyValue(name)
    .trim()
}

// ── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Merge two per-path arrays into a single array keyed by a shared x-index.
 * Recharts needs one data array where each entry may have values for p1 and/or p2.
 */
function mergePathData(metrics) {
  const p1 = Array.isArray(metrics?.[1]) ? metrics[1] : []
  const p2 = Array.isArray(metrics?.[2]) ? metrics[2] : []
  const len = Math.max(p1.length, p2.length)
  if (len === 0) return []

  return Array.from({ length: len }, (_, i) => {
    const a = p1[p1.length - len + i]
    const b = p2[p2.length - len + i]
    // Use whichever timestamp is available; prefer path-1 for alignment.
    const ts = a?.timestamp ?? b?.timestamp
    const label = ts
      ? new Date(ts).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit', second: '2-digit' })
      : String(i)
    return {
      name: label,
      rtt_path1: a?.rtt_ms ?? null,
      rtt_path2: b?.rtt_ms ?? null,
    }
  })
}

// ── Custom tooltip ───────────────────────────────────────────────────────────
function CustomTooltip({ active, payload, label }) {
  if (!active || !payload?.length) return null
  return (
    <div
      style={{
        background: 'var(--bg-surface)',
        border: '1px solid var(--border-default)',
        borderRadius: '0.375rem',
        padding: '0.5rem 0.75rem',
        fontSize: '0.75rem',
        fontFamily: 'var(--font-mono)',
        color: 'var(--text-primary)',
        boxShadow: '0 2px 8px rgba(0,0,0,0.12)',
      }}
    >
      <p style={{ marginBottom: '0.25rem', color: 'var(--text-muted)' }}>{label}</p>
      {payload.map((entry) => (
        <p key={entry.dataKey} style={{ color: entry.stroke, margin: 0 }}>
          {entry.name}: {entry.value != null ? `${entry.value.toFixed(1)} ms` : '—'}
        </p>
      ))}
    </div>
  )
}

// ── Component ────────────────────────────────────────────────────────────────

export default function RTTChart({
  metrics = {},
  prediction = null,
  degradationThreshold = 150,
  height = 260,
}) {
  const chartData = useMemo(() => mergePathData(metrics), [metrics])

  const isDegraded = prediction?.degradation_detected === true
  const colorPath0 = cssVar('--chart-path-0')
  const colorPath1 = cssVar('--chart-path-1')
  const colorDegraded = cssVar('--chart-degraded')
  const colorMuted = cssVar('--text-muted')
  const colorBorder = cssVar('--border-default')

  // ── Empty state ────────────────────────────────────────────────────────────
  if (chartData.length === 0) {
    return (
      <div
        id="rtt-chart-empty"
        className="flex flex-col items-center justify-center gap-2 rounded-lg border"
        style={{
          height,
          borderColor: 'var(--border-default)',
          color: 'var(--text-muted)',
          background: 'var(--bg-surface)',
        }}
      >
        <Activity className="h-8 w-8" style={{ color: 'var(--text-muted)' }} />
        <span className="text-sm">Waiting for metric data…</span>
      </div>
    )
  }

  return (
    <div
      id="rtt-chart"
      className="rounded-lg border"
      style={{
        background: isDegraded
          ? `color-mix(in srgb, var(--bg-surface) 94%, var(--chart-degraded))`
          : 'var(--bg-surface)',
        borderColor: isDegraded ? 'var(--chart-degraded)' : 'var(--border-default)',
        transition: 'background 0.4s, border-color 0.4s',
      }}
    >
      <ResponsiveContainer width="100%" height={height}>
        <LineChart
          data={chartData}
          margin={{ top: 16, right: 20, left: 0, bottom: 4 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke={colorBorder}
            opacity={0.6}
          />

          <XAxis
            dataKey="name"
            tick={{ fontSize: 10, fill: colorMuted, fontFamily: 'var(--font-mono)' }}
            tickLine={false}
            axisLine={{ stroke: colorBorder }}
            interval="preserveStartEnd"
          />

          <YAxis
            tick={{ fontSize: 10, fill: colorMuted, fontFamily: 'var(--font-mono)' }}
            tickLine={false}
            axisLine={false}
            tickFormatter={(v) => `${v}`}
            unit=" ms"
            width={52}
          />

          <Tooltip content={<CustomTooltip />} />

          <Legend
            wrapperStyle={{
              fontSize: '0.72rem',
              fontFamily: 'var(--font-sans)',
              paddingTop: '4px',
            }}
          />

          {/* Degradation threshold reference line */}
          <ReferenceLine
            y={degradationThreshold}
            stroke={colorDegraded}
            strokeDasharray="5 3"
            strokeWidth={1.5}
            label={{
              value: `threshold ${degradationThreshold} ms`,
              position: 'insideTopRight',
              fontSize: 10,
              fill: colorDegraded,
              fontFamily: 'var(--font-mono)',
            }}
          />

          <Line
            type="monotone"
            dataKey="rtt_path1"
            name="wlan0 (path 1)"
            stroke={colorPath0}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
            connectNulls
            isAnimationActive={false}
          />

          <Line
            type="monotone"
            dataKey="rtt_path2"
            name="eth0 (path 2)"
            stroke={colorPath1}
            strokeWidth={2}
            dot={false}
            activeDot={{ r: 4, strokeWidth: 0 }}
            connectNulls
            isAnimationActive={false}
          />
        </LineChart>
      </ResponsiveContainer>
    </div>
  )
}
