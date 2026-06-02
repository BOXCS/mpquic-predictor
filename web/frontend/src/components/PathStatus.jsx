/**
 * PathStatus.jsx — Active Path Health Badge Panel
 *
 * Displays the currently active path (wlan0 / eth0) and per-path health state.
 *
 * Props:
 *   activePath    {object|null}   { active_path, path_label, health[] }
 *                                 where health[n] = { path_id, path_label,
 *                                   latest_rtt_ms, latest_goodput_bps,
 *                                   latest_loss_pct, degradation_detected,
 *                                   prediction_confidence }
 *   latestPerPath {object}        { 1: MetricPoint|null, 2: MetricPoint|null }
 *
 * Health states (per path, derived from degradation_detected + rtt):
 *   healthy  → --state-success  + CheckCircle
 *   degraded → --state-warning  + AlertTriangle
 *   critical → --state-error    + XCircle  (degradation_detected AND high rtt)
 *   unknown  → --text-muted     + Activity (no data yet)
 *
 * Invariants:
 *   - No internal data fetching — all data via props.
 *   - No hardcoded hex colours.
 */

import { Wifi, Network, CheckCircle, AlertTriangle, XCircle, Activity } from 'lucide-react'

// ── Constants ────────────────────────────────────────────────────────────────
const CRITICAL_RTT_MS = 150  // matches degradationThreshold in RTTChart

const PATH_CONFIG = {
  1: { label: 'wlan0', Icon: Wifi },
  2: { label: 'eth0',  Icon: Network },
}

// ── Health derivation ────────────────────────────────────────────────────────
function deriveHealth(healthEntry, latestPoint) {
  const rtt = healthEntry?.latest_rtt_ms ?? latestPoint?.rtt_ms
  const degraded = healthEntry?.degradation_detected

  if (rtt == null && degraded == null) return 'unknown'
  if (degraded && rtt != null && rtt >= CRITICAL_RTT_MS) return 'critical'
  if (degraded) return 'degraded'
  return 'healthy'
}

const HEALTH_CONFIG = {
  healthy:  { label: 'Healthy',  color: 'var(--state-success)', Icon: CheckCircle },
  degraded: { label: 'Degraded', color: 'var(--state-warning)', Icon: AlertTriangle },
  critical: { label: 'Critical', color: 'var(--state-error)',   Icon: XCircle },
  unknown:  { label: 'No data',  color: 'var(--text-muted)',    Icon: Activity },
}

// ── Sub-components ───────────────────────────────────────────────────────────

function MetricRow({ label, value, unit = '' }) {
  if (value == null) return null
  return (
    <div className="flex items-center justify-between gap-4">
      <span style={{ color: 'var(--text-muted)', fontSize: '0.72rem' }}>{label}</span>
      <span
        className="metric-value"
        style={{ color: 'var(--text-primary)', fontSize: '0.78rem' }}
      >
        {typeof value === 'number' ? value.toFixed(1) : value}
        {unit && <span style={{ color: 'var(--text-muted)' }}> {unit}</span>}
      </span>
    </div>
  )
}

function PathCard({ pathId, isActive, healthEntry, latestPoint }) {
  const { label, Icon } = PATH_CONFIG[pathId] ?? { label: `path${pathId}`, Icon: Activity }
  const health = deriveHealth(healthEntry, latestPoint)
  const { label: healthLabel, color: healthColor, Icon: HealthIcon } = HEALTH_CONFIG[health]

  return (
    <div
      className="flex flex-col gap-3 rounded-lg border p-4"
      style={{
        background: isActive ? 'var(--accent-subtle)' : 'var(--bg-surface-raised)',
        borderColor: isActive ? 'var(--accent-primary)' : 'var(--border-default)',
        transition: 'background 0.3s, border-color 0.3s',
      }}
    >
      {/* Header row */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Icon className="h-5 w-5" style={{ color: isActive ? 'var(--accent-primary)' : 'var(--text-muted)' }} />
          <span
            style={{
              fontWeight: 600,
              fontSize: '0.85rem',
              color: isActive ? 'var(--accent-primary)' : 'var(--text-primary)',
            }}
          >
            {label}
          </span>
          {isActive && (
            <span
              className="rounded-full px-2 py-0.5"
              style={{
                background: 'var(--accent-primary)',
                color: '#fff',
                fontSize: '0.65rem',
                fontWeight: 700,
                letterSpacing: '0.04em',
              }}
            >
              ACTIVE
            </span>
          )}
        </div>

        {/* Health badge */}
        <div className="flex items-center gap-1">
          <HealthIcon className="h-4 w-4" style={{ color: healthColor }} />
          <span style={{ color: healthColor, fontSize: '0.72rem', fontWeight: 600 }}>
            {healthLabel}
          </span>
        </div>
      </div>

      {/* Metrics */}
      <div className="flex flex-col gap-1">
        <MetricRow
          label="RTT"
          value={healthEntry?.latest_rtt_ms ?? latestPoint?.rtt_ms}
          unit="ms"
        />
        <MetricRow
          label="Goodput"
          value={
            (healthEntry?.latest_goodput_bps ?? latestPoint?.goodput_bps) != null
              ? ((healthEntry?.latest_goodput_bps ?? latestPoint?.goodput_bps) / 1000).toFixed(1)
              : null
          }
          unit="kbps"
        />
        <MetricRow
          label="Loss"
          value={healthEntry?.latest_loss_pct ?? latestPoint?.loss_pct}
          unit="%"
        />
        {healthEntry?.prediction_confidence != null && (
          <MetricRow
            label="Confidence"
            value={(healthEntry.prediction_confidence * 100).toFixed(0)}
            unit="%"
          />
        )}
      </div>
    </div>
  )
}

// ── Main component ───────────────────────────────────────────────────────────

export default function PathStatus({ activePath = null, latestPerPath = {} }) {
  const activeId = activePath?.active_path
  const healthArr = Array.isArray(activePath?.health) ? activePath.health : []
  const healthByPath = Object.fromEntries(healthArr.map((h) => [h.path_id, h]))

  return (
    <div id="path-status" className="flex flex-col gap-3">
      {/* Summary header */}
      <div className="flex items-center justify-between px-1">
        <span
          style={{
            fontSize: '0.75rem',
            fontWeight: 700,
            textTransform: 'uppercase',
            letterSpacing: '0.06em',
            color: 'var(--text-muted)',
          }}
        >
          Path Status
        </span>
        {activeId != null && (
          <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
            Active:{' '}
            <span style={{ color: 'var(--accent-primary)', fontWeight: 600 }}>
              {activePath?.path_label ?? `path ${activeId}`}
            </span>
          </span>
        )}
      </div>

      {/* One card per path */}
      {[1, 2].map((pid) => (
        <PathCard
          key={pid}
          pathId={pid}
          isActive={activeId === pid}
          healthEntry={healthByPath[pid] ?? null}
          latestPoint={latestPerPath[pid] ?? null}
        />
      ))}

      {/* No data state */}
      {activeId == null && (
        <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', textAlign: 'center', padding: '0.5rem 0' }}>
          Waiting for path data…
        </p>
      )}
    </div>
  )
}
