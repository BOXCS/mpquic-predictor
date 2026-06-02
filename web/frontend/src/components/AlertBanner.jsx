/**
 * AlertBanner.jsx — Dismissible Switching-Event Alert Banner
 *
 * Shown when a new path switching event arrives. Displays the most recent
 * event: from_path → to_path, timestamp, and prediction confidence.
 * Auto-dismisses after autoDismissMs (default 8 s). Can be dismissed manually.
 *
 * Props:
 *   switchingEvents   {Array}   Newest-first list from useMetrics().
 *                               Each entry: { id, timestamp, from_path,
 *                               to_path, reason }
 *   prediction        {object|null}  { confidence } — shown alongside event.
 *   autoDismissMs     {number}  ms before auto-dismiss (default: 8000).
 *
 * Behaviour:
 *   - A new event causes the banner to appear and starts an auto-dismiss timer.
 *   - The dismiss button or timeout hides it.
 *   - Renders nothing when there are no events or after dismiss.
 *
 * Invariants:
 *   - No internal data fetching — all data via props.
 *   - No hardcoded hex colours.
 */

import { useState, useEffect, useRef, useCallback } from 'react'
import { ArrowLeftRight, X } from 'lucide-react'

const PATH_LABELS = { 1: 'wlan0', 2: 'eth0' }

function formatTime(isoStr) {
  if (!isoStr) return '—'
  try {
    return new Date(isoStr).toLocaleTimeString([], {
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    })
  } catch {
    return isoStr
  }
}

export default function AlertBanner({
  switchingEvents = [],
  prediction = null,
  autoDismissMs = 8000,
}) {
  const [visible, setVisible] = useState(false)
  const shownIdRef = useRef(null)
  const timerRef   = useRef(null)

  const latestEvent = switchingEvents[0] ?? null
  const latestId    = latestEvent?.id ?? null

  const dismiss = useCallback(() => {
    clearTimeout(timerRef.current)
    setVisible(false)
  }, [])

  // Show banner when a genuinely new event id arrives.
  useEffect(() => {
    if (latestId == null || latestId === shownIdRef.current) return

    shownIdRef.current = latestId
    setVisible(true)
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(dismiss, autoDismissMs)
  }, [latestId, autoDismissMs, dismiss])

  // Cleanup timer on unmount.
  useEffect(() => () => clearTimeout(timerRef.current), [])

  if (!visible || !latestEvent) return null

  const fromLabel = PATH_LABELS[latestEvent.from_path] ?? `path ${latestEvent.from_path}`
  const toLabel   = PATH_LABELS[latestEvent.to_path]   ?? `path ${latestEvent.to_path}`
  const conf      = prediction?.confidence != null
    ? `${(prediction.confidence * 100).toFixed(0)}% confidence`
    : null

  return (
    <div
      id="alert-banner"
      role="alert"
      aria-live="polite"
      className="flex items-start gap-3 rounded-lg border p-4"
      style={{
        background: 'color-mix(in srgb, var(--bg-surface) 85%, var(--state-warning))',
        borderColor: 'var(--state-warning)',
        color: 'var(--text-primary)',
        animation: 'slideIn 0.25s ease-out',
      }}
    >
      {/* Icon */}
      <ArrowLeftRight
        className="h-5 w-5 mt-0.5 shrink-0"
        style={{ color: 'var(--state-warning)' }}
      />

      {/* Content */}
      <div className="flex flex-col gap-0.5 flex-1 min-w-0">
        <p style={{ fontWeight: 600, fontSize: '0.85rem', margin: 0 }}>
          Path switched:{' '}
          <span style={{ color: 'var(--state-warning)' }}>
            {fromLabel} → {toLabel}
          </span>
        </p>

        {latestEvent.reason && (
          <p style={{ fontSize: '0.75rem', color: 'var(--text-muted)', margin: 0 }}>
            {latestEvent.reason}
            {conf ? ` · ${conf}` : ''}
          </p>
        )}

        <p
          className="metric-value"
          style={{ fontSize: '0.7rem', color: 'var(--text-muted)', margin: 0 }}
        >
          {formatTime(latestEvent.timestamp)}
        </p>
      </div>

      {/* Dismiss button */}
      <button
        id="alert-banner-dismiss"
        onClick={dismiss}
        aria-label="Dismiss alert"
        className="shrink-0 rounded-md p-0.5 transition-opacity hover:opacity-70"
        style={{
          color: 'var(--text-muted)',
          background: 'transparent',
          border: 'none',
          cursor: 'pointer',
        }}
      >
        <X className="h-4 w-4" />
      </button>

      {/* Slide-in keyframe */}
      <style>{`
        @keyframes slideIn {
          from { opacity: 0; transform: translateY(-6px); }
          to   { opacity: 1; transform: translateY(0); }
        }
      `}</style>
    </div>
  )
}
