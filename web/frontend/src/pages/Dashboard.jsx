/**
 * Dashboard.jsx — Real-Time Monitoring Page
 *
 * Full-viewport layout with:
 *   - Fixed top navbar
 *   - Metric summary stat cards
 *   - Two-column grid (RTTChart left, PathStatus + AlertBanner right)
 */

import { useMetrics } from '../hooks/useMetrics'
import RTTChart from '../components/RTTChart'
import PathStatus from '../components/PathStatus'
import AlertBanner from '../components/AlertBanner'
import Navbar from '../components/Navbar'

function StatCard({ title, value, unit, highlightWarning = false }) {
  return (
    <div
      className="flex flex-col rounded-lg border p-4"
      style={{
        backgroundColor: highlightWarning ? 'color-mix(in srgb, var(--bg-surface) 94%, var(--state-warning))' : 'var(--bg-surface)',
        borderColor: highlightWarning ? 'var(--state-warning)' : 'var(--border-default)'
      }}
    >
      <span
        style={{
          fontSize: '0.72rem',
          color: highlightWarning ? 'var(--state-warning)' : 'var(--text-muted)',
          fontWeight: 600,
          textTransform: 'uppercase',
          letterSpacing: '0.05em',
          marginBottom: '0.5rem',
        }}
      >
        {title}
      </span>
      <div className="flex items-baseline gap-1">
        <span
          className="font-mono text-2xl font-bold metric-value"
          style={{ color: highlightWarning ? 'var(--state-warning)' : 'var(--text-primary)' }}
        >
          {value ?? '—'}
        </span>
        {unit && value != null && (
          <span style={{ fontSize: '0.85rem', color: highlightWarning ? 'var(--state-warning)' : 'var(--text-muted)' }}>
            {unit}
          </span>
        )}
      </div>
    </div>
  )
}

export default function Dashboard() {
  const {
    metrics,
    latestPerPath,
    prediction,
    activePath,
    switchingEvents,
    isConnected,
  } = useMetrics()

  // ── Derived Stats ──────────────────────────────────────────────────────────
  const rtt1 = latestPerPath?.[1]?.rtt_ms
  const rtt2 = latestPerPath?.[2]?.rtt_ms
  
  const gp1 = latestPerPath?.[1]?.goodput_bps ?? 0
  const gp2 = latestPerPath?.[2]?.goodput_bps ?? 0
  const totalGoodputKbps = (gp1 + gp2) / 1000

  const conf = prediction?.confidence
  const confPct = conf != null ? (conf * 100).toFixed(0) : null
  const isDegraded = prediction?.degradation_detected === true

  return (
    <div
      id="dashboard-page"
      className="min-h-screen pt-14"
      style={{ backgroundColor: 'var(--bg-base)' }}
    >
      <Navbar isConnected={isConnected} />

      <main className="mx-auto max-w-7xl p-6 flex flex-col gap-6">
        
        {/* Stat Cards Row */}
        <div className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <StatCard
            title="RTT (wlan0)"
            value={rtt1 != null ? rtt1.toFixed(1) : null}
            unit="ms"
          />
          <StatCard
            title="RTT (eth0)"
            value={rtt2 != null ? rtt2.toFixed(1) : null}
            unit="ms"
          />
          <StatCard
            title="Total Goodput"
            value={totalGoodputKbps > 0 ? totalGoodputKbps.toFixed(1) : null}
            unit="kbps"
          />
          <StatCard
            title="Prediction Confidence"
            value={confPct}
            unit="%"
            highlightWarning={isDegraded}
          />
        </div>

        {/* Two-Column Grid */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          
          {/* Main Chart Area (2/3 width) */}
          <div className="lg:col-span-2 flex flex-col gap-4">
            <h2
              style={{
                fontSize: '1rem',
                fontWeight: 600,
                color: 'var(--text-primary)',
                margin: 0
              }}
            >
              Real-Time RTT
            </h2>
            <RTTChart
              metrics={metrics}
              prediction={prediction}
              degradationThreshold={150}
            />
          </div>

          {/* Side Panel (1/3 width) */}
          <div className="flex flex-col gap-4">
            <h2
              style={{
                fontSize: '1rem',
                fontWeight: 600,
                color: 'var(--text-primary)',
                margin: 0
              }}
            >
              Path Management
            </h2>
            
            {/* Banner renders at top of side panel if visible */}
            <AlertBanner
              switchingEvents={switchingEvents}
              prediction={prediction}
              autoDismissMs={8000}
            />
            
            <PathStatus
              activePath={activePath}
              latestPerPath={latestPerPath}
            />
          </div>
          
        </div>
      </main>
    </div>
  )
}
