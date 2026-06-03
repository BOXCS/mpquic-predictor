/**
 * Scenario.jsx — Degradation Scenario List Page
 *
 * Displays all 12 simulation scenarios and their status.
 * Highlights the live scenario inferred from current metrics.
 * Auto-refreshes every 10 seconds via REST.
 */

import { useState, useEffect } from 'react'
import Navbar from '../components/Navbar'
import { BarChart2, CheckCircle, Circle } from 'lucide-react'
import { useWebSocket } from '../hooks/useWebSocket'
// import { useWS } from '../context/WebSocketContext'

const envApiUrl = import.meta.env.VITE_API_URL;
const API_URL = (envApiUrl && envApiUrl.trim() !== '') ? envApiUrl : '';

function ScenarioCard({ scenario, isLive }) {
  const { name, simulated, description, path1, path2 } = scenario

  return (
    <div
      className="flex flex-col p-4 rounded-lg border transition-colors"
      style={{
        backgroundColor: isLive ? 'var(--accent-subtle)' : 'var(--bg-surface)',
        borderColor: isLive ? 'var(--accent-primary)' : 'var(--border-default)',
        boxShadow: isLive ? '0 0 0 1px var(--accent-primary)' : 'none',
      }}
    >
      <div className="flex items-start justify-between mb-2">
        <div className="flex flex-col">
          <span className="font-bold text-base mb-1" style={{ color: isLive ? 'var(--accent-primary)' : 'var(--text-primary)' }}>
            {name}
            {isLive && (
              <span className="ml-2 text-xs uppercase tracking-wider font-bold px-1.5 py-0.5 rounded-sm" style={{ backgroundColor: 'var(--accent-primary)', color: '#fff' }}>
                LIVE
              </span>
            )}
          </span>
          <span className="text-xs" style={{ color: 'var(--text-muted)' }}>{description}</span>
        </div>
        <div className="flex items-center gap-1 shrink-0 ml-4">
          {simulated ? (
            <CheckCircle className="h-4 w-4" style={{ color: 'var(--state-success)' }} />
          ) : (
            <Circle className="h-4 w-4" style={{ color: 'var(--text-muted)' }} />
          )}
          <span className="text-xs font-semibold" style={{ color: simulated ? 'var(--state-success)' : 'var(--text-muted)' }}>
            {simulated ? 'Simulated' : 'Pending'}
          </span>
        </div>
      </div>

      <div className="mt-auto pt-3 border-t grid grid-cols-2 gap-4" style={{ borderColor: 'var(--border-default)' }}>
        <div className="flex flex-col gap-1">
          <span className="text-[0.65rem] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>wlan0</span>
          <span className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>
            {path1.base_delay_ms}ms ±{path1.jitter_ms}ms | {path1.loss_pct}% loss
          </span>
        </div>
        <div className="flex flex-col gap-1">
          <span className="text-[0.65rem] font-bold uppercase tracking-wider" style={{ color: 'var(--text-muted)' }}>eth0</span>
          <span className="text-xs font-mono" style={{ color: 'var(--text-primary)' }}>
            {path2.base_delay_ms}ms ±{path2.jitter_ms}ms | {path2.loss_pct}% loss
          </span>
        </div>
      </div>
    </div>
  )
}

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 border rounded-lg mt-8" style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
      <BarChart2 className="h-8 w-8 mb-4" style={{ color: 'var(--text-muted)' }} />
      <span style={{ color: 'var(--text-muted)' }}>{message}</span>
    </div>
  )
}

export default function Scenario() {
  const { isConnected } = useWebSocket // Just for Navbar dot

  const [scenarios, setScenarios] = useState([])
  const [liveScenario, setLiveScenario] = useState(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    let mounted = true

    const fetchData = async () => {
      try {
        const [scenRes, liveRes] = await Promise.all([
          fetch(`${API_URL}/scenarios/`).then(r => r.json()),
          fetch(`${API_URL}/scenarios/live`).then(r => r.json())
        ])
        if (mounted) {
          setScenarios(scenRes.data || [])
          setLiveScenario(liveRes.data?.estimated_scenario || null)
          setLoading(false)
        }
      } catch (err) {
        console.error(err)
        if (mounted) setLoading(false)
      }
    }

    fetchData()
    const interval = setInterval(fetchData, 10000)

    return () => {
      mounted = false
      clearInterval(interval)
    }
  }, [])

  return (
    <div className="min-h-screen pt-14" style={{ backgroundColor: 'var(--bg-base)' }}>
      <Navbar isConnected={isConnected} />

      <main className="mx-auto max-w-7xl p-6 flex flex-col gap-6">
        <div className="flex items-center justify-between border-b pb-4" style={{ borderColor: 'var(--border-default)' }}>
          <h1 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>Degradation Scenarios</h1>
        </div>

        {loading && scenarios.length === 0 ? (
          <div className="p-8 text-center" style={{ color: 'var(--text-muted)' }}>Loading scenarios...</div>
        ) : scenarios.length === 0 ? (
          <EmptyState message="Failed to load scenarios or no scenarios found." />
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
            {scenarios.map((sc) => (
              <ScenarioCard
                key={sc.name}
                scenario={sc}
                isLive={sc.name === liveScenario}
              />
            ))}
          </div>
        )}
      </main>
    </div>
  )
}
