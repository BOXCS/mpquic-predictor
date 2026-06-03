/**
 * History.jsx — Metric and Event History Page
 *
 * Displays paginated history of network metrics and switching events
 * fetched via REST APIs.
 */

import { useState, useEffect } from 'react'
import Navbar from '../components/Navbar'
import { Clock } from 'lucide-react'
import { useWebSocket } from '../hooks/useWebSocket'
// import { useWS } from '../context/WebSocketContext'

const envApiUrl = import.meta.env.VITE_API_URL;
const API_URL = (envApiUrl && envApiUrl.trim() !== '') ? envApiUrl : '';

function formatTime(isoStr) {
  if (!isoStr) return '—'
  try {
    return new Date(isoStr).toLocaleString()
  } catch {
    return isoStr
  }
}

function EmptyState({ message }) {
  return (
    <div className="flex flex-col items-center justify-center p-12 border rounded-lg" style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}>
      <Clock className="h-8 w-8 mb-4" style={{ color: 'var(--text-muted)' }} />
      <span style={{ color: 'var(--text-muted)' }}>{message}</span>
    </div>
  )
}

export default function History() {
  const { isConnected } = useWebSocket() // Just for Navbar dot
  const [activeTab, setActiveTab] = useState('events') // 'events' | 'metrics'

  const [events, setEvents] = useState([])
  const [metrics, setMetrics] = useState([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    setLoading(true)
    if (activeTab === 'events') {
      fetch(`${API_URL}/metrics/switching-events?limit=100`)
        .then(res => res.json())
        .then(res => setEvents(res.data || []))
        .catch(err => console.error(err))
        .finally(() => setLoading(false))
    } else {
      fetch(`${API_URL}/metrics/?limit=100`)
        .then(res => res.json())
        .then(res => setMetrics(res.data || []))
        .catch(err => console.error(err))
        .finally(() => setLoading(false))
    }
  }, [activeTab])

  return (
    <div className="min-h-screen pt-14" style={{ backgroundColor: 'var(--bg-base)' }}>
      <Navbar isConnected={isConnected} />

      <main className="mx-auto max-w-7xl p-6 flex flex-col gap-6">
        <div className="flex items-center justify-between border-b pb-4" style={{ borderColor: 'var(--border-default)' }}>
          <h1 className="text-xl font-bold" style={{ color: 'var(--text-primary)' }}>History</h1>

          <div className="flex gap-2 p-1 rounded-md" style={{ backgroundColor: 'var(--bg-surface-raised)' }}>
            <button
              onClick={() => setActiveTab('events')}
              className="px-4 py-1.5 rounded-sm text-sm font-medium transition-colors"
              style={{
                backgroundColor: activeTab === 'events' ? 'var(--bg-surface)' : 'transparent',
                color: activeTab === 'events' ? 'var(--text-primary)' : 'var(--text-muted)',
                boxShadow: activeTab === 'events' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
              }}
            >
              Switching Events
            </button>
            <button
              onClick={() => setActiveTab('metrics')}
              className="px-4 py-1.5 rounded-sm text-sm font-medium transition-colors"
              style={{
                backgroundColor: activeTab === 'metrics' ? 'var(--bg-surface)' : 'transparent',
                color: activeTab === 'metrics' ? 'var(--text-primary)' : 'var(--text-muted)',
                boxShadow: activeTab === 'metrics' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none'
              }}
            >
              Network Metrics
            </button>
          </div>
        </div>

        {loading ? (
          <div className="p-8 text-center" style={{ color: 'var(--text-muted)' }}>Loading...</div>
        ) : activeTab === 'events' ? (
          events.length === 0 ? (
            <EmptyState message="No switching events recorded." />
          ) : (
            <div className="overflow-x-auto rounded-lg border" style={{ borderColor: 'var(--border-default)', backgroundColor: 'var(--bg-surface)' }}>
              <table className="w-full text-left text-sm" style={{ color: 'var(--text-primary)' }}>
                <thead className="border-b bg-gray-50/5" style={{ borderColor: 'var(--border-default)', backgroundColor: 'var(--bg-surface-raised)' }}>
                  <tr>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>Timestamp</th>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>From Path</th>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>To Path</th>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>Reason</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: 'var(--border-default)' }}>
                  {events.map((ev) => (
                    <tr key={ev.id} className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                      <td className="px-4 py-3 font-mono text-xs">{formatTime(ev.timestamp)}</td>
                      <td className="px-4 py-3 font-mono text-xs">{ev.from_path === 1 ? 'wlan0' : 'eth0'}</td>
                      <td className="px-4 py-3 font-mono text-xs">{ev.to_path === 1 ? 'wlan0' : 'eth0'}</td>
                      <td className="px-4 py-3 text-xs">{ev.reason}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        ) : (
          metrics.length === 0 ? (
            <EmptyState message="No metrics recorded." />
          ) : (
            <div className="overflow-x-auto rounded-lg border" style={{ borderColor: 'var(--border-default)', backgroundColor: 'var(--bg-surface)' }}>
              <table className="w-full text-left text-sm" style={{ color: 'var(--text-primary)' }}>
                <thead className="border-b bg-gray-50/5" style={{ borderColor: 'var(--border-default)', backgroundColor: 'var(--bg-surface-raised)' }}>
                  <tr>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>Timestamp</th>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>Path</th>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>RTT (ms)</th>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>Goodput (kbps)</th>
                    <th className="px-4 py-3 font-semibold" style={{ color: 'var(--text-muted)' }}>Loss (%)</th>
                  </tr>
                </thead>
                <tbody className="divide-y" style={{ borderColor: 'var(--border-default)' }}>
                  {metrics.map((m) => (
                    <tr key={m.id} className="hover:bg-black/5 dark:hover:bg-white/5 transition-colors">
                      <td className="px-4 py-3 font-mono text-xs">{formatTime(m.timestamp)}</td>
                      <td className="px-4 py-3 font-mono text-xs">{m.path_id === 1 ? 'wlan0' : 'eth0'}</td>
                      <td className="px-4 py-3 font-mono text-xs">{m.rtt_ms.toFixed(1)}</td>
                      <td className="px-4 py-3 font-mono text-xs">{(m.goodput_bps / 1000).toFixed(1)}</td>
                      <td className="px-4 py-3 font-mono text-xs">{m.loss_pct.toFixed(1)}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )
        )}
      </main>
    </div>
  )
}
