import { useLocation, Link } from 'react-router-dom'

export default function Navbar({ isConnected }) {
  const { pathname } = useLocation()

  const navItems = [
    { name: 'Dashboard', path: '/' },
    { name: 'History', path: '/history' },
    { name: 'Scenarios', path: '/scenarios' },
  ]

  return (
    <nav
      className="fixed top-0 left-0 right-0 h-14 border-b flex items-center justify-between px-6 z-50"
      style={{ backgroundColor: 'var(--bg-surface)', borderColor: 'var(--border-default)' }}
    >
      <div className="flex items-center gap-8">
        <span className="font-bold" style={{ color: 'var(--text-primary)' }}>
          MP-QUIC Monitor
        </span>
        <div className="flex items-center gap-1">
          {navItems.map((item) => {
            const active = pathname === item.path
            return (
              <Link
                key={item.path}
                to={item.path}
                className="px-3 py-1.5 rounded-md text-sm font-medium transition-colors"
                style={{
                  color: active ? 'var(--accent-primary)' : 'var(--text-muted)',
                  backgroundColor: active ? 'var(--accent-subtle)' : 'transparent',
                }}
              >
                {item.name}
              </Link>
            )
          })}
        </div>
      </div>

      <div className="flex items-center gap-2">
        <span className="relative flex h-3 w-3">
          {isConnected && (
            <span
              className="animate-ping absolute inline-flex h-full w-full rounded-full opacity-75"
              style={{ backgroundColor: 'var(--state-success)' }}
            ></span>
          )}
          <span
            className="relative inline-flex rounded-full h-3 w-3"
            style={{ backgroundColor: isConnected ? 'var(--state-success)' : 'var(--state-error)' }}
          ></span>
        </span>
        <span
          className="text-sm font-medium"
          style={{ color: isConnected ? 'var(--state-success)' : 'var(--state-error)' }}
        >
          {isConnected ? 'Connected' : 'Disconnected'}
        </span>
      </div>
    </nav>
  )
}
