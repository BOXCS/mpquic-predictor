import { BrowserRouter, Routes, Route, Navigate } from 'react-router-dom'
import Dashboard from './pages/Dashboard'
import History from './pages/History'
import Scenario from './pages/Scenario'

/**
 * App — root component
 *
 * Sets up React Router with three pages:
 *   /            → Dashboard (real-time monitoring)
 *   /history     → History (past metrics & switching events)
 *   /scenarios   → Scenario (14 degradation scenario list)
 */
function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Dashboard />} />
        <Route path="/history" element={<History />} />
        <Route path="/scenarios" element={<Scenario />} />
        {/* Redirect unknown paths to dashboard */}
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  )
}

export default App
