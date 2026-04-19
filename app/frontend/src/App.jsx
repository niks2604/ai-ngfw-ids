import { Route, Routes } from 'react-router-dom'
import Sidebar from './components/Sidebar.jsx'
import Topbar from './components/Topbar.jsx'
import Overview from './pages/Overview.jsx'
import LiveFlows from './pages/LiveFlows.jsx'
import Explainability from './pages/Explainability.jsx'
import ThreatAnalytics from './pages/ThreatAnalytics.jsx'

export default function App() {
  return (
    <div className="flex min-h-screen">
      <Sidebar />
      <main className="flex-1">
        <Topbar />
        <div className="px-6 pb-10">
          <Routes>
            <Route path="/" element={<Overview />} />
            <Route path="/flows" element={<LiveFlows />} />
            <Route path="/explain" element={<Explainability />} />
            <Route path="/threats" element={<ThreatAnalytics />} />
          </Routes>
        </div>
      </main>
    </div>
  )
}
