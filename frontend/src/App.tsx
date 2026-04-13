import { Routes, Route, useLocation } from 'react-router-dom';
import { AnimatePresence } from 'framer-motion';
import AppShell from './components/layout/AppShell';
import Home from './pages/Home';
import Results from './pages/Results';
import Library from './pages/Library';

export default function App() {
  const location = useLocation();
  return (
    <AppShell>
      <AnimatePresence mode="wait">
        <Routes location={location} key={location.pathname}>
          <Route path="/" element={<Home />} />
          <Route path="/results" element={<Results />} />
          <Route path="/library" element={<Library />} />
        </Routes>
      </AnimatePresence>
    </AppShell>
  );
}
