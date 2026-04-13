import { motion } from 'framer-motion';
import { useQuery } from '@tanstack/react-query';
import { fetchHealth, fetchDocuments } from '../lib/api';
import SearchInput from '../components/search/SearchInput';
import SuggestionChips from '../components/search/SuggestionChips';

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.3 } },
};

export default function Home() {
  const { data: health } = useQuery({
    queryKey: ['health'],
    queryFn: fetchHealth,
    retry: false,
  });
  const { data: docs } = useQuery({
    queryKey: ['documents'],
    queryFn: fetchDocuments,
    retry: false,
  });

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="flex-1 flex flex-col relative overflow-hidden"
    >
      {/* Background Decorative Blobs */}
      <div className="absolute inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -right-[10%] w-[60%] h-[60%] bg-indigo-900/10 rounded-full blur-[120px]" />
        <div className="absolute -bottom-[10%] -left-[5%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[100px]" />
      </div>

      {/* Hero Search Section */}
      <section className="flex-1 flex flex-col items-center justify-center px-6 py-12 z-10">
        <div className="w-full max-w-3xl space-y-12 text-center">
          <div className="space-y-4">
            <h2 className="text-4xl md:text-6xl font-extrabold text-white font-headline tracking-tighter leading-tight">
              What shall we{' '}
              <span className="text-indigo-400">uncover</span> today?
            </h2>
            <p className="text-lg text-slate-400 font-body max-w-lg mx-auto">
              Synthesize your documentation with editorial precision through
              our semantic search engine.
            </p>
          </div>

          <SearchInput />
          <SuggestionChips />
        </div>
      </section>

      {/* System Health Footer */}
      <footer className="p-8 z-10 flex flex-col items-center gap-6">
        <div className="flex items-center gap-4 text-slate-500 font-label text-xs uppercase tracking-[0.2em] font-semibold">
          <span>Documents Indexed: {docs?.total?.toLocaleString() ?? '—'}</span>
          <span className="w-1 h-1 bg-slate-700 rounded-full" />
          <span>
            Database:{' '}
            {health?.database === 'connected' ? 'Connected' : '—'}
          </span>
        </div>
        <div className="inline-flex items-center gap-2 px-4 py-2 bg-emerald-950/30 text-emerald-400 rounded-full text-xs font-bold border border-emerald-900/50 backdrop-blur-sm">
          <span className="flex h-2 w-2 relative">
            <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75" />
            <span className="relative inline-flex rounded-full h-2 w-2 bg-emerald-500" />
          </span>
          {health?.status === 'ok' ? 'System Healthy' : 'Connecting...'}
        </div>
      </footer>

      {/* Texture Overlay */}
      <div className="fixed inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-[0.05] pointer-events-none z-0" />
    </motion.div>
  );
}
