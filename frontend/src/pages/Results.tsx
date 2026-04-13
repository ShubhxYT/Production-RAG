import { useSearchParams, useNavigate } from 'react-router-dom';
import { useQuery, useMutation } from '@tanstack/react-query';
import { motion } from 'framer-motion';
import { submitQuery, submitFeedback } from '../lib/api';
import SynthesisPanel from '../components/results/SynthesisPanel';
import SourceCards from '../components/results/SourceCards';
import FeedbackBar from '../components/results/FeedbackBar';
import FollowUpInput from '../components/results/FollowUpInput';

const pageVariants = {
  initial: { opacity: 0, y: 20 },
  animate: { opacity: 1, y: 0, transition: { duration: 0.5, ease: 'easeOut' } },
  exit: { opacity: 0, y: -20, transition: { duration: 0.3 } },
};

function ShimmerSkeleton() {
  return (
    <div className="space-y-8 animate-pulse">
      <div className="flex items-center justify-between">
        <div className="h-8 bg-slate-800 rounded-lg w-1/3" />
        <div className="h-8 bg-emerald-950/30 rounded-full w-36" />
      </div>
      <div className="bg-slate-900/40 border border-white/5 p-8 rounded-2xl space-y-4">
        <div className="h-4 bg-slate-800 rounded w-full" />
        <div className="h-4 bg-slate-800 rounded w-5/6" />
        <div className="h-4 bg-slate-800 rounded w-4/6" />
        <div className="h-4 bg-slate-800 rounded w-full" />
        <div className="h-4 bg-slate-800 rounded w-3/4" />
      </div>
      <div className="grid grid-cols-4 gap-3">
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-16 bg-slate-900/60 border border-white/5 rounded-xl" />
        ))}
      </div>
    </div>
  );
}

export default function Results() {
  const [searchParams] = useSearchParams();
  const question = searchParams.get('q') || '';
  const navigate = useNavigate();

  const { data, isLoading, error } = useQuery({
    queryKey: ['query', question],
    queryFn: () => submitQuery({ question, top_k: 5 }),
    enabled: !!question,
  });

  const feedbackMutation = useMutation({
    mutationFn: submitFeedback,
  });

  const handleFollowUp = (q: string) => {
    navigate(`/results?q=${encodeURIComponent(q)}`);
  };

  const handleFeedback = (type: 'thumbs_up' | 'thumbs_down') => {
    feedbackMutation.mutate({
      query_log_id: data?.query_log_id ?? null,
      feedback_type: type,
      query_text: question,
    });
  };

  const groundedPercent = data?.sources?.length
    ? Math.round(
        (data.sources.reduce((sum, s) => sum + s.similarity_score, 0) /
          data.sources.length) *
          100
      )
    : 0;

  return (
    <motion.div
      variants={pageVariants}
      initial="initial"
      animate="animate"
      exit="exit"
      className="flex-1 flex flex-col relative z-10"
    >
      {/* Background */}
      <div className="fixed inset-0 z-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-[20%] -right-[10%] w-[60%] h-[60%] bg-indigo-900/10 rounded-full blur-[120px]" />
        <div className="absolute -bottom-[10%] -left-[5%] w-[40%] h-[40%] bg-blue-900/10 rounded-full blur-[100px]" />
        <div className="absolute inset-0 bg-[url('https://www.transparenttextures.com/patterns/cubes.png')] opacity-[0.05]" />
      </div>

      {/* Header */}
      <header className="flex justify-between items-center px-8 py-4 w-full sticky top-0 z-50 bg-slate-950/60 backdrop-blur-xl border-b border-white/5">
        <div className="flex items-center flex-1 max-w-2xl">
          <div className="relative w-full">
            <span className="material-symbols-outlined absolute left-4 top-1/2 -translate-y-1/2 text-slate-500">
              search
            </span>
            <input
              className="w-full bg-slate-900/50 border border-slate-800/50 rounded-xl py-2 pl-12 pr-4 text-sm text-on-surface focus:ring-1 focus:ring-indigo-500 focus:bg-slate-900 transition-all placeholder-slate-500"
              placeholder={question}
              type="text"
              defaultValue={question}
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  const value = (e.target as HTMLInputElement).value.trim();
                  if (value) handleFollowUp(value);
                }
              }}
            />
          </div>
        </div>
        <div className="flex items-center gap-6 ml-6">
          <div className="text-xl font-bold bg-gradient-to-br from-indigo-400 to-indigo-600 bg-clip-text text-transparent font-headline tracking-tight">
            Cerebro Flux
          </div>
          <div className="flex items-center gap-4">
            <button className="material-symbols-outlined text-slate-500 hover:text-white transition-colors p-2 rounded-full hover:bg-slate-800/50">
              settings
            </button>
            <button className="material-symbols-outlined text-slate-500 hover:text-white transition-colors p-2 rounded-full hover:bg-slate-800/50">
              help
            </button>
            <div className="h-9 w-9 rounded-full bg-slate-800 overflow-hidden ring-2 ring-indigo-500/20 shadow-lg" />
          </div>
        </div>
      </header>

      {/* Main Content */}
      <main className="max-w-4xl mx-auto w-full px-8 py-12 flex-grow relative z-10">
        {isLoading ? (
          <ShimmerSkeleton />
        ) : error ? (
          <div className="bg-red-950/30 border border-red-500/20 rounded-2xl p-8 text-red-300">
            <p className="font-headline font-bold mb-2">Query Failed</p>
            <p className="text-sm text-red-400">
              Unable to reach the backend. Make sure the FullRag server is
              running on port 8001.
            </p>
          </div>
        ) : data ? (
          <>
            <SynthesisPanel
              answer={data.answer}
              groundedPercent={groundedPercent}
              latency={data.latency}
            />
            <SourceCards sources={data.sources} />
            <FeedbackBar
              onFeedback={handleFeedback}
              submitted={feedbackMutation.isSuccess}
            />
          </>
        ) : null}
      </main>

      {/* Follow-up Footer */}
      <FollowUpInput onSubmit={handleFollowUp} />
    </motion.div>
  );
}
