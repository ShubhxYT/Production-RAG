import type { LatencyBreakdown } from '../../lib/types';

interface Props {
  answer: string;
  groundedPercent: number;
  latency: LatencyBreakdown;
}

function renderAnswerWithCitations(text: string) {
  const parts = text.split(/(\[\d+\])/g);
  return parts.map((part, i) => {
    if (/^\[\d+\]$/.test(part)) {
      return (
        <span
          key={i}
          className="inline-flex items-center justify-center bg-indigo-500/20 text-indigo-300 rounded-lg px-2 py-0.5 text-[10px] font-bold mx-1 cursor-pointer hover:bg-indigo-500/40 transition-colors border border-indigo-500/20"
        >
          {part}
        </span>
      );
    }
    return <span key={i}>{part}</span>;
  });
}

export default function SynthesisPanel({ answer, groundedPercent, latency }: Props) {
  return (
    <section className="mb-10">
      <div className="flex items-center justify-between mb-8">
        <h2 className="text-3xl font-extrabold font-headline tracking-tight text-white">
          Curated Synthesis
        </h2>
        <div className="flex items-center px-4 py-2 bg-emerald-950/30 border border-emerald-500/20 rounded-full backdrop-blur-md">
          <span
            className="material-symbols-outlined text-emerald-400 text-sm mr-2"
            style={{ fontVariationSettings: "'FILL' 1" }}
          >
            verified
          </span>
          <span className="text-[11px] font-bold text-emerald-400 tracking-widest uppercase">
            {groundedPercent}% Grounded
          </span>
        </div>
      </div>

      <div className="bg-slate-900/40 border border-white/5 backdrop-blur-sm p-8 rounded-2xl shadow-2xl leading-relaxed text-slate-200/90 space-y-6">
        {answer.split('\n\n').map((paragraph, i) => (
          <p key={i} className="text-lg font-body font-light">
            {renderAnswerWithCitations(paragraph)}
          </p>
        ))}
      </div>

      <div className="mt-4 flex gap-4 text-[10px] text-slate-500 uppercase tracking-widest font-label">
        <span>Retrieval: {latency.retrieval_ms.toFixed(0)}ms</span>
        <span>Generation: {latency.generation_ms.toFixed(0)}ms</span>
        <span>Total: {latency.total_ms.toFixed(0)}ms</span>
      </div>
    </section>
  );
}
