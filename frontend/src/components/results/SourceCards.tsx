import type { Source } from '../../lib/types';

interface Props {
  sources: Source[];
}

function getSourceType(path: string): string {
  if (path.endsWith('.pdf')) return 'Paper';
  if (path.endsWith('.docx')) return 'Report';
  if (path.endsWith('.md')) return 'Document';
  if (path.endsWith('.csv')) return 'Data';
  if (path.endsWith('.html')) return 'Web';
  return 'Source';
}

export default function SourceCards({ sources }: Props) {
  return (
    <section className="mb-12">
      <div className="flex items-center gap-2 mb-4 opacity-60">
        <span className="material-symbols-outlined text-slate-400 text-xs">
          database
        </span>
        <h3 className="text-[10px] font-bold font-headline tracking-widest uppercase text-slate-400">
          Sources
        </h3>
      </div>

      <div className="grid grid-cols-1 sm:grid-cols-2 md:grid-cols-4 gap-3">
        {sources.map((source, i) => (
          <div
            key={i}
            className="group bg-slate-900/60 border border-white/5 px-4 py-3 rounded-xl hover:bg-slate-800/80 hover:border-indigo-500/30 transition-all flex items-center justify-between cursor-pointer"
          >
            <div className="flex items-center gap-3 overflow-hidden">
              <span className="text-[8px] font-bold text-indigo-400 uppercase tracking-tighter shrink-0 bg-indigo-500/10 px-1.5 py-0.5 rounded">
                {getSourceType(source.source_path)}
              </span>
              <h4 className="text-xs text-slate-300 font-medium truncate group-hover:text-white transition-colors">
                {source.document_title || source.source_path.split('/').pop()}
              </h4>
            </div>
            <span className="text-[10px] text-slate-500 font-label shrink-0 ml-1">
              {(source.similarity_score * 100).toFixed(0)}%
            </span>
          </div>
        ))}
      </div>
    </section>
  );
}
