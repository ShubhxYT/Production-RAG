import type { DocumentItem } from '../../lib/types';

interface Props {
  document: DocumentItem;
}

function getDocIcon(path: string): string {
  if (path.endsWith('.pdf')) return 'description';
  if (path.endsWith('.docx')) return 'article';
  if (path.endsWith('.csv')) return 'data_table';
  if (path.endsWith('.md')) return 'text_snippet';
  if (path.endsWith('.html')) return 'web';
  return 'draft';
}

export default function DocumentCard({ document: doc }: Props) {
  const isProcessing = doc.chunk_count === 0;

  return (
    <div className="bg-slate-800/50 p-6 rounded-xl border border-transparent hover:shadow-2xl hover:shadow-indigo-500/5 transition-all group">
      <div className="flex justify-between items-start mb-4">
        <div className="p-3 bg-indigo-900/40 rounded-lg text-primary">
          <span className="material-symbols-outlined">
            {getDocIcon(doc.source_path)}
          </span>
        </div>
        {isProcessing ? (
          <div className="flex items-center gap-2">
            <span className="animate-pulse w-2 h-2 rounded-full bg-primary" />
            <span className="text-[10px] font-bold tracking-widest uppercase text-primary">
              Processing
            </span>
          </div>
        ) : (
          <span className="bg-tertiary-fixed-dim text-on-tertiary-fixed px-3 py-1 rounded-full text-[10px] font-bold tracking-widest uppercase">
            Indexed
          </span>
        )}
      </div>

      <h3 className="font-headline font-bold text-lg mb-1 group-hover:text-primary transition-colors">
        {doc.title || doc.source_path.split('/').pop()}
      </h3>
      <p className="text-xs font-medium text-outline mb-6 flex items-center gap-1">
        <span className="material-symbols-outlined text-[14px]">
          folder_open
        </span>
        {doc.source_path}
      </p>

      <div className="flex items-center justify-between pt-4 border-t border-slate-700/50">
        <span className="text-xs font-semibold text-on-surface-variant">
          {isProcessing
            ? 'Calculating...'
            : `${doc.chunk_count.toLocaleString()} Chunks`}
        </span>
        <span className="text-xs text-outline italic">
          {new Date(doc.created_at).toLocaleDateString()}
        </span>
      </div>
    </div>
  );
}
