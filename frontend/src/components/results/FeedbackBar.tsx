interface Props {
  onFeedback: (type: 'thumbs_up' | 'thumbs_down') => void;
  submitted: boolean;
}

export default function FeedbackBar({ onFeedback, submitted }: Props) {
  if (submitted) {
    return (
      <div className="flex items-center justify-center gap-2 py-4 text-emerald-400 text-sm font-medium">
        <span className="material-symbols-outlined text-sm">check_circle</span>
        Thanks for your feedback
      </div>
    );
  }

  return (
    <div className="flex items-center justify-center gap-4 py-4">
      <span className="text-xs text-slate-500 uppercase tracking-widest font-label">
        Was this helpful?
      </span>
      <button
        onClick={() => onFeedback('thumbs_up')}
        className="p-2 rounded-full hover:bg-slate-800/50 text-slate-400 hover:text-emerald-400 transition-colors"
      >
        <span className="material-symbols-outlined">thumb_up</span>
      </button>
      <button
        onClick={() => onFeedback('thumbs_down')}
        className="p-2 rounded-full hover:bg-slate-800/50 text-slate-400 hover:text-red-400 transition-colors"
      >
        <span className="material-symbols-outlined">thumb_down</span>
      </button>
    </div>
  );
}
