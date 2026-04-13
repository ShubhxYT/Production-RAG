import { useState, type FormEvent } from 'react';
import { useNavigate } from 'react-router-dom';

export default function SearchInput() {
  const [query, setQuery] = useState('');
  const navigate = useNavigate();

  const handleSubmit = (e: FormEvent) => {
    e.preventDefault();
    if (query.trim()) {
      navigate(`/results?q=${encodeURIComponent(query.trim())}`);
    }
  };

  return (
    <form onSubmit={handleSubmit} className="relative group">
      <div className="absolute inset-0 bg-indigo-400/5 blur-xl rounded-xl opacity-0 group-focus-within:opacity-100 transition-opacity duration-500" />
      <div className="relative flex items-center bg-slate-900 shadow-sm border border-transparent focus-within:border-indigo-500/20 rounded-xl p-2 transition-all duration-300">
        <div className="pl-4 pr-2 text-slate-400">
          <span className="material-symbols-outlined text-2xl">search</span>
        </div>
        <input
          value={query}
          onChange={(e) => setQuery(e.target.value)}
          className="w-full bg-transparent border-none focus:ring-0 text-lg font-body py-4 placeholder:text-slate-400 text-white"
          placeholder="Ask anything about your documents..."
          type="text"
          autoFocus
        />
        <div className="flex items-center gap-2 pr-2">
          <kbd className="hidden md:flex items-center gap-1 px-2 py-1 bg-slate-800 text-slate-400 rounded-md text-[10px] font-bold">
            ⌘ K
          </kbd>
          <button
            type="submit"
            className="bg-indigo-600 hover:bg-indigo-700 text-white p-3 rounded-xl transition-all active:scale-95 shadow-md"
          >
            <span className="material-symbols-outlined">arrow_forward</span>
          </button>
        </div>
      </div>
    </form>
  );
}
