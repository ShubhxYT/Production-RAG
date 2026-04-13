import { useNavigate } from 'react-router-dom';

const suggestions = [
  { icon: 'article', label: 'Quarterly Review' },
  { icon: 'data_object', label: 'API Documentation' },
  { icon: 'verified_user', label: 'Security Protocols' },
];

export default function SuggestionChips() {
  const navigate = useNavigate();

  return (
    <div className="flex flex-wrap justify-center gap-3">
      {suggestions.map((s) => (
        <button
          key={s.label}
          onClick={() =>
            navigate(`/results?q=${encodeURIComponent(s.label)}`)
          }
          className="px-5 py-2.5 bg-slate-800 hover:bg-slate-700 text-slate-300 rounded-full text-sm font-medium transition-all flex items-center gap-2"
        >
          <span className="material-symbols-outlined text-base">
            {s.icon}
          </span>
          {s.label}
        </button>
      ))}
    </div>
  );
}
