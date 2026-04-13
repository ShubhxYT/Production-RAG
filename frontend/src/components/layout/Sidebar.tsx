import { NavLink, useNavigate } from 'react-router-dom';

export default function Sidebar() {
  const navigate = useNavigate();

  const linkClass = ({ isActive }: { isActive: boolean }) =>
    `flex items-center gap-3 font-headline text-sm font-medium px-4 py-3 rounded-xl transition-all duration-300 ${
      isActive
        ? 'bg-indigo-500/10 text-indigo-300'
        : 'text-slate-400 hover:text-indigo-300 hover:bg-slate-800/50'
    }`;

  return (
    <aside className="hidden md:flex flex-col h-screen w-64 bg-slate-900/80 backdrop-blur-xl fixed left-0 top-0 py-8 px-4 z-50">
      {/* Logo */}
      <div className="mb-10 px-2">
        <div className="flex items-center gap-3">
          <div className="w-10 h-10 rounded-xl bg-indigo-600 flex items-center justify-center text-white shadow-lg">
            <span
              className="material-symbols-outlined"
              style={{ fontVariationSettings: "'FILL' 1" }}
            >
              auto_awesome
            </span>
          </div>
          <div>
            <h1 className="text-lg font-black text-slate-50 font-headline leading-tight">
              FullRag
            </h1>
            <p className="text-[10px] text-slate-500 font-medium tracking-widest uppercase">
              The Intellectual Sanctuary
            </p>
          </div>
        </div>
      </div>

      {/* New Research CTA */}
      <button
        onClick={() => navigate('/')}
        className="w-full py-3 mb-6 primary-gradient text-white rounded-xl font-headline font-bold text-sm tracking-wide shadow-md hover:scale-[0.98] transition-transform"
      >
        New Research
      </button>

      {/* Navigation */}
      <nav className="flex-1 space-y-1">
        <NavLink to="/" end className={linkClass}>
          <span className="material-symbols-outlined">explore</span>
          Discover
        </NavLink>
        <NavLink to="/library" className={linkClass}>
          <span
            className="material-symbols-outlined"
            style={{ fontVariationSettings: "'FILL' 1" }}
          >
            library_books
          </span>
          Library
        </NavLink>
        <button className="w-full flex items-center gap-3 font-headline text-sm font-medium text-slate-400 hover:text-indigo-300 hover:bg-slate-800/50 px-4 py-3 rounded-xl transition-all duration-300">
          <span className="material-symbols-outlined">folder_shared</span>
          Collections
        </button>
        <button className="w-full flex items-center gap-3 font-headline text-sm font-medium text-slate-400 hover:text-indigo-300 hover:bg-slate-800/50 px-4 py-3 rounded-xl transition-all duration-300">
          <span className="material-symbols-outlined">history</span>
          History
        </button>
      </nav>

      {/* Bottom Section */}
      <div className="mt-auto space-y-4 pt-6 border-t border-slate-800/50">
        <div className="space-y-1">
          <button className="w-full flex items-center gap-3 px-4 py-2 rounded-xl text-slate-500 hover:text-indigo-300 hover:bg-slate-800/50 transition-all duration-300">
            <span className="material-symbols-outlined">settings</span>
            <span className="font-headline text-sm">Settings</span>
          </button>
          <button className="w-full flex items-center gap-3 px-4 py-2 rounded-xl text-slate-500 hover:text-indigo-300 hover:bg-slate-800/50 transition-all duration-300">
            <span className="material-symbols-outlined">help</span>
            <span className="font-headline text-sm">Support</span>
          </button>
        </div>
        <div className="flex items-center gap-2 px-4 py-2 opacity-50">
          <div className="w-2 h-2 rounded-full bg-emerald-500 shadow-[0_0_8px_rgba(16,185,129,0.6)] animate-pulse" />
          <span className="text-[10px] font-label font-medium uppercase tracking-tighter">
            System Healthy
          </span>
        </div>
      </div>
    </aside>
  );
}
