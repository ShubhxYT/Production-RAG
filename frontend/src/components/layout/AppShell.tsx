import { type ReactNode } from 'react';
import Sidebar from './Sidebar';
import MobileNav from './MobileNav';

export default function AppShell({ children }: { children: ReactNode }) {
  return (
    <div className="min-h-screen bg-slate-950 text-on-surface">
      <Sidebar />

      {/* Mobile Header */}
      <header className="md:hidden fixed top-0 w-full z-50 bg-slate-950/80 backdrop-blur-xl px-6 py-3 flex justify-between items-center">
        <h1 className="text-xl font-bold text-indigo-400 font-headline tracking-tight">
          FullRag
        </h1>
        <div className="flex items-center gap-4">
          <span className="material-symbols-outlined text-slate-400">
            settings
          </span>
          <div className="w-8 h-8 rounded-full bg-slate-800" />
        </div>
      </header>

      {/* Main Content */}
      <main className="flex-1 md:ml-64 min-h-screen flex flex-col pt-14 md:pt-0 pb-20 md:pb-0">
        {children}
      </main>

      <MobileNav />
    </div>
  );
}
