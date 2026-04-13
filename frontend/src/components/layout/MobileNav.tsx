import { NavLink } from 'react-router-dom';

const tabs = [
  { to: '/', icon: 'search_spark', label: 'Query', end: true },
  { to: '/library', icon: 'library_books', label: 'Library', end: false },
  { to: '#history', icon: 'history', label: 'History', end: false },
  { to: '#health', icon: 'health_and_safety', label: 'Health', end: false },
];

export default function MobileNav() {
  return (
    <nav className="md:hidden fixed bottom-0 w-full bg-slate-950/90 backdrop-blur-xl flex justify-around items-center py-4 px-4 z-50 border-t border-slate-800/50">
      {tabs.map((tab) => (
        <NavLink
          key={tab.label}
          to={tab.to}
          end={tab.end}
          className={({ isActive }) =>
            `flex flex-col items-center gap-1 ${
              isActive ? 'text-indigo-400' : 'text-slate-500'
            }`
          }
        >
          {({ isActive }) => (
            <>
              <span
                className="material-symbols-outlined"
                style={
                  isActive
                    ? { fontVariationSettings: "'FILL' 1" }
                    : undefined
                }
              >
                {tab.icon}
              </span>
              <span className="text-[10px] font-bold">{tab.label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  );
}
