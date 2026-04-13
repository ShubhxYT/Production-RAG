import { useQuery } from '@tanstack/react-query';
import { fetchEvaluationSummary } from '../../lib/api';

export default function HealthMetrics() {
  const { data } = useQuery({
    queryKey: ['evaluation'],
    queryFn: () => fetchEvaluationSummary(),
    retry: false,
  });

  const metrics = [
    {
      label: 'Positive Rate',
      value: data ? `${(data.positive_rate * 100).toFixed(1)}%` : '—',
      bar: data ? data.positive_rate * 100 : 0,
      color: 'bg-tertiary',
    },
    {
      label: 'Avg Rating',
      value: data?.avg_rating != null ? data.avg_rating.toFixed(1) : '—',
      suffix: data?.avg_rating != null ? '/ 5.0' : '',
      bar: data?.avg_rating != null ? (data.avg_rating / 5) * 100 : 0,
      color: 'bg-primary',
    },
    {
      label: 'Total Feedback',
      value: data?.total_feedback?.toString() ?? '—',
      suffix: '',
      bar: Math.min(((data?.total_feedback ?? 0) / 100) * 100, 100),
      color: 'bg-indigo-400',
    },
    {
      label: 'Feedback Mix',
      value: data ? 'Active' : '—',
      suffix: '',
      bar: 0,
      color: 'bg-tertiary',
      segments: data
        ? [
            (data.counts_by_type?.thumbs_up ?? 0) > 0,
            (data.counts_by_type?.thumbs_down ?? 0) > 0,
            (data.counts_by_type?.correction ?? 0) > 0,
            false,
          ]
        : [false, false, false, false],
    },
  ];

  return (
    <section className="bg-slate-900/50 rounded-xl p-8 lg:p-12 relative overflow-hidden">
      <div className="relative z-10">
        <div className="flex items-center gap-3 mb-8">
          <div className="p-2 bg-tertiary-container text-on-tertiary-container rounded-lg">
            <span className="material-symbols-outlined">analytics</span>
          </div>
          <h2 className="text-xl font-bold font-headline">
            Knowledge Base Health
          </h2>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-8">
          {metrics.map((m) => (
            <div key={m.label} className="space-y-2">
              <span className="text-sm font-medium text-outline uppercase tracking-wider">
                {m.label}
              </span>
              <div className="flex items-end gap-2">
                <span className="text-4xl font-extrabold text-on-surface">
                  {m.value}
                </span>
                {m.suffix && (
                  <span className="text-on-surface-variant text-sm font-medium mb-1">
                    {m.suffix}
                  </span>
                )}
              </div>
              {m.segments ? (
                <div className="flex gap-1 mt-4">
                  {m.segments.map((filled, i) => (
                    <div
                      key={i}
                      className={`flex-1 h-1.5 rounded-full ${
                        filled ? m.color : 'bg-slate-800'
                      }`}
                    />
                  ))}
                </div>
              ) : (
                <div className="w-full bg-slate-800 rounded-full h-1.5 mt-4">
                  <div
                    className={`${m.color} h-1.5 rounded-full transition-all duration-1000`}
                    style={{ width: `${m.bar}%` }}
                  />
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Decorative Gradient */}
      <div className="absolute top-0 right-0 w-96 h-96 opacity-20 pointer-events-none transform translate-x-1/4 -translate-y-1/4 bg-gradient-to-br from-indigo-500/30 to-transparent blur-3xl" />
    </section>
  );
}
