interface StartCardProps {
  title: string;
  value: string;
  description: string;
  color?: 'cyan' | 'amber' | 'rose';
  featured?: boolean;
  meter?: number;
}

export default function StartCard({ title, value, description, color = 'cyan', featured = false, meter }: StartCardProps) {
  const match = value.match(/^([\d.,]+)(.*)$/);
  const number = match?.[1] ?? value;
  const unit = match?.[2] ?? '';
  const accent = color === 'amber' ? 'var(--color-warn)' : color === 'rose' ? 'var(--color-danger)' : 'var(--color-brand)';

  return (
    <div className="instrument-card" style={{ '--instrument-color': accent } as React.CSSProperties}>
      <p className="text-xs font-medium tracking-wide text-ink-soft">{title}</p>
      <div className="mt-4 flex items-center justify-between gap-3">
        <div>
          <p className="num instrument-value text-4xl font-semibold tracking-tight">
            {number}<span className="ml-1 text-sm font-normal text-muted">{unit}</span>
          </p>
          <p className="mt-3 text-[11px] text-muted">{description}</p>
        </div>
        {meter !== undefined ? (
          <svg viewBox="0 0 100 100" className="instrument-gauge h-20 w-20 shrink-0" role="meter" aria-label={title} aria-valuemin={0} aria-valuemax={100} aria-valuenow={meter}>
            <circle cx="50" cy="50" r="47" fill="none" stroke="#29739a" strokeWidth="0.6" strokeDasharray="1 3" />
            <circle cx="50" cy="50" r="41" fill="none" stroke="#163e65" strokeWidth="8" strokeDasharray="193.2 257.6" transform="rotate(135 50 50)" />
            <circle cx="50" cy="50" r="41" fill="none" stroke={accent} strokeWidth="8" strokeDasharray={`${meter * 1.932} 257.6`} transform="rotate(135 50 50)" />
            <circle cx="50" cy="50" r="30" fill="none" stroke="#245579" strokeWidth="1" strokeDasharray="1 5" />
            <text x="50" y="55" textAnchor="middle" fill={accent} fontSize="17" fontWeight="600">{meter}%</text>
          </svg>
        ) : (
          <div className="metric-emblem" aria-hidden="true">
            <svg viewBox="0 0 64 64" className="h-14 w-14" fill="none" stroke={accent} strokeWidth="1.3">
              <path d="m32 3 25 14v30L32 61 7 47V17Z" opacity="0.25" />
              {featured ? <><circle cx="32" cy="24" r="7"/><path d="M19 46v-5a13 13 0 0 1 26 0v5M15 25a5 5 0 0 0 0 10m34-10a5 5 0 0 1 0 10"/></> : color === 'amber' ? <><path d="M19 17v10l9 9v12m17-31v10l-9 9v12M15 17h8m18 0h8M24 48h8m0 0h8"/><circle cx="32" cy="31" r="3"/></> : <><path d="m32 16 17 30H15Z"/><path d="M32 26v10m0 4v2"/></>}
            </svg>
          </div>
        )}
      </div>
      <div className="instrument-rail" aria-hidden="true" />
    </div>
  );
}
