import { Bell } from 'lucide-react';
import { Link, useLocation } from 'react-router-dom';
import { navItems } from './navigation';

const dateFormatter = new Intl.DateTimeFormat('ko-KR', {
  year: 'numeric',
  month: 'long',
  day: 'numeric',
  weekday: 'long',
});

export default function Header() {
  const { pathname } = useLocation();
  const current = navItems.find((item) => item.path === pathname) ?? navItems[0];
  const today = dateFormatter.format(new Date());

  return (
    <header className="flex flex-wrap items-end justify-between gap-x-6 gap-y-3 border-b border-line pb-4">
      <div className="min-w-0">
        <h1 className="text-2xl font-bold tracking-tight text-ink">
          {current.label}
        </h1>
        <p className="mt-1 text-[13px] text-muted">{current.description}</p>
      </div>

      <div className="flex items-center gap-4 text-[13px]">
        <span className="hidden text-muted md:inline">{today}</span>

        <span className="rounded-sm border border-warn/40 bg-warn-soft px-2 py-0.5 text-xs font-semibold text-warn">
          시연 모드
        </span>

        <Link
          to="/alerts"
          aria-label="알림"
          className="rounded-sm border border-line-strong bg-surface p-1.5 text-ink-soft transition-colors hover:bg-sunken hover:text-ink"
        >
          <Bell size={16} />
        </Link>
      </div>
    </header>
  );
}
