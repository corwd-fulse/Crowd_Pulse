import { Link } from 'react-router-dom';

interface ComingSoonProps {
  title: string;
  description: string;
}

export default function ComingSoon({
  title,
  description,
}: ComingSoonProps) {
  return (
    <section className="max-w-2xl pt-5">
      <p className="text-xs font-semibold text-warn">개발 준비 중</p>

      <h2 className="mt-2 text-lg font-bold text-ink">
        {title} 화면은 아직 구현되지 않았습니다.
      </h2>

      <p className="mt-2 text-sm leading-relaxed text-muted">{description}</p>

      <Link
        to="/dashboard"
        className="mt-6 inline-block rounded-sm border border-line-strong bg-surface px-3.5 py-2 text-[13px] font-semibold text-ink transition-colors hover:bg-sunken"
      >
        종합 대시보드로 이동
      </Link>
    </section>
  );
}
