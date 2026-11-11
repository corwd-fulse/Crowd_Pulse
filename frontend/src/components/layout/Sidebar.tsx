import { NavLink } from 'react-router-dom';
import { navGroups, navItems } from './navigation';

function Brand() {
  return (
    <div className="flex items-center gap-2.5">
      {/* 심박 파형 모양의 로고 마크 */}
      <svg
        width="22"
        height="22"
        viewBox="0 0 22 22"
        fill="none"
        aria-hidden="true"
        className="shrink-0 text-white"
      >
        <polyline
          points="1,12 6,12 9,4 13,18 16,12 21,12"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="square"
          strokeLinejoin="miter"
        />
      </svg>
      <div className="leading-none">
        <p className="text-[15px] font-bold tracking-wide text-white">
          CROWD PULSE
        </p>
        <p className="mt-1 text-[11px] text-side-text">인파 분석·모니터링</p>
      </div>
    </div>
  );
}

export default function Sidebar() {
  return (
    <>
      {/* 태블릿·모바일: 상단 바 + 가로 스크롤 메뉴 */}
      <div className="bg-side lg:hidden">
        <div className="px-4 pt-3 pb-2 md:px-6">
          <Brand />
        </div>
        <nav
          aria-label="주 메뉴"
          className="flex gap-1 overflow-x-auto px-2 [scrollbar-width:none] md:px-4 [&::-webkit-scrollbar]:hidden"
        >
          {navItems.map((item) => (
            <NavLink
              key={item.path}
              to={item.path}
              className={({ isActive }) =>
                `shrink-0 border-b-2 px-3 py-3 text-[13px] whitespace-nowrap transition-colors ${
                  isActive
                    ? 'border-white font-semibold text-white'
                    : 'border-transparent text-side-text hover:text-white'
                }`
              }
            >
              {item.label}
            </NavLink>
          ))}
        </nav>
      </div>

      {/* 데스크톱: 고정 사이드바 */}
      <aside className="hidden min-h-screen w-56 shrink-0 self-stretch flex-col bg-side lg:flex">
        <div className="px-5 pt-6 pb-8">
          <Brand />
        </div>

        <nav aria-label="주 메뉴" className="flex-1 overflow-y-auto">
          {navGroups.map((group) => (
            <div key={group.title} className="mb-6">
              <p className="px-5 pb-1.5 text-[11px] font-semibold tracking-wider text-side-text/70">
                {group.title}
              </p>
              <ul>
                {group.items.map((item) => (
                  <li key={item.path}>
                    <NavLink
                      to={item.path}
                      className={({ isActive }) =>
                        `block border-l-2 py-2 pr-5 pl-[18px] text-[13.5px] transition-colors ${
                          isActive
                            ? 'border-[#70b8ad] bg-white/[0.07] font-semibold text-white'
                            : 'border-transparent text-side-text hover:bg-white/[0.04] hover:text-white'
                        }`
                      }
                    >
                      {item.label}
                    </NavLink>
                  </li>
                ))}
              </ul>
            </div>
          ))}
        </nav>

        <div className="border-t border-side-line px-5 py-4">
          <p className="text-[13px] text-side-text/70">설정 (준비 중)</p>
          <p className="mt-3 text-[11px] text-side-text/60">
            CROWD PULSE AI 2.0
          </p>
        </div>
      </aside>
    </>
  );
}
