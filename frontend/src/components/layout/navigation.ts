// 사이드바 메뉴와 헤더 제목이 같은 목록을 쓰도록 한 곳에서 관리한다.
export interface NavItem {
  label: string;
  path: string;
  description: string;
}

export interface NavGroup {
  title: string;
  items: NavItem[];
}

export const navGroups: NavGroup[] = [
  {
    title: '관제',
    items: [
      {
        label: '종합 대시보드',
        path: '/dashboard',
        description: '실시간 인파 현황과 혼잡 위험을 확인합니다.',
      },
      {
        label: '실시간 모니터링',
        path: '/monitoring',
        description: 'CCTV 영상과 AI 객체 탐지 결과를 확인합니다.',
      },
    ],
  },
  {
    title: '분석',
    items: [
      {
        label: '인파 흐름 분석',
        path: '/flow',
        description: '인파의 이동 방향과 병목 현상을 분석합니다.',
      },
      {
        label: '혼잡도 예측',
        path: '/prediction',
        description: '시계열 AI 기반 혼잡도 예측 결과를 확인합니다.',
      },
    ],
  },
  {
    title: '이벤트',
    items: [
      {
        label: '위험 이벤트',
        path: '/alerts',
        description: '인파 밀집 및 위험 이벤트 발생 기록을 관리합니다.',
      },
    ],
  },
];

export const navItems: NavItem[] = navGroups.flatMap((group) => group.items);
