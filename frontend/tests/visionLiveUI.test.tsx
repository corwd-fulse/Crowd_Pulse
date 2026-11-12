import {render, screen} from '@testing-library/react';
import {expect, it, vi} from 'vitest';
import type {ReactElement} from 'react';
import {MemoryRouter} from 'react-router-dom';
import Dashboard from '../src/pages/DashBoard';

// Keep real API requests and real chart calculation; supply layout dimensions
// because jsdom has no browser layout engine.
vi.mock('recharts', async (importOriginal) => {
  const react = await import('react');
  return {...await importOriginal<typeof import('recharts')>(),
    ResponsiveContainer: ({children}: {children: ReactElement<{width: number; height: number}>}) => react.cloneElement(children, {width:800,height:240}),
  };
});

it.skipIf(process.env.RUN_LIVE_API !== '1')('renders the actual persisted result through real GET requests without mock fallback', async () => {
  const id='9bfe14ca-8932-465e-9347-4a50c0fa4904';
  const {container}=render(<MemoryRouter initialEntries={[`/dashboard?job=${id}`]}><Dashboard/></MemoryRouter>);
  await screen.findByText('테스트 분석 결과',{}, {timeout:15000});
  expect(screen.getByText('미승인 ROI')).toBeTruthy();
  expect(screen.getByText('방향 품질 미검증')).toBeTruthy();
  expect(screen.getByText(/전체 34행/)).toBeTruthy();
  expect(container.querySelectorAll('.recharts-line-curve')).toHaveLength(2);
  expect(container.querySelector('[data-source="mock"]')).toBeNull();
  expect(screen.getByRole('link',{name:/bottleneck_scores.*4행/})).toBeTruthy();
},20000);
