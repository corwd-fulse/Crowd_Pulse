import {render,screen} from '@testing-library/react';
import {expect,it,vi} from 'vitest';
import VenueControlTower from '../src/components/dashboard/VenueControlTower';
import {emptyVenue,getVenue} from '../src/services/venueControl';

vi.mock('../src/services/venueControl',async original=>({...await original<typeof import('../src/services/venueControl')>(),getVenue:vi.fn()}));

it('renders saved unmapped zones while the selected job is still loading',async()=>{
  vi.mocked(getVenue).mockResolvedValue({venue_id:'main',revision:1,created_at:'2026-11-17T00:00:00Z',configuration:{...emptyVenue,zones:[{id:'a',name:'A',polygon:[[0,0],[100,0],[100,100]],color:'#44b6c8',mapping:null,capacity:null,visible:true,note:''}]}});
  render(<VenueControlTower job={null}/>);
  expect(await screen.findByRole('button',{name:'구역 A 선택'})).toBeTruthy();
  expect(screen.getByLabelText('지도 공통 타임라인').getAttribute('disabled')).not.toBeNull();
});
