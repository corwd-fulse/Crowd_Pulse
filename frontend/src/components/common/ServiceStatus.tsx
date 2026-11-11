
import { useEffect, useState } from 'react';
import { getHealth } from '../../services/api';

type ConnectionStatus = 'loading' | 'connected' | 'disconnected';

export default function ServerStatus() {
  const [status, setStatus] = useState<ConnectionStatus>('loading');

  useEffect(() => {
    let active = true;

    async function checkConnection() {
      try {
        const response = await getHealth();

        if (active) {
          setStatus(response.status === 'ok' ? 'connected' : 'disconnected');
        }
      } catch {
        if (active) {
          setStatus('disconnected');
        }
      }
    }

    void checkConnection();

    return () => {
      active = false;
    };
  }, []);

  const statusConfig = {
    loading: {
      label: '서버 확인 중',
      color: 'bg-slate-400',
    },
    connected: {
      label: 'API 연결됨',
      color: 'bg-emerald-500',
    },
    disconnected: {
      label: 'API 연결 끊김',
      color: 'bg-rose-500',
    },
  };

  const current = statusConfig[status];

  return (
    <div
      role="status"
      className="flex items-center gap-2 text-xs text-slate-400"
    >
      <span className={`h-2 w-2 rounded-full ${current.color}`} />
      <span>{current.label}</span>
    </div>
  );
}
