import React, {useMemo, useState} from 'react';
import data from '../../data/leaderboard.json';
import styles from './styles.module.css';

type SortKey = 'trackingReturn' | 'localError' | 'wristError' | 'globalRootError';
type Row = (typeof data)[number];

const options: Array<{key: SortKey; en: string; zh: string; higher: boolean}> = [
  {key: 'localError', en: 'Local Error', zh: '局部误差', higher: false},
  {key: 'trackingReturn', en: 'Tracking Return', zh: 'Tracking Return', higher: true},
  {key: 'globalRootError', en: 'Global Root', zh: '全局 Root', higher: false},
  {key: 'wristError', en: 'Wrist Error', zh: '手腕误差', higher: false},
];

const copy = {
  en: {
    sort: 'Rank by', rank: 'Rank', policy: 'Policy', tracking: 'Tracking Return ↑',
    local: 'Local Error ↓', wrist: 'Wrist Error ↓', root: 'Global Root ↓', progress: 'Progress ↑',
  },
  zh: {
    sort: '排序指标', rank: '排名', policy: 'Policy', tracking: 'Tracking Return ↑',
    local: '局部误差 ↓', wrist: '手腕误差 ↓', root: '全局 Root ↓', progress: '完成度 ↑',
  },
};

export default function Leaderboard({locale = 'en'}: {locale?: 'en' | 'zh'}) {
  const [sortKey, setSortKey] = useState<SortKey>('localError');
  const text = copy[locale];
  const selected = options.find((option) => option.key === sortKey)!;
  const rows = useMemo(() => [...(data as Row[])].sort((a, b) => {
    const left = a[sortKey];
    const right = b[sortKey];
    if (left == null) return 1;
    if (right == null) return -1;
    return selected.higher ? right - left : left - right;
  }), [sortKey, selected.higher]);

  return <>
    <div className={styles.controls} role="group" aria-label={text.sort}>
      <span>{text.sort}</span>
      {options.map((option) => <button
        type="button"
        key={option.key}
        className={sortKey === option.key ? styles.active : ''}
        aria-pressed={sortKey === option.key}
        onClick={() => setSortKey(option.key)}
      >{locale === 'zh' ? option.zh : option.en}{option.higher ? ' ↑' : ' ↓'}</button>)}
    </div>
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead><tr>
          <th>{text.rank}</th><th>{text.policy}</th><th>{text.tracking}</th>
          <th>{text.local}</th><th>{text.wrist}</th><th>{text.root}</th><th>{text.progress}</th>
        </tr></thead>
        <tbody>{rows.map((row, index) => <tr key={row.key} className={index < 3 ? styles.podium : ''}>
          <td className={styles.rank}>{row[sortKey] == null ? '—' : index + 1}</td>
          <td className={styles.policy}><a href={row.url} target="_blank" rel="noreferrer">{row.name}</a></td>
          <td className={sortKey === 'trackingReturn' ? styles.sorted : ''}>
            <strong>L {row.lafanReturn.toFixed(3)} · P {row.phumaReturn.toFixed(3)} · R {row.root90Return.toFixed(3)}</strong>
            <small>LAFAN · PHUMA · Root-90</small>
          </td>
          <td className={sortKey === 'localError' ? styles.sorted : ''}>
            <strong>L {row.lafanLocal.toFixed(2)} · P {row.phumaLocal.toFixed(2)} · R {row.root90Local.toFixed(2)}</strong>
            <small>mm · LAFAN / PHUMA / Root-90</small>
          </td>
          <td className={sortKey === 'wristError' ? styles.sorted : ''}>
            <strong>L {row.lafanWrist.toFixed(2)} · P {row.phumaWrist.toFixed(2)} · R {row.root90Wrist.toFixed(2)}</strong>
            <small>mm · LAFAN / PHUMA / Root-90</small>
          </td>
          <td className={sortKey === 'globalRootError' ? styles.sorted : ''}>
            <strong>L {row.lafanGlobalRoot.toFixed(3)} · P {row.phumaGlobalRoot.toFixed(3)} · R {row.root90GlobalRoot.toFixed(3)}</strong>
            <small>m · LAFAN / PHUMA / Root-90</small>
          </td>
          <td>
            <strong>L {row.lafanProgress.toFixed(1)}% · P {row.phumaProgress.toFixed(1)}% · R {row.root90Progress.toFixed(1)}%</strong>
            <small>LAFAN · PHUMA · Root-90</small>
          </td>
        </tr>)}</tbody>
      </table>
    </div>
  </>;
}
