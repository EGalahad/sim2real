import React, {useMemo, useState} from 'react';
import data from '../../data/leaderboard.json';
import styles from './styles.module.css';

type Row = (typeof data)[number];
type MetricKey =
  | 'bodyPos'
  | 'bodyOri'
  | 'globalRoot'
  | 'gpuHours'
  | 'wristPos'
  | 'wristOri'
  | 'trackingReturn'
  | 'progress';

const columns: Array<{
  key: MetricKey;
  en: string;
  zh: string;
  unit: string;
  higher: boolean;
  digits: number;
}> = [
  {key: 'bodyPos', en: 'Body Pos', zh: '身体位置', unit: 'mm', higher: false, digits: 2},
  {key: 'bodyOri', en: 'Body Ori', zh: '身体姿态', unit: 'rad', higher: false, digits: 3},
  {key: 'globalRoot', en: 'Global Root', zh: '全局 Root', unit: 'm', higher: false, digits: 3},
  {key: 'gpuHours', en: 'GPU Hours', zh: 'GPU 时长', unit: 'h', higher: false, digits: 1},
  {key: 'wristPos', en: 'Wrist Pos', zh: '手腕位置', unit: 'mm', higher: false, digits: 2},
  {key: 'wristOri', en: 'Wrist Ori', zh: '手腕姿态', unit: 'rad', higher: false, digits: 3},
  {key: 'trackingReturn', en: 'Tracking Return', zh: 'Tracking Return', unit: '', higher: true, digits: 3},
  {key: 'progress', en: 'Progress', zh: '完成度', unit: '%', higher: true, digits: 1},
];

const copy = {
  en: {
    rank: 'Rank', policy: 'Policy', legend: 'L = LAFAN-40, P = PHUMA-30, R = Root-90',
  },
  zh: {
    rank: '排名', policy: 'Policy', legend: 'L = LAFAN-40，P = PHUMA-30，R = Root-90',
  },
};

function metricValue(row: Row, key: MetricKey): number | null {
  return row.metrics[key].mean;
}

function formatValue(value: number | null, digits: number): string {
  return value == null ? '—' : value.toFixed(digits);
}

function metricLabel(row: Row, key: MetricKey, digits: number): React.ReactNode {
  const value = metricValue(row, key);
  const text = formatValue(value, digits);
  if (key === 'gpuHours' && value != null && row.metrics.gpuHours.sourceUrl) {
    return <a href={row.metrics.gpuHours.sourceUrl} target="_blank" rel="noreferrer">{text}</a>;
  }
  return text;
}

function splitLine(row: Row, key: MetricKey, digits: number): string {
  const splits = row.metrics[key].splits;
  if (splits.lafan == null && splits.phuma == null && splits.root90 == null) {
    return '—';
  }
  return [
    `L ${formatValue(splits.lafan, digits)}`,
    `P ${formatValue(splits.phuma, digits)}`,
    `R ${formatValue(splits.root90, digits)}`,
  ].join(' · ');
}

export default function Leaderboard({locale = 'en'}: {locale?: 'en' | 'zh'}) {
  const [sortKey, setSortKey] = useState<MetricKey>('bodyPos');
  const text = copy[locale];
  const selected = columns.find((option) => option.key === sortKey)!;
  const rows = useMemo(() => [...(data as Row[])].sort((a, b) => {
    const left = metricValue(a, sortKey);
    const right = metricValue(b, sortKey);
    if (left == null) return 1;
    if (right == null) return -1;
    return selected.higher ? right - left : left - right;
  }), [sortKey, selected.higher]);

  return <>
    <p className={styles.legend}>{text.legend}</p>
    <div className={styles.tableWrap}>
      <table className={styles.table}>
        <thead><tr>
          <th>{text.rank}</th>
          <th>{text.policy}</th>
          {columns.map((column) => <th key={column.key}>
            <button
              type="button"
              className={sortKey === column.key ? styles.sortButtonActive : styles.sortButton}
              onClick={() => setSortKey(column.key)}
            >
              {locale === 'zh' ? column.zh : column.en}
              {column.unit ? ` (${column.unit})` : ''}
              {column.higher ? ' ↑' : ' ↓'}
            </button>
          </th>)}
        </tr></thead>
        <tbody>{rows.map((row, index) => <tr key={row.key} className={index < 3 ? styles.podium : ''}>
          <td className={styles.rank}>{metricValue(row, sortKey) == null ? '—' : index + 1}</td>
          <td className={styles.policy}><a href={row.url} target="_blank" rel="noreferrer">{row.name}</a></td>
          {columns.map((column) => <td key={column.key} className={sortKey === column.key ? styles.sorted : ''}>
            <strong>{metricLabel(row, column.key, column.digits)}</strong>
            <small>{splitLine(row, column.key, column.digits)}</small>
          </td>)}
        </tr>)}</tbody>
      </table>
    </div>
  </>;
}
