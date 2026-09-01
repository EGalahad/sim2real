import React, {useEffect, useMemo, useRef, useState} from 'react';
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
type DatasetKey =
  | 'locomotion'
  | 'manipulation'
  | 'ground'
  | 'dance'
  | 'lafan'
  | 'phuma'
  | 'root90';

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

const datasets: Array<{key: DatasetKey; en: string; zh: string; short: string}> = [
  {key: 'locomotion', en: 'Locomotion', zh: '移动', short: 'Loc'},
  {key: 'manipulation', en: 'Manipulation', zh: '操作', short: 'Man'},
  {key: 'ground', en: 'Ground', zh: '起身', short: 'Gnd'},
  {key: 'dance', en: 'Dance', zh: '舞蹈', short: 'Dnc'},
  {key: 'lafan', en: 'Legacy LAFAN-40', zh: '旧版 LAFAN-40', short: 'L'},
  {key: 'phuma', en: 'Legacy PHUMA-30', zh: '旧版 PHUMA-30', short: 'P'},
  {key: 'root90', en: 'Legacy Root-90', zh: '旧版 Root-90', short: 'R'},
];

const copy = {
  en: {
    rank: 'Rank', policy: 'Policy', rankHint: 'Click any metric header to rank.',
    plotTitle: 'Build a comparison', policies: 'Policies', metrics: 'Metrics', datasets: 'Datasets',
    plotHint: 'Each metric uses its own scale; GPU Hours is logarithmic. Missing values are n/a.',
    tableLabel: 'Scrollable policy leaderboard',
  },
  zh: {
    rank: '排名', policy: 'Policy', rankHint: '点击任意指标表头排序。',
    plotTitle: '自定义对比', policies: '策略', metrics: '指标', datasets: '数据集',
    plotHint: '每个指标使用独立坐标轴；GPU Hours 使用对数轴；缺失值标记为 n/a。',
    tableLabel: '可滚动策略排行榜',
  },
};

const defaultPolicies = new Set([
  'mimic_lite_roa', 'mimic_lite_ppo', 'sonic_g1', 'heft', 'holomotion',
]);
const defaultMetrics = new Set<MetricKey>([
  'bodyPos', 'globalRoot', 'gpuHours', 'wristPos', 'trackingReturn',
]);
const defaultDatasets = new Set<DatasetKey>([
  'locomotion', 'manipulation',
]);
const palette = [
  '#f1d36b', '#2f6236', '#5e9d5d', '#9eb875', '#d9826b', '#4f86a8', '#8b6bb1',
  '#d6a14d', '#4d8b7f', '#b96b71', '#7397bf', '#af8c72', '#7c9e52', '#6f6f6f',
];

function metricValue(row: Row, key: MetricKey, selectedDatasets: Set<DatasetKey>): number | null {
  if (key === 'gpuHours') return row.metrics.gpuHours.mean;
  const values = [...selectedDatasets]
    .map((dataset) => row.metrics[key].datasets[dataset])
    .filter((value): value is number => value != null);
  return values.length ? values.reduce((sum, value) => sum + value, 0) / values.length : null;
}

function formatValue(value: number | null, digits: number): string {
  return value == null ? '—' : value.toFixed(digits);
}

function metricLabel(row: Row, key: MetricKey, digits: number, selectedDatasets: Set<DatasetKey>): React.ReactNode {
  const value = metricValue(row, key, selectedDatasets);
  const text = formatValue(value, digits);
  if (key === 'gpuHours' && value != null && row.metrics.gpuHours.sourceUrl) {
    return <a href={row.metrics.gpuHours.sourceUrl} target="_blank" rel="noreferrer">{text}</a>;
  }
  return text;
}

function splitLine(row: Row, key: MetricKey, digits: number, selectedDatasets: Set<DatasetKey>): string {
  if (key === 'gpuHours') return '—';
  return datasets
    .filter((dataset) => selectedDatasets.has(dataset.key))
    .map((dataset) => `${dataset.short} ${formatValue(row.metrics[key].datasets[dataset.key], digits)}`)
    .join(' · ');
}

function toggleSet<T>(current: Set<T>, value: T): Set<T> {
  const next = new Set(current);
  next.has(value) ? next.delete(value) : next.add(value);
  return next.size ? next : current;
}

function ComparisonCanvas({
  rows,
  selectedPolicies,
  selectedMetrics,
  selectedDatasets,
  locale,
}: {
  rows: Row[];
  selectedPolicies: Set<string>;
  selectedMetrics: Set<MetricKey>;
  selectedDatasets: Set<DatasetKey>;
  locale: 'en' | 'zh';
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const wrapRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    const wrap = wrapRef.current;
    if (!canvas || !wrap) return;

    const draw = () => {
      const policies = rows.filter((row) => selectedPolicies.has(row.key));
      const metrics = columns.filter((column) => selectedMetrics.has(column.key));
      const width = Math.max(240, wrap.clientWidth);
      const compact = width < 600;
      const panelColumns = compact
        ? Math.min(2, metrics.length)
        : Math.max(1, Math.floor((width - 32) / 270));
      const panelRows = Math.ceil(metrics.length / panelColumns);
      const panelWidth = (width - 32) / panelColumns;
      const legendRows = Math.ceil(policies.length / Math.max(1, Math.floor(width / 175)));
      const footerHeight = 45 + legendRows * 30;
      const panelHeight = Math.max(1, (wrap.clientHeight - footerHeight) / panelRows);
      const legendTop = panelRows * panelHeight + 30;
      const height = legendTop + 15 + legendRows * 30;
      const ratio = window.devicePixelRatio || 1;
      canvas.width = width * ratio;
      canvas.height = height * ratio;
      canvas.style.width = `${width}px`;
      canvas.style.height = `${height}px`;
      const ctx = canvas.getContext('2d');
      if (!ctx) return;
      ctx.scale(ratio, ratio);
      ctx.fillStyle = '#f4f2ec';
      ctx.fillRect(0, 0, width, height);
      ctx.strokeStyle = 'rgba(55, 59, 52, 0.07)';
      ctx.lineWidth = 1;
      for (let x = 0; x <= width; x += 20) {
        ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, height); ctx.stroke();
      }
      for (let y = 0; y <= height; y += 20) {
        ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(width, y); ctx.stroke();
      }

      metrics.forEach((metric, metricIndex) => {
        const panelColumn = metricIndex % panelColumns;
        const panelRow = Math.floor(metricIndex / panelColumns);
        const x0 = 16 + panelColumn * panelWidth;
        const rowTop = panelRow * panelHeight;
        const top = rowTop + Math.min(54, panelHeight * .25);
        const bottom = rowTop + panelHeight - Math.min(36, panelHeight * .12);
        const chartHeight = bottom - top;
        const plotLeft = x0 + (compact ? 12 : 58);
        const plotRight = x0 + panelWidth - (compact ? 8 : 14);
        const values = policies
          .map((row) => metricValue(row, metric.key, selectedDatasets))
          .filter((value): value is number => value != null && value > 0);
        const logScale = metric.key === 'gpuHours';
        const maximum = Math.max(...values, 1) * 1.12;
        const logMinimumExponent = values.length
          ? Math.floor(Math.log10(Math.min(...values) / 1.12))
          : 0;
        const logMaximumExponent = values.length
          ? Math.max(logMinimumExponent + 1, Math.ceil(Math.log10(Math.max(...values) * 1.12)))
          : 1;
        ctx.font = `${compact ? '700 9px' : '700 15px'} ui-monospace, SFMono-Regular, Menlo, monospace`;
        ctx.fillStyle = '#171913';
        ctx.textAlign = 'center';
        ctx.fillText(
          `${locale === 'zh' ? metric.zh : metric.en}${metric.unit ? ` / ${metric.unit}` : ''}`,
          (plotLeft + plotRight) / 2,
          rowTop + Math.min(28, panelHeight * .15),
        );
        ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
        const tickCount = logScale ? logMaximumExponent - logMinimumExponent : 4;
        if (compact) {
          ctx.strokeStyle = '#171913';
          ctx.beginPath(); ctx.moveTo(plotLeft, bottom); ctx.lineTo(plotRight, bottom); ctx.stroke();
        } else {
          for (let tick = 0; tick <= tickCount; tick += 1) {
            const value = logScale
              ? 10 ** (logMinimumExponent + tick)
              : maximum * (tick / tickCount);
            const y = bottom - chartHeight * (tick / tickCount);
            ctx.strokeStyle = tick === 0 ? '#171913' : 'rgba(65, 67, 58, 0.18)';
            ctx.beginPath(); ctx.moveTo(plotLeft, y); ctx.lineTo(plotRight, y); ctx.stroke();
            ctx.fillStyle = '#6f716b';
            ctx.textAlign = 'right';
            const tickLabel = logScale && value >= 1000 ? `${value / 1000}k` : formatValue(value, metric.digits);
            ctx.fillText(tickLabel, plotLeft - 8, y + 4);
          }
        }
        const slot = (plotRight - plotLeft) / policies.length;
        const barWidth = Math.min(34, slot * 0.62);
        policies.forEach((row, policyIndex) => {
          const value = metricValue(row, metric.key, selectedDatasets);
          const x = plotLeft + slot * policyIndex + (slot - barWidth) / 2;
          if (value == null) {
            ctx.fillStyle = '#7b7d76';
            ctx.textAlign = 'center';
            ctx.fillText('n/a', x + barWidth / 2, bottom - 8);
            return;
          }
          const barRatio = logScale
            ? (Math.log10(value) - logMinimumExponent) / (logMaximumExponent - logMinimumExponent)
            : value / maximum;
          const barHeight = chartHeight * Math.max(0, barRatio);
          const y = bottom - barHeight;
          ctx.fillStyle = '#171913';
          ctx.fillRect(x + 4, y + 5, barWidth, barHeight);
          ctx.fillStyle = palette[rows.indexOf(row) % palette.length];
          ctx.fillRect(x, y, barWidth, barHeight);
          ctx.strokeStyle = '#171913';
          ctx.lineWidth = 1.5;
          ctx.strokeRect(x, y, barWidth, barHeight);
          if (!compact) {
            ctx.save();
            ctx.translate(x + barWidth / 2, Math.max(top + 12, y - 7));
            ctx.rotate(-Math.PI / 2);
            ctx.fillStyle = '#171913';
            ctx.textAlign = 'left';
            ctx.fillText(formatValue(value, metric.digits), 0, 4);
            ctx.restore();
          }
        });
      });

      const perRow = Math.max(1, Math.floor(width / 175));
      policies.forEach((row, index) => {
        const x = 28 + (index % perRow) * 175;
        const y = legendTop + Math.floor(index / perRow) * 30;
        ctx.fillStyle = palette[rows.indexOf(row) % palette.length];
        ctx.fillRect(x, y - 13, 18, 18);
        ctx.strokeStyle = '#171913';
        ctx.strokeRect(x, y - 13, 18, 18);
        ctx.fillStyle = '#171913';
        ctx.font = '12px ui-monospace, SFMono-Regular, Menlo, monospace';
        ctx.textAlign = 'left';
        ctx.fillText(row.name, x + 26, y + 1);
      });
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(wrap);
    return () => observer.disconnect();
  }, [locale, rows, selectedDatasets, selectedMetrics, selectedPolicies]);

  return <div ref={wrapRef} className={styles.canvasWrap}>
    <canvas ref={canvasRef} aria-label="Custom policy comparison bar chart" />
  </div>;
}

export default function Leaderboard({locale = 'en'}: {locale?: 'en' | 'zh'}) {
  const [sortKey, setSortKey] = useState<MetricKey>('bodyPos');
  const [selectedPolicies, setSelectedPolicies] = useState(defaultPolicies);
  const [selectedMetrics, setSelectedMetrics] = useState(defaultMetrics);
  const [selectedDatasets, setSelectedDatasets] = useState(defaultDatasets);
  const plotStageRef = useRef<HTMLDivElement>(null);
  const plotRef = useRef<HTMLElement>(null);
  const tableStageRef = useRef<HTMLDivElement>(null);
  const tableRef = useRef<HTMLDivElement>(null);
  const text = copy[locale];
  const selected = columns.find((option) => option.key === sortKey)!;
  const rows = useMemo(() => [...(data as Row[])].sort((a, b) => {
    const left = metricValue(a, sortKey, selectedDatasets);
    const right = metricValue(b, sortKey, selectedDatasets);
    if (left == null) return 1;
    if (right == null) return -1;
    return selected.higher ? right - left : left - right;
  }), [selectedDatasets, sortKey, selected.higher]);

  const datasetLegend = datasets
    .filter((dataset) => selectedDatasets.has(dataset.key))
    .map((dataset) => `${dataset.short} = ${locale === 'zh' ? dataset.zh : dataset.en}`)
    .join(locale === 'zh' ? '，' : ', ');

  useEffect(() => {
    let animationFrame = 0;
    const updateScrollProgress = () => {
      animationFrame = 0;
      const plotStage = plotStageRef.current;
      const plot = plotRef.current;
      const tableStage = tableStageRef.current;
      const table = tableRef.current;
      if (!plotStage || !plot || !tableStage || !table) return;

      const navbarHeight =
        document.querySelector<HTMLElement>('.navbar')?.getBoundingClientRect().height ?? 0;
      const viewportHeight = Math.max(1, window.innerHeight - navbarHeight);
      const canvas = plot.querySelector('canvas')?.parentElement;
      const canvasTravel = canvas ? Math.max(0, canvas.scrollHeight - canvas.clientHeight) : 0;
      const plotTravel = viewportHeight * .35;
      plotStage.style.height = `${viewportHeight + canvasTravel + plotTravel}px`;
      const plotStart = plotStage.getBoundingClientRect().top + window.scrollY - navbarHeight;
      const plotOffset = Math.max(0, window.scrollY - plotStart);
      if (canvas) canvas.scrollTop = Math.min(canvasTravel, plotOffset);
      const plotProgress = Math.min(
        1,
        Math.max(0, (plotOffset - canvasTravel) / plotTravel),
      );
      const easedProgress = Math.min(
        1,
        Math.max(0, (plotProgress - .25) / .5),
      );
      if (window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
        plot.style.transform = 'none';
        plot.style.opacity = '1';
      } else {
        plot.style.transform = `scale(${1 - easedProgress * .025}) translateY(${-easedProgress * 8}px)`;
        plot.style.opacity = `${1 - easedProgress * .12}`;
      }

      const tableTravel = Math.max(0, table.scrollHeight - table.clientHeight);
      tableStage.style.height = `${viewportHeight + tableTravel}px`;
      const tableStart = tableStage.getBoundingClientRect().top + window.scrollY - navbarHeight;
      table.scrollTop = Math.min(tableTravel, Math.max(0, window.scrollY - tableStart));
    };
    const scheduleUpdate = () => {
      if (!animationFrame) animationFrame = window.requestAnimationFrame(updateScrollProgress);
    };
    const observer = new ResizeObserver(scheduleUpdate);
    if (plotRef.current) observer.observe(plotRef.current);
    if (tableRef.current) observer.observe(tableRef.current);
    window.addEventListener('scroll', scheduleUpdate, {passive: true});
    window.addEventListener('resize', scheduleUpdate);
    updateScrollProgress();
    return () => {
      observer.disconnect();
      window.removeEventListener('scroll', scheduleUpdate);
      window.removeEventListener('resize', scheduleUpdate);
      if (animationFrame) window.cancelAnimationFrame(animationFrame);
    };
  }, []);

  return <>
    <div ref={plotStageRef} className={styles.plotStage}>
    <section ref={plotRef} className={styles.plotBuilder}>
      <div className={styles.plotHeader}>
        <h2>{text.plotTitle}</h2>
        <p>{text.plotHint}</p>
      </div>
      <fieldset className={styles.selectorRow}>
        <legend>{text.policies}</legend>
        <div className={styles.chips}>{(data as Row[]).map((row) => <label key={row.key} className={styles.chip}>
          <input
            type="checkbox"
            checked={selectedPolicies.has(row.key)}
            onChange={() => setSelectedPolicies((current) => toggleSet(current, row.key))}
          />
          <span>{row.name}</span>
        </label>)}</div>
      </fieldset>
      <fieldset className={styles.selectorRow}>
        <legend>{text.metrics}</legend>
        <div className={styles.chips}>{columns.map((metric) => <label key={metric.key} className={styles.chip}>
          <input
            type="checkbox"
            checked={selectedMetrics.has(metric.key)}
            onChange={() => setSelectedMetrics((current) => toggleSet(current, metric.key))}
          />
          <span>{locale === 'zh' ? metric.zh : metric.en}</span>
        </label>)}</div>
      </fieldset>
      <ComparisonCanvas
        rows={data as Row[]}
        selectedPolicies={selectedPolicies}
        selectedMetrics={selectedMetrics}
        selectedDatasets={selectedDatasets}
        locale={locale}
      />
    </section>
    </div>
    <fieldset className={styles.datasetBar}>
      <legend>{text.datasets}</legend>
      <div className={styles.datasetGroup}>
        <strong>MotionDecode</strong>
        <div className={styles.chips}>{datasets.slice(0, 4).map((dataset) => <label key={dataset.key} className={styles.chip}>
          <input
            type="checkbox"
            checked={selectedDatasets.has(dataset.key)}
            onChange={() => setSelectedDatasets((current) => toggleSet(current, dataset.key))}
          />
          <span>{dataset.short} · {locale === 'zh' ? dataset.zh : dataset.en}</span>
        </label>)}</div>
      </div>
      <div className={styles.datasetGroup}>
        <strong>Legacy</strong>
        <div className={styles.chips}>{datasets.slice(4).map((dataset) => <label key={dataset.key} className={styles.chip}>
          <input
            type="checkbox"
            checked={selectedDatasets.has(dataset.key)}
            onChange={() => setSelectedDatasets((current) => toggleSet(current, dataset.key))}
          />
          <span>{dataset.short} · {locale === 'zh' ? dataset.zh : dataset.en}</span>
        </label>)}</div>
      </div>
    </fieldset>
    <p className={styles.legend}>{datasetLegend} · {text.rankHint}</p>
    <div ref={tableStageRef} className={styles.tableStage}>
    <div
      ref={tableRef}
      className={styles.tableWrap}
      tabIndex={0}
      aria-label={text.tableLabel}
      onKeyDown={(event) => {
        const delta = event.key === 'ArrowDown' ? 48
          : event.key === 'ArrowUp' ? -48
          : event.key === 'PageDown' ? window.innerHeight * .8
          : event.key === 'PageUp' ? -window.innerHeight * .8
          : 0;
        if (delta) { event.preventDefault(); window.scrollBy(0, delta); }
      }}
    >
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
          <td className={styles.rank}>{metricValue(row, sortKey, selectedDatasets) == null ? '—' : index + 1}</td>
          <td className={styles.policy}><a href={row.url} target="_blank" rel="noreferrer">{row.name}</a></td>
          {columns.map((column) => <td key={column.key} className={sortKey === column.key ? styles.sorted : ''}>
            <strong>{metricLabel(row, column.key, column.digits, selectedDatasets)}</strong>
            <small>{splitLine(row, column.key, column.digits, selectedDatasets)}</small>
          </td>)}
        </tr>)}</tbody>
      </table>
    </div>
    </div>
  </>;
}
