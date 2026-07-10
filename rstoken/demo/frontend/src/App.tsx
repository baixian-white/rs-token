import {
  Activity,
  Antenna,
  ArrowRight,
  Check,
  ChevronDown,
  CircleAlert,
  Cpu,
  ImagePlus,
  Layers3,
  LoaderCircle,
  LockKeyhole,
  Pause,
  Play,
  RadioTower,
  RefreshCw,
  ShieldCheck,
  Signal,
  Sparkles,
  Upload,
  Wifi,
  WifiOff,
  Zap,
} from "lucide-react";
import { ChangeEvent, useCallback, useEffect, useMemo, useRef, useState } from "react";
import type { Channel, Decision, InferenceResult, Priority, Protection, Sample } from "./types";

const TRACE = [10, 9, 7, 5, 2, 0, -2, 1, 4, 7, 10, 13, 8, 5, 3, 6, 9, 12];

const CHANNELS: Array<{ value: Channel; label: string }> = [
  { value: "none", label: "无信道" },
  { value: "awgn", label: "AWGN" },
  { value: "rayleigh", label: "Rayleigh" },
];

const PRIORITIES: Array<{ value: Priority; label: string }> = [
  { value: "alert", label: "语义告警" },
  { value: "balanced", label: "自适应" },
  { value: "detail", label: "重建优先" },
];

function formatBits(value: number) {
  return value >= 1000 ? `${(value / 1000).toFixed(value % 1000 === 0 ? 2 : 1)} Kb` : `${value} b`;
}

function formatPct(value: number) {
  return `${(value * 100).toFixed(value < 0.01 ? 3 : 1)}%`;
}

function Metric({ label, value, accent = false }: { label: string; value: string; accent?: boolean }) {
  return (
    <div className={`metric ${accent ? "metric-accent" : ""}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function Segmented<T extends string>({
  items,
  value,
  onChange,
}: {
  items: Array<{ value: T; label: string }>;
  value: T;
  onChange: (value: T) => void;
}) {
  return (
    <div className="segmented">
      {items.map((item) => (
        <button
          type="button"
          key={item.value}
          className={value === item.value ? "active" : ""}
          onClick={() => onChange(item.value)}
        >
          {item.label}
        </button>
      ))}
    </div>
  );
}

function SignalChart({ values, channel }: { values: number[]; channel: Channel }) {
  const width = 460;
  const height = 88;
  const min = -5;
  const max = 15;
  const points = values
    .map((value, index) => {
      const x = values.length === 1 ? width : (index / (values.length - 1)) * width;
      const y = height - ((value - min) / (max - min)) * height;
      return `${x.toFixed(1)},${Math.max(0, Math.min(height, y)).toFixed(1)}`;
    })
    .join(" ");
  return (
    <div className="signal-chart" aria-label="信道质量曲线">
      <svg viewBox={`0 0 ${width} ${height}`} preserveAspectRatio="none">
        <line x1="0" y1="44" x2={width} y2="44" className="chart-zero" />
        <polyline points={points} className={channel === "rayleigh" ? "trace fading" : "trace"} />
      </svg>
      <span className="chart-max">15</span>
      <span className="chart-min">-5</span>
    </div>
  );
}

function PredictionList({ title, items }: { title: string; items: InferenceResult["task_predictions"] }) {
  return (
    <section className="prediction-block">
      <div className="section-label">{title}</div>
      {items.map((item, index) => (
        <div className="prediction-row" key={item.class_id}>
          <span className="prediction-rank">0{index + 1}</span>
          <div className="prediction-name">
            <strong>{item.name_zh}</strong>
            <span>{item.name}</span>
          </div>
          <div className="score-track"><i style={{ width: `${item.score * 100}%` }} /></div>
          <b>{(item.score * 100).toFixed(1)}</b>
        </div>
      ))}
    </section>
  );
}

function App() {
  const [health, setHealth] = useState<{ status: string; gpu?: string; detail?: string }>({ status: "loading" });
  const [samples, setSamples] = useState<Sample[]>([]);
  const [sampleId, setSampleId] = useState("airport");
  const [upload, setUpload] = useState<File | null>(null);
  const [previewUrl, setPreviewUrl] = useState<string | null>(null);
  const [channel, setChannel] = useState<Channel>("awgn");
  const [snr, setSnr] = useState(5);
  const [protection, setProtection] = useState<Protection>("none");
  const [priority, setPriority] = useState<Priority>("balanced");
  const [autoK, setAutoK] = useState(true);
  const [manualK, setManualK] = useState(2);
  const [budget, setBudget] = useState(20480);
  const [seed, setSeed] = useState(42);
  const [result, setResult] = useState<InferenceResult | null>(null);
  const [policy, setPolicy] = useState<Decision | null>(null);
  const [activeLayer, setActiveLayer] = useState(1);
  const [loading, setLoading] = useState(false);
  const [linkUpdating, setLinkUpdating] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [live, setLive] = useState(false);
  const [trace, setTrace] = useState<number[]>([5, 5, 5, 5, 5, 5]);
  const traceIndex = useRef(0);
  const busyRef = useRef(false);
  const fileInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    let cancelled = false;
    const check = async () => {
      try {
        const response = await fetch("/api/health");
        const data = await response.json();
        if (!cancelled) setHealth(data);
        if (response.ok && !cancelled) {
          const sampleResponse = await fetch("/api/samples");
          setSamples(await sampleResponse.json());
          return;
        }
      } catch {
        if (!cancelled) setHealth({ status: "error", detail: "服务不可用" });
      }
      if (!cancelled) window.setTimeout(check, 1500);
    };
    check();
    return () => { cancelled = true; };
  }, []);

  useEffect(() => {
    if (!autoK) return;
    const params = new URLSearchParams({
      channel,
      snr_db: String(snr),
      protection,
      priority,
      max_transmitted_bits: String(budget),
      ...(result ? { previous_k: String(result.decision.k) } : {}),
    });
    fetch(`/api/policy?${params}`)
      .then((response) => response.json())
      .then(setPolicy)
      .catch(() => undefined);
  }, [autoK, budget, channel, priority, protection, result, snr]);

  const runTransmission = useCallback(async (overrideSnr?: number, background = false) => {
    if (health.status !== "ready" || busyRef.current) return;
    busyRef.current = true;
    if (background) setLinkUpdating(true);
    else setLoading(true);
    setError(null);
    const form = new FormData();
    if (upload) form.append("file", upload);
    else form.append("sample_id", sampleId);
    form.append("channel", channel);
    form.append("snr_db", String(overrideSnr ?? snr));
    form.append("protection", protection);
    form.append("priority", priority);
    form.append("max_transmitted_bits", String(budget));
    form.append("auto_k", String(autoK));
    form.append("manual_k", String(manualK));
    form.append("seed", String(seed));
    if (result) form.append("previous_k", String(result.decision.k));
    try {
      const response = await fetch("/api/infer", { method: "POST", body: form });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || "推理失败");
      setResult(data);
      setPolicy(data.decision);
      setActiveLayer(data.decision.k);
    } catch (cause) {
      setError(cause instanceof Error ? cause.message : "推理失败");
    } finally {
      if (background) setLinkUpdating(false);
      else setLoading(false);
      busyRef.current = false;
    }
  }, [autoK, budget, channel, health.status, manualK, priority, protection, result, sampleId, seed, snr, upload]);

  useEffect(() => {
    if (!live || loading || linkUpdating) return;
    const timer = window.setTimeout(() => {
      traceIndex.current = (traceIndex.current + 1) % TRACE.length;
      const next = TRACE[traceIndex.current];
      setSnr(next);
      setTrace((current) => [...current.slice(-17), next]);
      runTransmission(next, true);
    }, 2600);
    return () => window.clearTimeout(timer);
  }, [linkUpdating, live, loading, result, runTransmission]);

  const handleUpload = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    setUpload(file);
    setPreviewUrl(URL.createObjectURL(file));
    setLive(false);
  };

  const displayedImage = useMemo(() => {
    const found = result?.images.progressive.find((item) => item.k === activeLayer);
    return found?.image || result?.images.reconstruction || null;
  }, [activeLayer, result]);

  const sourcePreview = previewUrl || (sampleId ? `/api/samples/${sampleId}/image` : null);
  const currentK = autoK ? (policy?.k ?? 1) : manualK;
  const updatingFrame = linkUpdating || (loading && result !== null);
  const policyStops = useMemo(() => {
    if (channel === "none") return [-5, -5, -5, -5, 15];
    if (channel === "awgn" && protection === "ldpc") return [-5, -0.5, 1.5, 3.5, 15];
    if (channel === "awgn") return [-5, 1.5, 4, 6.5, 15];
    if (protection === "ldpc") return [-5, 3.5, 7, 10, 15];
    return [-5, 6.5, 9.5, 13, 15];
  }, [channel, protection]);

  return (
    <div className="app-shell">
      <header className="topbar">
        <div className="brand-block">
          <div className="brand-mark"><RadioTower size={19} /></div>
          <div>
            <strong>RS-TOKEN</strong>
            <span>ADAPTIVE LINK CONSOLE</span>
          </div>
        </div>
        <div className="mission-status">
          <span>链路任务</span>
          <strong>UAV-07 / 场景回传</strong>
        </div>
        <div className={`health ${health.status}`}>
          <i />
          <span>{health.status === "ready" ? "推理节点在线" : health.status === "error" ? "节点异常" : "模型装载中"}</span>
          {health.gpu && <b>{health.gpu.replace("NVIDIA GeForce ", "")}</b>}
        </div>
      </header>

      <main className="console-grid">
        <aside className="control-rail">
          <section className="control-section source-control">
            <div className="section-heading">
              <span>01</span><strong>遥感载荷</strong>
              <button className="icon-button" title="上传图像" onClick={() => fileInput.current?.click()}><Upload size={16} /></button>
              <input ref={fileInput} type="file" accept="image/*" hidden onChange={handleUpload} />
            </div>
            <button className="source-preview" onClick={() => fileInput.current?.click()}>
              {sourcePreview ? <img src={sourcePreview} alt="待传输遥感图像" /> : <ImagePlus size={28} />}
              <span>{upload ? upload.name : samples.find((item) => item.id === sampleId)?.name_zh || "选择图像"}</span>
            </button>
            <div className="sample-strip">
              {samples.slice(0, 6).map((sample) => (
                <button
                  type="button"
                  title={`${sample.name_zh} / ${sample.name}`}
                  className={!upload && sampleId === sample.id ? "active" : ""}
                  key={sample.id}
                  onClick={() => { setUpload(null); setPreviewUrl(null); setSampleId(sample.id); setLive(false); }}
                >
                  <img src={`/api/samples/${sample.id}/image`} alt={sample.name_zh} />
                </button>
              ))}
            </div>
          </section>

          <section className="control-section">
            <div className="section-heading"><span>02</span><strong>物理信道</strong></div>
            <Segmented items={CHANNELS} value={channel} onChange={(value) => { setChannel(value); if (value === "none") setSnr(15); }} />
            <div className="slider-row">
              <div><span>SNR</span><strong>{channel === "none" ? "∞" : snr.toFixed(1)} <small>dB</small></strong></div>
              <input
                aria-label="SNR"
                type="range"
                min="-5"
                max="15"
                step="0.5"
                value={snr}
                disabled={channel === "none" || live}
                onChange={(event) => { setSnr(Number(event.target.value)); setTrace((current) => [...current.slice(-17), Number(event.target.value)]); }}
              />
              <div className="range-labels"><span>-5</span><span>0</span><span>5</span><span>10</span><span>15</span></div>
            </div>
            <label className="select-row">
              <span>链路保护</span>
              <div><ShieldCheck size={15} /><select value={protection} onChange={(e) => setProtection(e.target.value as Protection)}><option value="none">无编码</option><option value="ldpc">LDPC 1/2</option></select><ChevronDown size={14} /></div>
            </label>
            <label className="select-row">
              <span>每帧预算</span>
              <div><Zap size={15} /><select value={budget} onChange={(e) => setBudget(Number(e.target.value))}><option value="5120">5.12 Kb</option><option value="10240">10.24 Kb</option><option value="20480">20.48 Kb</option></select><ChevronDown size={14} /></div>
            </label>
          </section>

          <section className="control-section">
            <div className="section-heading"><span>03</span><strong>传输策略</strong></div>
            <Segmented items={PRIORITIES} value={priority} onChange={setPriority} />
            <div className="switch-row">
              <div><Sparkles size={16} /><span>自动前缀</span></div>
              <button type="button" className={`switch ${autoK ? "on" : ""}`} onClick={() => setAutoK(!autoK)} aria-label="自动前缀"><i /></button>
            </div>
            <div className="layer-selector">
              {[1, 2, 3, 4].map((k) => (
                <button
                  type="button"
                  key={k}
                  disabled={autoK}
                  onClick={() => setManualK(k)}
                  className={currentK === k ? "active" : currentK > k ? "included" : ""}
                >
                  <span>L{k - 1}</span><strong>{k}</strong>
                </button>
              ))}
            </div>
            <div className="decision-note">
              <Activity size={16} />
              <p>{policy?.reason || "等待信道遥测数据"}</p>
            </div>
          </section>

          <div className="action-stack">
            <button className="primary-action" onClick={() => runTransmission()} disabled={loading || linkUpdating || health.status !== "ready"}>
              {loading ? <LoaderCircle className="spin" size={18} /> : <Antenna size={18} />}
              <span>{loading ? "链路处理中" : "执行传输"}</span>
              {!loading && <ArrowRight size={17} />}
            </button>
            <button className={`live-action ${live ? "active" : ""}`} onClick={() => { setLive(!live); if (!live && !result) runTransmission(); }} disabled={health.status !== "ready"}>
              {live ? <Pause size={16} /> : <Play size={16} />}
              {live ? "暂停动态链路" : "启动动态链路"}
            </button>
          </div>
        </aside>

        <section className="workspace">
          <div className="workspace-header">
            <div>
              <span className="eyebrow">RECEIVER / FRAME 042</span>
              <h1>渐进式遥感重建</h1>
            </div>
            <div className="layer-flow">
              {[1, 2, 3, 4].map((k) => (
                <div key={k} className={k <= (result?.decision.k ?? currentK) ? "active" : ""}>
                  <span>L{k - 1}</span><i />
                </div>
              ))}
            </div>
          </div>

          <div className="image-stage">
            <div className="image-pane">
              <div className="image-label"><span>TX / SOURCE</span><b>256 × 256 RGB</b></div>
              {result?.images.input || sourcePreview ? <img src={result?.images.input || sourcePreview!} alt="发射端原始图像" /> : <div className="empty-image"><ImagePlus size={30} /></div>}
            </div>
            <div className="transmission-axis">
              <span>{result ? formatBits(result.decision.transmitted_bits) : "--"}</span>
              <i><ArrowRight size={17} /></i>
              <b>{result ? `${result.channel.bit_errors} bit errors` : "待传输"}</b>
            </div>
            <div className="image-pane result-pane">
              <div className="image-label"><span>RX / PREFIX K={activeLayer}</span><b>{result ? `${result.metrics.psnr_db.toFixed(2)} dB` : "PSNR --"}</b></div>
              {displayedImage ? <img key={displayedImage} className="recon-image" src={displayedImage} alt="接收端重建图像" /> : <div className="empty-image"><RadioTower size={31} /><span>RX BUFFER EMPTY</span></div>}
              {updatingFrame && <div className="link-update"><i /><span>RX NEXT FRAME</span></div>}
              {loading && !result && <div className="image-loading"><LoaderCircle className="spin" size={24} /><span>正在通过信道</span></div>}
            </div>
          </div>

          <div className="progressive-timeline">
            <div className="timeline-title"><Layers3 size={16} /><span>接收前缀</span></div>
            {[1, 2, 3, 4].map((k) => {
              const item = result?.images.progressive.find((entry) => entry.k === k);
              return (
                <button key={k} disabled={!item} className={activeLayer === k ? "active" : ""} onClick={() => setActiveLayer(k)}>
                  <span>k={k}</span>
                  <strong>{formatBits(k * 2560)}</strong>
                  <i>{item ? <Check size={13} /> : <LockKeyhole size={12} />}</i>
                </button>
              );
            })}
          </div>

          <div className="signal-band">
            <div className="band-heading">
              <div><Signal size={17} /><strong>信道遥测</strong></div>
              <span className={live ? "live" : ""}><i />{live ? "LIVE" : "HOLD"}</span>
            </div>
            <SignalChart values={trace} channel={channel} />
            <div className="band-readout">
              <span>当前</span><strong>{channel === "none" ? "∞" : `${snr.toFixed(1)} dB`}</strong>
              <span>自动档位</span><strong>k={currentK}</strong>
            </div>
          </div>

          <div className="adaptive-panel">
            <section className="policy-map">
              <div className="panel-title"><Signal size={16} /><strong>自适应阈值带</strong><span>{protection === "ldpc" ? "LDPC 1/2" : "UNCODED"}</span></div>
              <div className="policy-scale">
                {[1, 2, 3, 4].map((k) => {
                  const start = policyStops[k - 1];
                  const end = policyStops[k];
                  const width = Math.max(0, ((end - start) / 20) * 100);
                  return <div key={k} className={`policy-zone zone-${k}`} style={{ width: `${width}%` }}><b>k={k}</b></div>;
                })}
                {channel !== "none" && <i className="policy-marker" style={{ left: `${Math.max(0, Math.min(100, ((snr + 5) / 20) * 100))}%` }}><span>{snr.toFixed(1)}</span></i>}
              </div>
              <div className="policy-axis"><span>-5 dB</span><span>0</span><span>5</span><span>10</span><span>15 dB</span></div>
              <div className="policy-caption"><span>可靠性优先</span><span>重建质量优先</span></div>
            </section>
            <section className="frame-schedule">
              <div className="panel-title"><Layers3 size={16} /><strong>当前帧调度</strong><span>FRAME 042</span></div>
              <div className="schedule-head"><span>层</span><span>载荷角色</span><span>比特</span><span>状态</span></div>
              {[1, 2, 3, 4].map((k) => (
                <div className={k <= (result?.decision.k ?? currentK) ? "scheduled" : "held"} key={k}>
                  <b>L{k - 1}</b>
                  <span>{k === 1 ? "场景语义" : k === 2 ? "主体结构" : k === 3 ? "纹理细节" : "高频残差"}</span>
                  <code>2.56 Kb</code>
                  <em>{k <= (result?.decision.k ?? currentK) ? "已调度" : "保持"}</em>
                </div>
              ))}
            </section>
          </div>

          {error && <div className="error-banner"><CircleAlert size={17} /><span>{error}</span><button onClick={() => setError(null)}><RefreshCw size={15} /></button></div>}
        </section>

        <aside className="telemetry-rail">
          <section className="telemetry-section status-panel">
            <div className="section-label">链路状态</div>
            <div className={`quality-orbit k${result?.decision.k ?? currentK}`}>
              <div><span>k</span><strong>{result?.decision.k ?? currentK}</strong></div>
            </div>
            <strong className="quality-name">{result?.decision.quality || policy?.quality || "等待决策"}</strong>
            <span className="quality-channel">{channel === "none" ? "IDEAL LINK" : `${channel.toUpperCase()} / ${snr.toFixed(1)} dB`}</span>
          </section>

          <section className="telemetry-section metrics-grid">
            <div className="section-label">通信指标</div>
            <Metric label="源载荷" value={result ? formatBits(result.decision.source_bits) : "--"} accent />
            <Metric label="信道比特" value={result ? formatBits(result.decision.transmitted_bits) : "--"} />
            <Metric label="原始 BER" value={result ? formatPct(result.channel.raw_ber) : "--"} />
            <Metric label="译码后 BER" value={result ? formatPct(result.channel.post_ber) : "--"} />
            <Metric label="带宽节省" value={result ? `${result.metrics.bandwidth_saving_pct.toFixed(0)}%` : "--"} />
            <Metric label="端到端" value={result ? `${result.metrics.total_ms.toFixed(1)} ms` : "--"} />
          </section>

          <section className="telemetry-section">
            <div className="section-label">误码空间分布 / 16 × 16</div>
            <div className="error-map">
              {(result?.channel.error_grid || Array(256).fill(0)).map((count, index) => (
                <i key={index} className={count > 5 ? "severe" : count > 2 ? "medium" : count > 0 ? "low" : ""} title={`token ${index}: ${count} bit errors`} />
              ))}
            </div>
            <div className="map-legend"><span><i />无误码</span><span><i className="low" />轻微</span><span><i className="severe" />严重</span></div>
          </section>

          <PredictionList title="L0 语义任务" items={result?.task_predictions || []} />
          <PredictionList title="重建图判读" items={result?.recon_predictions || []} />

          <footer className="model-footer">
            <Cpu size={15} />
            <div><span>DEPLOYED MODEL</span><strong>10.87 M params / 38.31 GFLOPs</strong></div>
          </footer>
        </aside>
      </main>
    </div>
  );
}

export default App;
