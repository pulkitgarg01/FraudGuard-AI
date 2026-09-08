import type { PredictResponse, ExplanationFactor } from '../../types';
import RiskBadge from './RiskBadge';
import GaugeChart from './GaugeChart';
import { CardContainer, CardBody, CardItem } from './3d-card';
import { ShieldCheck, ShieldAlert, AlertTriangle, TrendingUp, TrendingDown, Info } from 'lucide-react';

interface Props {
  result: PredictResponse;
}

const RISK_PROB_COLORS: Record<string, string> = {
  LOW: '#10b981',
  MEDIUM: '#f59e0b',
  HIGH: '#ef4444',
};

// ── SHAP Factor Row ────────────────────────────────────────────────────────────
function FactorRow({ factor, color }: { factor: ExplanationFactor; color: string }) {
  const barWidth = `${Math.round(factor.relative_impact * 100)}%`;
  return (
    <div style={{ marginBottom: 10 }}>
      <div style={{ marginBottom: 4 }}>
        <span style={{ fontSize: 13, fontWeight: 500, color: 'var(--text-primary)' }}>
          {factor.feature}
        </span>
      </div>
      {/* Relative impact bar — proportional width only, NOT a probability */}
      <div style={{
        height: 5,
        borderRadius: 99,
        background: 'rgba(255,255,255,0.06)',
        overflow: 'hidden',
      }}>
        <div style={{
          height: '100%',
          width: barWidth,
          borderRadius: 99,
          background: color,
          opacity: 0.85,
          transition: 'width 0.4s ease',
        }} />
      </div>
    </div>
  );
}

// ── Main ResultPanel ───────────────────────────────────────────────────────────
export default function ResultPanel({ result }: Props) {
  const isLow    = result.risk_level === 'LOW';
  const isMedium = result.risk_level === 'MEDIUM';
  const isHigh   = result.risk_level === 'HIGH';

  const bannerClass  = isLow ? 'legitimate' : isMedium ? 'medium' : 'fraud';
  const verdictLabel = isLow
    ? 'Transaction Legitimate'
    : isMedium
    ? 'Suspicious Activity'
    : 'Fraud Detected';

  const probPercent = (result.fraud_probability * 100).toFixed(2);
  const color       = RISK_PROB_COLORS[result.risk_level] || '#6366f1';

  const explanation = result.explanation;
  const hasExplain  = explanation?.available &&
    ((explanation.top_risk_factors?.length ?? 0) > 0 ||
     (explanation.top_protective_factors?.length ?? 0) > 0);

  return (
    <CardContainer className="w-full">
      <CardBody className={`card-3d-body--${result.risk_level.toLowerCase()}`}>

        {/* Verdict Banner */}
        <CardItem translateZ="15" className="w-full">
          <div className={`result-verdict-banner ${bannerClass}`}>
            <div className={`verdict-icon ${bannerClass}`}>
              {isLow    && <ShieldCheck  size={22} color="white" strokeWidth={2.5} />}
              {isMedium && <AlertTriangle size={22} color="white" strokeWidth={2.5} />}
              {isHigh   && <ShieldAlert  size={22} color="white" strokeWidth={2.5} />}
            </div>
            <div>
              <div className={`verdict-label ${bannerClass}`}>{verdictLabel}</div>
              <div className="verdict-sub">
                ID: <span style={{ fontFamily: 'monospace', fontSize: 12 }}>{result.transaction_id}</span>
                &nbsp;&nbsp;·&nbsp;&nbsp;
                <RiskBadge level={result.risk_level} size="sm" />
              </div>
            </div>
          </div>
        </CardItem>

        {/* Metrics Row */}
        <div className="result-metrics-row" style={{ marginTop: 14 }}>

          <CardItem translateZ="22" className="w-full">
            <div className="result-metric-cell" style={{ padding: '18px 16px', height: '100%' }}>
              <GaugeChart value={result.risk_score} riskLevel={result.risk_level} label="Risk Score" size={135} />
            </div>
          </CardItem>

          <CardItem translateZ="18" className="w-full">
            <div className="result-metric-cell" style={{ flexDirection: 'column', alignItems: 'flex-start', padding: '18px 22px', height: '100%' }}>
              <div className="result-metric-label">Fraud Probability</div>
              <div className="result-metric-value" style={{ color, fontSize: 34, marginTop: 6 }}>
                {probPercent}<span style={{ fontSize: 18, fontWeight: 500, color: 'var(--text-muted)' }}>%</span>
              </div>
              <div className="prob-bar-track" style={{ width: '100%', marginTop: 12 }}>
                <div
                  className="prob-bar-fill"
                  style={{ width: `${result.fraud_probability * 100}%`, background: color }}
                />
              </div>
              <div style={{ display: 'flex', justifyContent: 'space-between', width: '100%', marginTop: 4 }}>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>0%</span>
                <span style={{ fontSize: 10, color: 'var(--text-muted)' }}>100%</span>
              </div>
            </div>
          </CardItem>

          <CardItem translateZ="12" className="w-full">
            <div className="result-metric-cell" style={{ flexDirection: 'column', alignItems: 'flex-start', padding: '18px 22px', gap: 10, height: '100%' }}>
              <div className="result-metric-label">Risk Distribution &amp; Level</div>
              <div style={{ marginTop: 2 }}>
                <RiskBadge level={result.risk_level} />
              </div>
              <div style={{ marginTop: 6, display: 'flex', flexDirection: 'column', gap: 8, width: '100%' }}>
                {['LOW', 'MEDIUM', 'HIGH'].map((lvl) => (
                  <div key={lvl} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                    <div style={{
                      width: 7, height: 7, borderRadius: '50%',
                      background: RISK_PROB_COLORS[lvl],
                      opacity: result.risk_level === lvl ? 1 : 0.25,
                    }} />
                    <span style={{
                      fontSize: 11,
                      fontWeight: result.risk_level === lvl ? 700 : 400,
                      color: result.risk_level === lvl ? 'var(--text-primary)' : 'var(--text-muted)',
                    }}>
                      {lvl} {lvl === 'LOW' ? '(0–30)' : lvl === 'MEDIUM' ? '(31–70)' : '(71–100)'}
                    </span>
                  </div>
                ))}
              </div>
            </div>
          </CardItem>

        </div>

        {/* ── SHAP Explanation Panel ────────────────────────────────────────── */}
        {hasExplain && (
          <CardItem translateZ="8" className="w-full" style={{ marginTop: 16 }}>
            <div style={{
              background: 'rgba(255,255,255,0.03)',
              border: '1px solid var(--border-base)',
              borderRadius: 12,
              padding: '18px 20px',
            }}>

              {/* Header */}
              <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 16 }}>
                <span style={{ fontSize: 13, fontWeight: 700, color: 'var(--text-primary)', letterSpacing: '0.02em' }}>
                  WHY THIS PREDICTION?
                </span>
                <span
                  title="Factors are generated using SHAP model explanations. They describe why the XGBoost model produced this output, not why fraud occurred in reality."
                  style={{ cursor: 'help', color: 'var(--text-muted)' }}
                >
                  <Info size={13} />
                </span>
              </div>

              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 20 }}>

                {/* Increasing Risk */}
                <div>
                  {(explanation!.top_risk_factors?.length ?? 0) > 0 ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>
                        <TrendingUp size={13} color="#ef4444" />
                        <span style={{ fontSize: 11, fontWeight: 600, color: '#ef4444', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Increasing Risk
                        </span>
                      </div>
                      {explanation!.top_risk_factors.map((f) => (
                        <FactorRow key={f.feature} factor={f} color="#ef4444" />
                      ))}
                    </>
                  ) : (
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      No significant risk factors identified.
                    </span>
                  )}
                </div>

                {/* Reducing Risk */}
                <div>
                  {(explanation!.top_protective_factors?.length ?? 0) > 0 ? (
                    <>
                      <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 12 }}>
                        <TrendingDown size={13} color="#10b981" />
                        <span style={{ fontSize: 11, fontWeight: 600, color: '#10b981', textTransform: 'uppercase', letterSpacing: '0.06em' }}>
                          Reducing Risk
                        </span>
                      </div>
                      {explanation!.top_protective_factors.map((f) => (
                        <FactorRow key={f.feature} factor={f} color="#10b981" />
                      ))}
                    </>
                  ) : (
                    <span style={{ fontSize: 12, color: 'var(--text-muted)' }}>
                      No significant protective factors identified.
                    </span>
                  )}
                </div>

              </div>

              {/* Honest disclaimer */}
              <div style={{ marginTop: 14, paddingTop: 12, borderTop: '1px solid var(--border-base)' }}>
                <span style={{ fontSize: 11, color: 'var(--text-muted)', lineHeight: 1.5 }}>
                  Factors are generated using SHAP model explanations and describe why the XGBoost model produced this assessment — not why fraud occurred in reality. Relative impact bars show proportional contribution among displayed factors only.
                </span>
              </div>

            </div>
          </CardItem>
        )}

      </CardBody>
    </CardContainer>
  );
}
