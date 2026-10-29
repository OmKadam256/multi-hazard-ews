"use client";

import React, { useState } from "react";
import { TrendingUp, Calendar, CloudRain, Sun, Activity, BarChart2 } from "lucide-react";

interface HistoricalTrendsProps {
  basinName: string;
}

export const HistoricalTrendsChart: React.FC<HistoricalTrendsProps> = ({ basinName }) => {
  const [activeTab, setActiveTab] = useState<"rainfall" | "spei" | "leadtime">("rainfall");

  // 30 days of rainfall data (mm) leading up to current event
  const rainfallDays = [
    { day: "T-30", val: 2 }, { day: "T-28", val: 0 }, { day: "T-26", val: 5 }, { day: "T-24", val: 1 },
    { day: "T-22", val: 0 }, { day: "T-20", val: 8 }, { day: "T-18", val: 12 }, { day: "T-16", val: 4 },
    { day: "T-14", val: 0 }, { day: "T-12", val: 18 }, { day: "T-10", val: 24 }, { day: "T-8", val: 32 },
    { day: "T-6", val: 45 }, { day: "T-4", val: 68 }, { day: "T-2", val: 85 }, { day: "T-0", val: 112 }
  ];

  // 6 months of SPEI & NDVI anomalies
  const speiMonths = [
    { m: "M-6", spei: 0.4, ndvi: 0.65 },
    { m: "M-5", spei: 0.1, ndvi: 0.62 },
    { m: "M-4", spei: -0.5, ndvi: 0.58 },
    { m: "M-3", spei: -0.9, ndvi: 0.52 },
    { m: "M-2", spei: -1.4, ndvi: 0.46 },
    { m: "M-1", spei: -1.7, ndvi: 0.41 },
    { m: "Now", spei: -1.9, ndvi: 0.38 }
  ];

  // 72-hour forecast trajectory
  const leadSteps = [
    { step: "+0h", prob: 0.42, ci_low: 0.32, ci_high: 0.52 },
    { step: "+12h", prob: 0.56, ci_low: 0.44, ci_high: 0.68 },
    { step: "+24h", prob: 0.72, ci_low: 0.58, ci_high: 0.86 },
    { step: "+36h", prob: 0.84, ci_low: 0.70, ci_high: 0.98 },
    { step: "+48h", prob: 0.88, ci_low: 0.74, ci_high: 0.99 },
    { step: "+60h", prob: 0.79, ci_low: 0.65, ci_high: 0.93 },
    { step: "+72h", prob: 0.62, ci_low: 0.48, ci_high: 0.76 }
  ];

  return (
    <div className="glass-panel rounded-2xl p-6 shadow-2xl border border-slate-800 space-y-6">
      {/* Header and Tab Selector */}
      <div className="flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4 pb-4 border-b border-slate-800">
        <div>
          <div className="flex items-center gap-2">
            <TrendingUp className="h-5 w-5 text-cyan-400" />
            <h3 className="text-base font-bold text-white">Historical Dynamics & Forecast Trajectory</h3>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Temporal analysis window for <span className="text-cyan-300 font-semibold">{basinName}</span>
          </p>
        </div>

        <div className="flex items-center bg-slate-900 rounded-xl p-1 border border-slate-800 text-xs font-medium">
          <button
            onClick={() => setActiveTab("rainfall")}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              activeTab === "rainfall"
                ? "bg-cyan-500/20 text-cyan-300 border border-cyan-500/40 shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            30d Antecedent Rainfall
          </button>
          <button
            onClick={() => setActiveTab("spei")}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              activeTab === "spei"
                ? "bg-amber-500/20 text-amber-300 border border-amber-500/40 shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            6-Month SPEI / NDVI
          </button>
          <button
            onClick={() => setActiveTab("leadtime")}
            className={`px-3 py-1.5 rounded-lg transition-all ${
              activeTab === "leadtime"
                ? "bg-purple-500/20 text-purple-300 border border-purple-500/40 shadow-sm"
                : "text-slate-400 hover:text-white"
            }`}
          >
            72h Lead Time Curve
          </button>
        </div>
      </div>

      {/* Chart Canvas */}
      <div className="bg-slate-950/70 p-5 rounded-xl border border-slate-900">
        {activeTab === "rainfall" && (
          <div>
            <div className="flex items-center justify-between mb-4 text-xs font-mono">
              <span className="text-slate-400 flex items-center gap-1.5">
                <CloudRain className="h-4 w-4 text-cyan-400" />
                Daily Hyetograph (CHIRPS & ERA5-Land Reanalysis)
              </span>
              <span className="text-cyan-400 font-bold">
                Peak Inflow: 112 mm/day • 7d Cumulative: 309 mm
              </span>
            </div>

            <svg viewBox="0 0 600 200" className="w-full h-48">
              {/* Grid lines */}
              <line x1="40" y1="30" x2="580" y2="30" stroke="#1e293b" strokeDasharray="4 4" />
              <line x1="40" y1="80" x2="580" y2="80" stroke="#1e293b" strokeDasharray="4 4" />
              <line x1="40" y1="130" x2="580" y2="130" stroke="#1e293b" strokeDasharray="4 4" />
              <line x1="40" y1="170" x2="580" y2="170" stroke="#334155" />

              {/* Y Axis labels */}
              <text x="10" y="35" fill="#64748b" fontSize="10" fontFamily="monospace">120mm</text>
              <text x="10" y="85" fill="#64748b" fontSize="10" fontFamily="monospace">80mm</text>
              <text x="10" y="135" fill="#64748b" fontSize="10" fontFamily="monospace">40mm</text>

              {/* Rainfall Bars */}
              {rainfallDays.map((d, i) => {
                const barWidth = 22;
                const x = 50 + i * 33;
                const h = (d.val / 120) * 140;
                const y = 170 - h;
                return (
                  <g key={i}>
                    <rect
                      x={x}
                      y={y}
                      width={barWidth}
                      height={h}
                      rx="3"
                      fill={d.val > 60 ? "#ef4444" : d.val > 30 ? "#38bdf8" : "#0284c7"}
                      className="transition-all duration-300 hover:opacity-80"
                    />
                    <text
                      x={x + barWidth / 2}
                      y="185"
                      fill="#94a3b8"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {d.day}
                    </text>
                  </g>
                );
              })}
            </svg>
          </div>
        )}

        {activeTab === "spei" && (
          <div>
            <div className="flex items-center justify-between mb-4 text-xs font-mono">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Sun className="h-4 w-4 text-amber-400" />
                Standardized Precipitation-Evapotranspiration Index (SPEI-3) & NDVI
              </span>
              <span className="text-amber-400 font-bold">
                Current SPEI: -1.9 (Severe Drought Threshold: -1.5)
              </span>
            </div>

            <svg viewBox="0 0 600 200" className="w-full h-48">
              {/* Zero baseline */}
              <line x1="40" y1="90" x2="580" y2="90" stroke="#e2e8f0" strokeWidth="1.2" opacity="0.3" />
              <text x="10" y="93" fill="#94a3b8" fontSize="10" fontFamily="monospace">0.0</text>
              
              {/* Drought warning line (-1.5) */}
              <line x1="40" y1="140" x2="580" y2="140" stroke="#ef4444" strokeDasharray="4 4" strokeWidth="1.5" />
              <text x="10" y="143" fill="#ef4444" fontSize="10" fontFamily="monospace">-1.5</text>

              {/* SPEI Area & Line */}
              <path
                d="M 60 75 Q 140 85 220 110 T 380 145 Q 460 160 540 170"
                fill="none"
                stroke="#f59e0b"
                strokeWidth="3.5"
                strokeLinecap="round"
              />

              {/* NDVI Line */}
              <path
                d="M 60 50 Q 140 60 220 80 T 380 115 Q 460 135 540 145"
                fill="none"
                stroke="#10b981"
                strokeWidth="2.5"
                strokeDasharray="5 3"
                strokeLinecap="round"
              />

              {/* Data points */}
              {speiMonths.map((m, i) => {
                const x = 60 + i * 80;
                return (
                  <g key={i}>
                    <circle cx={x} cy={75 + i * 15} r="4" fill="#f59e0b" />
                    <text
                      x={x}
                      y="190"
                      fill="#94a3b8"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {m.m}
                    </text>
                  </g>
                );
              })}
            </svg>

            <div className="flex items-center justify-center gap-6 mt-2 text-xs font-mono text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-amber-500 rounded" /> SPEI-3 Index
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-emerald-500 rounded" /> MODIS NDVI Vegetation Vigor
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-red-500 rounded border-dashed" /> Severe Drought Benchmark (-1.5)
              </span>
            </div>
          </div>
        )}

        {activeTab === "leadtime" && (
          <div>
            <div className="flex items-center justify-between mb-4 text-xs font-mono">
              <span className="text-slate-400 flex items-center gap-1.5">
                <Activity className="h-4 w-4 text-purple-400" />
                Calibrated Probability Lead Trajectory with 95% Confidence Bounds
              </span>
              <span className="text-purple-400 font-bold">
                Peak Hazard Probability: 88.0% at +48h Horizon
              </span>
            </div>

            <svg viewBox="0 0 600 200" className="w-full h-48">
              {/* High risk threshold */}
              <line x1="40" y1="65" x2="580" y2="65" stroke="#ef4444" strokeDasharray="4 4" strokeWidth="1.2" />
              <text x="10" y="68" fill="#ef4444" fontSize="10" fontFamily="monospace">70%</text>

              {/* Uncertainty confidence envelope */}
              <polygon
                points="60,110 140,85 220,60 300,35 380,30 460,40 540,75 540,130 460,110 380,85 300,90 220,115 140,140 60,150"
                fill="rgba(168, 85, 247, 0.15)"
              />

              {/* Mean forecast trajectory line */}
              <path
                d="M 60 130 Q 140 105 220 80 T 380 45 Q 460 65 540 95"
                fill="none"
                stroke="#c084fc"
                strokeWidth="3.5"
                strokeLinecap="round"
              />

              {/* Points & Step labels */}
              {leadSteps.map((s, i) => {
                const x = 60 + i * 80;
                return (
                  <g key={i}>
                    <circle cx={x} cy={130 - (s.prob - 0.4) * 160} r="4" fill="#a855f7" />
                    <text
                      x={x}
                      y="190"
                      fill="#94a3b8"
                      fontSize="9"
                      fontFamily="monospace"
                      textAnchor="middle"
                    >
                      {s.step}
                    </text>
                  </g>
                );
              })}
            </svg>

            <div className="flex items-center justify-center gap-6 mt-2 text-xs font-mono text-slate-400">
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-0.5 bg-purple-400 rounded" /> Mean Calibrated Probability
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 bg-purple-500/20 rounded border border-purple-500/40" /> 95% Uncertainty Confidence Envelope (Homoscedastic σ)
              </span>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};
